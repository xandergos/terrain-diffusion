"""
Takes a folder of elevation .tiff files and preprocesses them for training a diffusion model.
The output can be used to train superresolution or base diffusion models.

Requires a pre-trained encoder model.
"""

import asyncio
import os
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import click
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
import multiprocessing as mp
from functools import partial
from terrain_diffusion.data.preprocessing.utils import ElevationDataset, process_single_file_base
from terrain_diffusion.data.preprocessing.calculate_stds import calculate_stats_welford

@click.command()
@click.option('--highres-elevation-folder', type=str, required=True, help='Path to the folder containing high-resolution elevation files')
@click.option('--lowres-elevation-folder', type=str, required=True, help='Path to the folder containing low-resolution elevation files')
@click.option('--highres-size', type=int, default=4096, help='Size of the high-resolution images')
@click.option('--lowres-size', type=int, default=128, help='Size of the low-resolution images')
@click.option('--lowres-sigma', type=float, default=5.0, help='Sigma for Gaussian smoothing of low-resolution images')
@click.option('--resolution', type=int, default=90, help='Resolution of the input images in meters. Only used for labeling.')
@click.option('--num-chunks', type=int, default=1, help='Number of chunks to divide the image into for processing')
@click.option('--landcover-folder', type=str, default=None, help='Path to the folder containing land cover data (optional)')
@click.option('--watercover-folder', type=str, default=None, help='Path to the folder containing water cover data (optional)')
@click.option('--koppen-geiger-folder', type=str, default=None, help='Path to the folder containing koppen geiger data (optional)')
@click.option('--climate-folder', type=str, default=None, help='Path to the folder containing climate data (optional)')
@click.option('-o', '--output-file', type=str, default='dataset.h5', help='Path to the output HDF5 file')
@click.option('--num-workers', type=int, default=mp.cpu_count()-1, help='Number of parallel workers for processing')
@click.option('--overwrite', is_flag=True, help='Overwrite existing datasets in the output file')
@click.option('--prefetch', type=int, default=2, help='Number of prefetch factor for the dataloader')
@click.option('--edge-margin', type=int, default=0, help='Number of pixels to remove from the edges of the lowres image, automatically scaled up for the highres image')
@click.option('--min-stat-landcover-pct', type=float, default=0.1, help='Minimum percentage of landcover to include in the statistics calculation')
def process_base_dataset(
    highres_elevation_folder,
    lowres_elevation_folder,
    highres_size,
    lowres_size,
    lowres_sigma,
    resolution,
    num_chunks,
    landcover_folder,
    watercover_folder,
    koppen_geiger_folder,
    climate_folder,
    output_file,
    num_workers,
    overwrite,
    prefetch,
    edge_margin,
    min_stat_landcover_pct
):
    """
    Process elevation dataset into encoded HDF5 format.
    
    Processes .tiff files from elevation-folder at input-resolution, encoding them using
    a Laplacian pyramid encoder and a learned encoder model. Creates datasets at specified
    resolution in meters.
    
    Input .tiff files should contain elevation data in any resolution, with 1 channel: The elevation
    
    Output HDF5 file will contain datasets organized in groups:
    {resolution}/{chunk_id}/{subchunk_id}/{data_type}
    Where data_type is one of: residual, lowfreq, landcover, watercover, koppen_geiger, climate
    """
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Appending to it.")
    else:
        print(f"{output_file} does not exist. Creating it and building datasets.")
    
    with h5py.File(output_file, 'a') as f:
        # Create resolution group if it doesn't exist
        res_group = f.require_group(f"{resolution}")
        
        # Filter out files that are already in the HDF5 file
        if not overwrite:
            skip_chunk_ids = set()
            for chunk_id in res_group.keys():
                skip_chunk_ids.add(chunk_id)
            print(f"Skipping {len(skip_chunk_ids)} existing chunk ids")
        else:
            # Instead of deleting the entire resolution group, selectively delete datasets
            skip_chunk_ids = set()
            if str(resolution) in f:
                for chunk_id in res_group.keys():
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group.keys():
                        subchunk_group = chunk_group[subchunk_id]
                        # Delete only non-latent datasets
                        for data_type in ['residual', 'lowfreq', 'landcover', 'watercover', 
                                          'climate', 'koppen_geiger']:
                            if data_type in subchunk_group:
                                del subchunk_group[data_type]
        
        dataset = ElevationDataset(
            highres_elevation_folder,
            lowres_elevation_folder,
            resolution,
            highres_size,
            lowres_size,
            lowres_sigma,
            num_chunks,
            landcover_folder,
            watercover_folder,
            koppen_geiger_folder,
            climate_folder,
            skip_chunk_ids,
            edge_margin
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch)
        
        # Process files in parallel with hierarchical storage
        for chunks_data in tqdm(dataloader, desc="Saving datasets"):
            for chunk in chunks_data:
                chunk_group = res_group.require_group(chunk['chunk_id'])
                subchunk_group = chunk_group.require_group(chunk['subchunk_id'])
                
                # Save to HDF5
                for data_type, data in [
                    ('residual', chunk['residual']),
                    ('lowfreq', chunk['lowfreq']),
                    ('landcover', chunk['landcover']),
                    ('landcover_mini', chunk['landcover_mini']),
                    ('watercover', chunk['watercover']),
                    ('koppen_geiger', chunk['koppen_geiger']),
                    ('climate', chunk['climate'])
                ]:
                    if data_type in ['residual', 'landcover', 'watercover']:
                        chunk_shape = (128, 128)
                    elif data_type in ['lowfreq', 'koppen_geiger', 'landcover_mini']:
                        chunk_shape = (32, 32)
                    elif data_type == 'climate':
                        chunk_shape = (1, 32, 32)
                    
                    if data is None:
                        continue
                    dset = subchunk_group.create_dataset(data_type, data=data.numpy(), compression='lzf', chunks=chunk_shape)
                    dset.attrs.update({
                        'pct_land': chunk['pct_land'],
                        'resolution': resolution,
                        'data_type': data_type,
                        'chunk_id': chunk['chunk_id'],
                        'subchunk_id': chunk['subchunk_id']
                    })
                    f.flush()
        
        # Calculate stats for residual and climate datasets
        datasets_to_process = ['residual', 'climate']
        for dataset_name in datasets_to_process:
            means, stds = calculate_stats_welford(res_group, dataset_name, min_stat_landcover_pct)
            
            print(f"{dataset_name} mean: {means}")
            print(f"{dataset_name} std: {stds}")
            
            # Update attributes in hierarchical structure
            res_group.attrs[f'{dataset_name}_std'] = stds
            res_group.attrs[f'{dataset_name}_mean'] = means

        print(f"Finished processing. Total chunks in resolution {resolution}: {len(res_group.keys())}")

if __name__ == '__main__':
    process_base_dataset()