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
@click.option('-o', '--output-file', type=str, default='dataset.h5', help='Path to the output HDF5 file')
@click.option('--num-workers', type=int, default=mp.cpu_count()-1, help='Number of parallel workers for processing')
@click.option('--overwrite', is_flag=True, help='Overwrite existing datasets in the output file')
@click.option('--prefetch', type=int, default=2, help='Number of prefetch factor for the dataloader')
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
    output_file,
    num_workers,
    overwrite,
    prefetch
):
    """
    Process elevation dataset into encoded HDF5 format.
    
    Processes .tiff files from elevation-folder at input-resolution, encoding them using
    a Laplacian pyramid encoder and a learned encoder model. Creates datasets at specified
    resolution in meters.
    
    Input .tiff files should contain elevation data in any resolution, with 1 channel: The elevation
    
    Output HDF5 file will contain the following datasets for each input file:
    - {filename}_{target_resolution}m_highfreq: High frequency residual (1 channel)
    - {filename}_{target_resolution}m_lowfreq: Low frequency residual (1 channel)
    - {filename}_{target_resolution}m_latent: Encoded latents from the encoder model (encoder output channels)
    """
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Appending to it.")
    else:
        print(f"{output_file} does not exist. Creating it and building datasets.")
    with h5py.File(output_file, 'a') as f:
        # Filter out files that are already in the HDF5 file
        if not overwrite:
            skip_chunk_ids = set()
            for key in f.keys():
                if f[key].attrs['resolution'] == resolution:
                    chunk_id = f[key].attrs['chunk_id']
                    skip_chunk_ids.add(chunk_id)
            print(f"Skipping {len(skip_chunk_ids)} existing chunk ids")
        else:
            for key in f.keys():
                if f[key].attrs['resolution'] == resolution:
                    del f[key]
            skip_chunk_ids = set()
        
        dataset = ElevationDataset(
            highres_elevation_folder,
            lowres_elevation_folder,
            highres_size,
            lowres_size,
            lowres_sigma,
            num_chunks,
            landcover_folder,
            watercover_folder,
            skip_chunk_ids
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch)
        
        # Process files in parallel with initial assumption
        for chunks_data in tqdm(dataloader, desc="Saving datasets"):
            for chunk in chunks_data:
                # Save to HDF5
                for data_type, data in [
                    ('residual', chunk['residual']),
                    ('lowfreq', chunk['lowfreq']),
                    ('landcover', chunk['landcover']),
                    ('watercover', chunk['watercover'])
                ]:
                    if data is None:
                        continue
                    dset_name = f"{chunk['chunk_id']}${resolution}m${chunk['subchunk_id']}${data_type}"
                    if dset_name in f:
                        del f[dset_name]
                    dset = f.create_dataset(dset_name, data=data.numpy(), compression='lzf')
                    dset.attrs.update({
                        'pct_land': chunk['pct_land'],
                        'resolution': resolution,
                        'data_type': data_type,
                        'chunk_id': chunk['chunk_id'],
                        'subchunk_id': chunk['subchunk_id']
                    })
                    f.flush()
        
        # Calculate std using saved residual datasets
        print("Calculating residual mean and std using Welford's algorithm...")
        mean = 0.0
        M2 = 0.0
        count = 0
        
        for key in tqdm(f.keys()):
            if 'residual' in key and f[key].attrs['resolution'] == resolution:
                data = f[key][:].flatten()
                count += len(data)
                delta = data - mean
                mean += np.sum(delta) / count
                delta2 = data - mean
                M2 += np.sum(delta * delta2)
        
        residual_mean = mean
        residual_std = np.sqrt(M2 / (count - 1))
        print(f"Residual mean: {residual_mean}")
        print(f"Residual std: {residual_std}")
        
        print("Adding residual std to dataset...")
        for key in tqdm(f.keys()):
            if f[key].attrs['resolution'] == resolution and 'residual' in key:
                f[key].attrs['residual_std'] = residual_std
                f[key].attrs['residual_mean'] = residual_mean
            f.attrs[f'residual_std:{resolution}'] = residual_std
            f.attrs[f'residual_mean:{resolution}'] = residual_mean
        
        print(f"Finished processing. Total datasets in file: {len(f.keys())}")


if __name__ == '__main__':
    process_base_dataset()