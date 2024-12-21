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
from terrain_diffusion.data.laplacian_encoder import LaplacianPyramidEncoder
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
import multiprocessing as mp
from functools import partial
from terrain_diffusion.data.preprocessing.utils import ElevationDataset, process_single_file_base

@click.command()
@click.option('--base-resolution', default=240, help='Resolution input images are in meters')
@click.option('--target-resolution', default=480, help='Resolution output images should be in meters')
@click.option('--num-chunks', default=2, help='Number of chunks to write to HDF5 file (along each axis)')
@click.option('--image-size', default=4096, help='Size of the input image')
@click.option('--lapl-enc-resize', default=8, help='How much to downsample the input image for the low-res channel')
@click.option('--lapl-enc-sigma', default=5, help='Amount to blur the low-res channel')
@click.option('--lapl-enc-lowres-mean', default=-2651, help='Mean value for encoder normalization (low res)')
@click.option('--lapl-enc-lowres-std', default=2420, help='Std value for encoder normalization (low res)')
@click.option('--lapl-enc-highres-mean', default=0, help='Mean value for encoder normalization (high res)')
@click.option('--lapl-enc-highres-std', default=160, help='Std value for encoder normalization (high res)')
@click.option('--output-file', default='dataset_encoded.h5', help='Output HDF5 filename')
@click.option('--elevation-folder', required=True, help='Folder containing elevation .tiff/tif files')
@click.option('--encoder-model-path', required=True, help='Path to encoder model checkpoint')
@click.option('--use-fp16', is_flag=True, help='Use FP16 for encoding', default=False)
@click.option('--compile-model', is_flag=True, help='Compile the model', default=False)
@click.option('--num-workers', type=int, default=None, help='Number of workers to use for encoding')
@click.option('--overwrite', is_flag=True, help='Overwrite existing datasets', default=False)
def process_encoded_dataset(base_resolution, target_resolution, num_chunks, image_size, lapl_enc_resize, lapl_enc_sigma, lapl_enc_lowres_mean, lapl_enc_lowres_std, 
                   lapl_enc_highres_mean, lapl_enc_highres_std, output_file, elevation_folder, encoder_model_path, use_fp16, compile_model, num_workers, overwrite):
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
    device = 'cuda'

    model = EDMAutoencoder.from_pretrained(encoder_model_path)
    model = model.encoder
    model.to(device)

    if compile_model:
        model = torch.compile(model)

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Appending to it.")
    else:
        print(f"{output_file} does not exist. Creating it and building datasets.")
    with h5py.File(output_file, 'a') as f:
        encoder = LaplacianPyramidEncoder([lapl_enc_resize], lapl_enc_sigma, 
                                        [lapl_enc_highres_mean, lapl_enc_lowres_mean], [lapl_enc_highres_std, lapl_enc_lowres_std])
        
        files = os.listdir(elevation_folder)
        
        # Filter out files that are already in the HDF5 file
        if not overwrite:
            existing_files = set()
            for key in f.keys():
                if f[key].attrs['resolution'] == target_resolution:
                    filename = f[key].attrs['filename']
                    existing_files.add(filename)
            
            files = [f for f in files if f not in existing_files]
            if len(files) == 0:
                print("All files have already been processed. Exiting.")
                return
            print(f"Processing {len(files)} new files...")
        
        dataset = ElevationDataset(files, elevation_folder, image_size, base_resolution, target_resolution, num_chunks, encoder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)
        
        # Process files in parallel
        for chunks_data in tqdm(dataloader, total=len(files)):
            for chunk in chunks_data:
                # Model inference on main thread
                transformed_latent = []
                highfreq = chunk['highfreq']
                
                for horiz_flip in [False, True]:
                    for rot_deg in [0, 90, 180, 270]:
                        highfreq_transformed = highfreq
                        if horiz_flip:
                            highfreq_transformed = torch.flip(highfreq_transformed, dims=[-1])
                        if rot_deg != 0:
                            highfreq_transformed = torch.rot90(highfreq_transformed, k=rot_deg // 90, dims=[-2, -1])
                        
                        with torch.no_grad():
                            input = highfreq_transformed[None].to(device=device) / 0.5
                            with torch.autocast(device_type=device, dtype=torch.float16 if use_fp16 else torch.float32):
                                latent_highfreq = model(input, noise_labels=None, conditional_inputs=[]).cpu()[0]
                        transformed_latent.append(latent_highfreq.numpy())
                
                transformed_latent = np.stack(transformed_latent)
                
                # Save to HDF5
                for data_type, data in [
                    ('highfreq', chunk['highfreq'].numpy()),
                    ('lowfreq', chunk['lowfreq'].numpy()),
                    ('latent', transformed_latent)
                ]:
                    dset_name = f"{chunk['filename']}${target_resolution}m${chunk['chunk_id']}${data_type}"
                    dset = f.create_dataset(dset_name, data=data, compression='lzf')
                    dset.attrs.update({
                        'pct_land': chunk['pct_land'],
                        'resolution': target_resolution,
                        'data_type': data_type,
                        'filename': chunk['filename'],
                        'chunk_id': chunk['chunk_id']
                    })
        
        print(f"Finished processing. Total datasets in file: {len(f.keys())}")


if __name__ == '__main__':
    process_encoded_dataset()