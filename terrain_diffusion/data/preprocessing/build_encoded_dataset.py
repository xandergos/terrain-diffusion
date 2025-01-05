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
@click.option('--dataset', required=True, help='Path to base HDF5 dataset containing highfreq/lowfreq')
@click.option('--resolution', type=int, required=True, help='Resolution of the input images in meters')
@click.option('--encoder', 'encoder_model_path', required=True, help='Path to encoder model checkpoint')
@click.option('--use-fp16', is_flag=True, help='Use FP16 for encoding', default=False)
@click.option('--compile-model', is_flag=True, help='Compile the model', default=False)
@click.option('--overwrite', is_flag=True, help='Overwrite existing datasets', default=False)
def process_encoded_dataset(dataset, resolution, encoder_model_path, use_fp16, compile_model, overwrite):
    """
    Add latent encodings to an existing HDF5 dataset containing high/low frequency components.
    
    Takes a base HDF5 dataset containing highfreq/lowfreq components and adds corresponding latent
    encodings using the specified encoder model.
    """
    device = 'cuda'
    model = EDMAutoencoder.from_pretrained(encoder_model_path)
    model = model.encoder
    model.to(device)

    if compile_model:
        model = torch.compile(model)

    # Open base dataset in append mode instead of reading two files
    with h5py.File(dataset, 'a') as f:
        base_keys = list(f.keys())
        base_keys = [k for k in base_keys if k.endswith('$residual')]
        base_keys = [k for k in base_keys if str(f[k].attrs['resolution']) == str(resolution)]
        
        for key in tqdm(base_keys):
            latent_key = key.replace('$residual', '$latent')
            if latent_key in f and not overwrite:
                continue
            
            # Parse components from key
            residual = torch.from_numpy(f[key][:])[None]
            residual = (residual - f[key].attrs['residual_mean']) / f[key].attrs['residual_std']
            
            transformed_latent = []
            for horiz_flip in [False, True]:
                for rot_deg in [0, 90, 180, 270]:
                    residual_transformed = residual
                    if horiz_flip:
                        residual_transformed = torch.flip(residual_transformed, dims=[-1])
                    if rot_deg != 0:
                        residual_transformed = torch.rot90(residual_transformed, k=rot_deg // 90, dims=[-2, -1])
                    
                    with torch.no_grad():
                        input = residual_transformed[None].to(device=device)
                        with torch.autocast(device_type=device, dtype=torch.float16 if use_fp16 else torch.float32):
                            latent_residual = model(input, noise_labels=None, conditional_inputs=[]).cpu()[0]
                    transformed_latent.append(latent_residual.numpy())
            
            transformed_latent = np.stack(transformed_latent)
            
            if latent_key in f:
                del f[latent_key]
            dset = f.create_dataset(latent_key, data=transformed_latent, compression='lzf')
            dset.attrs.update(f[key].attrs)
            dset.attrs['data_type'] = 'latent'
            
        # Calculate per-channel latent mean and std using Welford's algorithm
        print("Calculating per-channel latent mean and std using Welford's algorithm...")
        latent_keys = [k for k in f.keys() if k.endswith('$latent') and f[k].attrs['resolution'] == resolution]
        if not latent_keys:
            return
        
        # Initialize statistics arrays using first latent to get number of channels
        num_channels = f[latent_keys[0]].shape[-1]
        means = np.zeros(num_channels)
        M2s = np.zeros(num_channels)
        count = 0
        
        for key in tqdm(latent_keys, desc="Computing statistics"):
            # Data shape is (num_transforms, height, width, channels)
            data = f[key][:]
            batch_size = data.shape[0] * data.shape[1] * data.shape[2]
            # Reshape to (N, channels)
            data = data.reshape(-1, num_channels)
            
            count += batch_size
            delta = data - means
            means += np.sum(delta, axis=0) / count
            delta2 = data - means
            M2s += np.sum(delta * delta2, axis=0)
        
        latent_means = means
        latent_stds = np.sqrt(M2s / (count - 1))
        
        print(f"Latent means: {latent_means}")
        print(f"Latent stds: {latent_stds}")
        
        print("Adding latent statistics to datasets...")
        for key in tqdm(latent_keys, desc="Updating attributes"):
            f[key].attrs['latent_stds'] = latent_stds
            f[key].attrs['latent_means'] = latent_means
        
        # Store in global attributes
        f.attrs[f'latent_stds:{resolution}'] = latent_stds
        f.attrs[f'latent_means:{resolution}'] = latent_means

        print(f"Finished processing. Total datasets in file: {len(f.keys())}")


if __name__ == '__main__':
    process_encoded_dataset()