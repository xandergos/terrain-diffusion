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
from terrain_diffusion.training.diffusion.unet import EDMUnet2D

def preprocess_elevation(img):
    ocean = torch.from_numpy(img[:, :, 1])
    land = torch.from_numpy(img[:, :, 0])

    land = F.interpolate(land[None, None], (4096, 4096), mode='area')[0, 0]
    ocean = F.interpolate(ocean[None, None], (4096, 4096), mode='area')[0, 0]
    ocean = torch.minimum(ocean, torch.tensor(-1.0))

    land_mask = land > 0
    
    ocean = F.adaptive_avg_pool2d(ocean[None], (256, 256))
    ocean = F.interpolate(ocean[None], (4096, 4096), mode='bicubic')[0, 0]
    ocean = torch.minimum(ocean, torch.tensor(-1.0))
    
    img = land * land_mask.float() + ocean * (1 - land_mask.float())
    
    return img.numpy()

@click.command()
@click.option('--base-resolution', default=240, help='Resolution input images are in meters')
@click.option('--target-resolution', default=480, help='Resolution output images should be in meters')
@click.option('--chunk-size', default=2048, help='Size of chunks to write to HDF5 file')
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
def process_encoded_dataset(base_resolution, target_resolution, chunk_size, lapl_enc_resize, lapl_enc_sigma, lapl_enc_lowres_mean, lapl_enc_lowres_std, 
                   lapl_enc_highres_mean, lapl_enc_highres_std, output_file, elevation_folder, encoder_model_path, use_fp16, compile_model):
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

    model = EDMUnet2D.from_pretrained(encoder_model_path)
    model.to(device)

    if use_fp16:
        model.half()
    if compile_model:
        model = torch.compile(model)

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Appending to it.")
    else:
        print(f"{output_file} does not exist. Creating it and building datasets.")
    with h5py.File(output_file, 'a') as f:
        encoder = LaplacianPyramidEncoder([lapl_enc_resize], lapl_enc_sigma, 
                                        [lapl_enc_highres_mean, lapl_enc_lowres_mean], [lapl_enc_highres_std, lapl_enc_lowres_std])
        for file in tqdm(os.listdir(elevation_folder)):
            img = tiff.imread(os.path.join(elevation_folder, file)).astype(np.float32)
            img = preprocess_elevation(img)
            
            downsample_factor = target_resolution // base_resolution
            img = F.adaptive_avg_pool2d(torch.from_numpy(img)[None], (img.shape[0] // downsample_factor, img.shape[1] // downsample_factor))
            
            # Calculate number of chunks needed
            h, w = img.shape[-2:]
            num_chunks_h = (h + chunk_size - 1) // chunk_size
            num_chunks_w = (w + chunk_size - 1) // chunk_size
            
            # Process each chunk
            for chunk_h in range(num_chunks_h):
                for chunk_w in range(num_chunks_w):
                    start_h = chunk_h * chunk_size
                    start_w = chunk_w * chunk_size
                    end_h = min(start_h + chunk_size, h)
                    end_w = min(start_w + chunk_size, w)
                    
                    img_chunk = img[..., start_h:end_h, start_w:end_w]
                    pct_land = torch.mean((img_chunk > 0).float()).item()
                    
                    encoded, downsampled_encoding = encoder.encode(img_chunk, return_downsampled=True)
                    highfreq = encoded[:1]
                    lowfreq = downsampled_encoding[-1]
                    transformed_latent = []
                    
                    for horiz_flip in [False, True]:
                        for rot_deg in [0, 90, 180, 270]:
                            # Apply horizontal flip if needed
                            highfreq_transformed = highfreq
                            if horiz_flip:
                                highfreq_transformed = torch.flip(highfreq_transformed, dims=[-1])
                                
                            # Apply rotation if needed 
                            if rot_deg != 0:
                                highfreq_transformed = torch.rot90(highfreq_transformed, k=rot_deg // 90, dims=[-2, -1])
                            
                            with torch.no_grad():
                                if use_fp16:
                                    input = highfreq_transformed[None].to(device=device, dtype=torch.float16) / 0.5
                                else:
                                    input = highfreq_transformed[None].to(device=device) / 0.5
                                latent_highfreq = model(input, noise_labels=None, conditional_inputs=None).cpu()[0]
                            
                            transformed_latent.append(latent_highfreq.numpy())
                            
                    transformed_latent = np.stack(transformed_latent)
                    
                    chunk_id = f'chunk_{chunk_h}_{chunk_w}'
                    dset = f.create_dataset(f'{file}${target_resolution}m${chunk_id}$highfreq', data=highfreq.numpy(), compression='lzf')
                    dset.attrs['pct_land'] = pct_land
                    dset.attrs['resolution'] = target_resolution
                    dset.attrs['data_type'] = 'highfreq'
                    dset.attrs['filename'] = file
                    dset.attrs['chunk_id'] = chunk_id
                    
                    dset = f.create_dataset(f'{file}${target_resolution}m${chunk_id}$lowfreq', data=lowfreq.numpy(), compression='lzf')
                    dset.attrs['pct_land'] = pct_land
                    dset.attrs['resolution'] = target_resolution
                    dset.attrs['data_type'] = 'highfreq'
                    dset.attrs['filename'] = file
                    dset.attrs['chunk_id'] = chunk_id
                    
                    dset = f.create_dataset(f'{file}${target_resolution}m${chunk_id}$latent', data=transformed_latent, compression='lzf')
                    dset.attrs['pct_land'] = pct_land
                    dset.attrs['resolution'] = target_resolution
                    dset.attrs['data_type'] = 'highfreq'
                    dset.attrs['filename'] = file
                    dset.attrs['chunk_id'] = chunk_id

        print(f"Finished processing. Total datasets in file: {len(f.keys())}")
        
if __name__ == '__main__':
    process_encoded_dataset()