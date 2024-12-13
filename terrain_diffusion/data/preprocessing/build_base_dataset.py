"""
Takes a folder of elevation .tiff files and preprocesses them for training a neural network that does not require latents.
This is primarily used for training autoencoders or generative models that do not require latents.
"""

import os
import tifffile as tiff
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import click
from terrain_diffusion.data.laplacian_encoder import LaplacianPyramidEncoder

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
@click.option('--target-resolution', default=240, help='Resolution output images should be in meters')
@click.option('--lapl-enc-resize', default=8, help='How much to downsample the input image for the low-res channel')
@click.option('--lapl-enc-sigma', default=5, help='Amount to blur the low-res channel')
@click.option('--lapl-enc-lowres-mean', default=-2651, help='Mean value for encoder normalization (low res)')
@click.option('--lapl-enc-lowres-std', default=2420, help='Std value for encoder normalization (low res)')
@click.option('--lapl-enc-highres-mean', default=0, help='Mean value for encoder normalization (high res)')
@click.option('--lapl-enc-highres-std', default=160, help='Std value for encoder normalization (high res)')
@click.option('--output-file', default='dataset_base.h5', help='Output HDF5 filename')
@click.option('--elevation-folder', required=True, help='Folder containing elevation .tiff/tif files')
def process_dataset(base_resolution, target_resolution, lapl_enc_resize, lapl_enc_sigma, 
                   lapl_enc_lowres_mean, lapl_enc_lowres_std, lapl_enc_highres_mean, 
                   lapl_enc_highres_std, output_file, elevation_folder):
    """
    Process elevation dataset into HDF5 format with Laplacian pyramid encoding.
    
    Processes .tiff files from elevation-folder at input-resolution, encoding them using
    a Laplacian pyramid encoder. Creates datasets at specified resolution in meters.
    
    Input .tiff files should contain elevation data in any resolution, with 1 channel: The elevation
    
    Output HDF5 file will contain the following datasets for each input file:
    - {filename}_{target_resolution}m_highfreq: High frequency residual (1 channel)
    - {filename}_{target_resolution}m_lowfreq: Low frequency residual (1 channel)
    """
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Appending to it.")
    else:
        print(f"{output_file} does not exist. Creating it and building datasets.")
        
    with h5py.File(output_file, 'a') as f:
        encoder = LaplacianPyramidEncoder([lapl_enc_resize], lapl_enc_sigma, 
                                        [lapl_enc_highres_mean, lapl_enc_lowres_mean], 
                                        [lapl_enc_highres_std, lapl_enc_lowres_std])
        
        for file in tqdm(os.listdir(elevation_folder)):
            img = tiff.imread(os.path.join(elevation_folder, file)).astype(np.float32)
            img = preprocess_elevation(img)
            
            # Resize to target resolution
            downsample_factor = base_resolution // target_resolution
            img = F.adaptive_avg_pool2d(torch.from_numpy(img)[None], 
                                      (img.shape[0] // downsample_factor, 
                                       img.shape[1] // downsample_factor))
            
            # Use Laplacian encoder
            encoded, downsampled_encoding = encoder.encode(img, return_downsampled=True)
            
            # Calculate percent of land
            pct_land = torch.mean((img > 0).float()).item()
            
            # Save to HDF5
            dset = f.create_dataset(f'{file}_{target_resolution}m_highfreq', 
                                  data=encoded[:1].numpy(), compression='lzf')
            dset.attrs['pct_land'] = pct_land
            dset.attrs['label'] = f'{target_resolution}m_highfreq'
            
            dset = f.create_dataset(f'{file}_{target_resolution}m_lowfreq', 
                                  data=downsampled_encoding[-1].numpy(), compression='lzf')
            dset.attrs['pct_land'] = pct_land
            dset.attrs['label'] = f'{target_resolution}m_lowfreq'

        print(f"Finished processing. Total datasets in file: {len(f.keys())}")

if __name__ == '__main__':
    process_dataset()