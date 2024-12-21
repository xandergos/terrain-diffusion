import random
import click
import os
import tifffile as tiff
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from terrain_diffusion.data.laplacian_encoder import LaplacianPyramidEncoder
from terrain_diffusion.data.preprocessing.utils import preprocess_elevation


@click.command()
@click.option('-f', '--folder', required=True, help='Folder containing elevation .tiff/tif files')
@click.option('--base-resolution', default=240, help='Resolution input images are in meters')
@click.option('--target-resolution', default=480, help='Resolution output images should be in meters')
@click.option('--image-size', default=2048, help='Size of input images')
@click.option('--lapl-enc-resize', default=8, help='How much to downsample the input image for the low-res channel')
@click.option('--lapl-enc-sigma', default=5, help='Amount to blur the low-res channel')
@click.option('--file-limit', default=1000, help='Number of files to process')
def calculate_stats(folder, base_resolution, target_resolution, image_size, lapl_enc_resize, lapl_enc_sigma, file_limit):
    """
    Calculate mean and standard deviation statistics for high and low resolution channels.

    Uses online mean/std calculation to process elevation data through a Laplacian pyramid encoder
    and compute statistics for both high and low frequency components.

    Args:
        folder: Path to folder containing elevation .tiff/tif files
        base_resolution: Resolution of input images in meters
        target_resolution: Target resolution to process images at in meters
        lapl_enc_resize: Downsampling factor for low-res channel
        lapl_enc_sigma: Blur sigma for low-res channel
    """
    # Initialize running stats
    highres_mean = 0
    highres_m2 = 0  # For variance calculation
    highres_count = 0

    lowres_mean = 0 
    lowres_m2 = 0
    lowres_count = 0

    # Initialize encoder with dummy stats
    encoder = LaplacianPyramidEncoder([lapl_enc_resize], lapl_enc_sigma, [0, 0], [0.5, 0.5])

    files = os.listdir(folder)
    random.shuffle(files)
    files = files[:file_limit]
    for file in tqdm(files):
        if not file.endswith(('.tiff', '.tif')):
            continue
            
        img = tiff.imread(os.path.join(folder, file)).astype(np.float32)
        if len(img.shape) == 2 and np.any(img < 0):
            continue
        img = preprocess_elevation(img, image_size)
        
        downsample_factor = target_resolution // base_resolution
        img = F.adaptive_avg_pool2d(torch.from_numpy(img)[None], 
                                (img.shape[0] // downsample_factor, img.shape[1] // downsample_factor))
        
        encoded, downsampled_encoding = encoder.encode(img, return_downsampled=True)
        highfreq = encoded[:1].numpy().flatten().astype(np.float64)
        lowfreq = downsampled_encoding[-1].numpy().flatten().astype(np.float64)
        
        # Update high-res stats
        highres_count += len(highfreq)
        delta = highfreq - highres_mean
        highres_mean += np.sum(delta / highres_count)
        delta2 = highfreq - highres_mean
        highres_m2 += np.sum(delta * delta2)
        
        # Update low-res stats
        delta = lowfreq - lowres_mean
        lowres_count += len(lowfreq)
        lowres_mean += np.sum(delta / lowres_count)
        delta2 = lowfreq - lowres_mean
        lowres_m2 += np.sum(delta * delta2)

    highres_std = np.sqrt(highres_m2 / (highres_count - 1))
    lowres_std = np.sqrt(lowres_m2 / (lowres_count - 1))

    print(f"High-res channel stats:")
    print(f"Mean: {highres_mean:.2f}")
    print(f"Std:  {highres_std:.2f}")
    print(f"\nLow-res channel stats:")
    print(f"Mean: {lowres_mean:.2f}") 
    print(f"Std:  {lowres_std:.2f}")

if __name__ == "__main__":
    calculate_stats()
