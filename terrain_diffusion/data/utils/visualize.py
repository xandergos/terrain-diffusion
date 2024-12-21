import random
import click
import os
import tifffile as tiff
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from terrain_diffusion.data.laplacian_encoder import LaplacianPyramidEncoder
from terrain_diffusion.data.preprocessing.build_base_dataset import preprocess_elevation
import matplotlib.pyplot as plt

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
    # Initialize encoder with dummy stats
    encoder = LaplacianPyramidEncoder([lapl_enc_resize], lapl_enc_sigma, [0, 0], [0.5, 0.5])

    files = os.listdir(folder)
    random.shuffle(files)
    files = files[:file_limit]
    for file in tqdm(files):
        if not file.endswith(('.tiff', '.tif')):
            continue
            
        img = tiff.imread(os.path.join(folder, file)).astype(np.float32)
        if np.any(img < 0):
            continue
        img = preprocess_elevation(img, image_size)
        
        downsample_factor = target_resolution // base_resolution
        img = F.adaptive_avg_pool2d(torch.from_numpy(img)[None], 
                                (img.shape[0] // downsample_factor, img.shape[1] // downsample_factor))
        
        encoded, downsampled_encoding = encoder.encode(img, return_downsampled=True)
        highfreq = encoded[0].numpy()
        lowfreq = downsampled_encoding[-1].numpy()[0]
        
        plt.imshow(highfreq, cmap='gray')
        plt.show()

if __name__ == "__main__":
    calculate_stats()
