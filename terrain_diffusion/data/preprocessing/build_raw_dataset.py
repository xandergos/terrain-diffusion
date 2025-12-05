"""
Takes a folder of elevation .tiff files, resizes them to 2048x2048, and saves them to an HDF5 file.
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import click
import rasterio


def resize_bilinear(data: np.ndarray, size: int) -> np.ndarray:
    """Resize 2D array to size x size using bilinear interpolation."""
    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)
    return resized.squeeze().numpy()


@click.command()
@click.option('--elevation-folder', type=str, required=True, help='Path to the folder containing elevation .tiff files')
@click.option('--output-size', type=int, default=2048, help='Output size for resized images')
@click.option('-o', '--output-file', type=str, default='raw_dataset.h5', help='Path to the output HDF5 file')
@click.option('--overwrite', is_flag=True, help='Overwrite existing datasets in the output file')
@click.option('--limit', type=int, default=None, help='Limit number of files to process')
@click.option('--seed', type=int, default=42, help='Random seed for shuffling files')
def process_raw_dataset(elevation_folder, output_size, output_file, overwrite, limit, seed):
    """
    Process elevation .tiff files into HDF5 format.
    
    Resizes each .tiff file to output_size x output_size using bilinear interpolation
    and stores them in a flat HDF5 file with dataset names matching file names.
    """
    tiff_files = sorted(glob.glob(os.path.join(elevation_folder, '*.tiff')) + 
                        glob.glob(os.path.join(elevation_folder, '*.tif')))
    
    if not tiff_files:
        print(f"No .tiff files found in {elevation_folder}")
        return
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(tiff_files)
    
    # Apply limit
    if limit is not None:
        tiff_files = tiff_files[:limit]
    
    print(f"Processing {len(tiff_files)} .tiff files")
    
    with h5py.File(output_file, 'a') as f:
        for tiff_path in tqdm(tiff_files, desc="Processing files"):
            filename = os.path.splitext(os.path.basename(tiff_path))[0]
            
            if filename in f and not overwrite:
                print(f"Skipping {filename} (already exists)")
                continue
            
            if filename in f and overwrite:
                del f[filename]
            
            with rasterio.open(tiff_path) as src:
                data = src.read(1).astype(np.float32)
                data[data == 0.0] = np.nan
            
            # Calculate pct_land before resizing (based on NaNs)
            pct_land = 1.0 - (np.isnan(data).sum() / data.size)
            
            # Resize to output size
            resized = resize_bilinear(data, output_size)
            
            # Convert to int16, NaN becomes min int16 value
            nan_mask = np.isnan(resized)
            resized[nan_mask] = 0  # Temporary value for casting
            resized_int16 = resized.astype(np.int16)
            resized_int16[nan_mask] = np.iinfo(np.int16).min
            
            # Save to HDF5
            dset = f.create_dataset(filename, data=resized_int16, compression='lzf', chunks=(256, 256))
            dset.attrs['pct_land'] = pct_land
            f.flush()
            
            # Delete the source file
            # os.remove(tiff_path)
    
    print(f"Finished processing {len(tiff_files)} files into {output_file}")


if __name__ == '__main__':
    process_raw_dataset()
