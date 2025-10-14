#!/usr/bin/env python3
"""
Script to visualize TIFF files from data/dem_data with signed sqrt transform.
Shows side-by-side comparison of original and transformed 512x512 crops.

Usage:
    python visualize_sqrt.py              # Interactive mode
    python visualize_sqrt.py --batch N    # Process N images non-interactively
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import random
import argparse
from pathlib import Path

def signed_sqrt_transform(values):
    """Apply signed sqrt transform to elevation data"""
    return np.sign(values) * np.sqrt(np.abs(values))

def min_max_normalize(data):
    """Min-max normalize data to [0, 1] range"""
    return data

def get_random_crop(data, crop_size=512):
    """Extract a random 512x512 crop from the data"""
    h, w = data.shape
    if h < crop_size or w < crop_size:
        # If image is smaller than crop size, pad it
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='edge')
        h, w = data.shape
    
    # Random crop coordinates
    start_h = random.randint(0, h - crop_size)
    start_w = random.randint(0, w - crop_size)
    
    return data[start_h:start_h + crop_size, start_w:start_w + crop_size]

def load_and_process_tiff(tiff_path):
    """Load TIFF file and return processed data"""
    try:
        with rasterio.open(tiff_path) as src:
            data = src.read(1)  # Read first band
            
        # Handle nodata values
        if hasattr(src, 'nodata') and src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
            
        return data
    except Exception as e:
        print(f"Error loading {tiff_path}: {e}")
        return None

def visualize_comparison(original_crop, transformed_crop, tiff_name):
    """Create side-by-side visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original
    im1 = ax1.imshow(original_crop, cmap='terrain', aspect='equal')
    ax1.set_title(f'Original - {tiff_name}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot transformed
    im2 = ax2.imshow(transformed_crop, cmap='terrain', aspect='equal')
    ax2.set_title(f'Signed Sqrt Transform - {tiff_name}')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize TIFF files with signed sqrt transform')
    parser.add_argument('--batch', type=int, help='Process N images non-interactively')
    args = parser.parse_args()
    
    # Path to DEM data directory
    dem_data_path = Path("/mnt/ntfs2/shared/terrain-diffusion/data/dem_data")
    
    # Get all TIFF files
    tiff_files = list(dem_data_path.glob("*.tif"))
    
    if not tiff_files:
        print("No TIFF files found in data/dem_data directory!")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Process files in random order
    random.shuffle(tiff_files)
    
    # Limit number of files if batch mode
    if args.batch:
        tiff_files = tiff_files[:args.batch]
        print(f"Processing {len(tiff_files)} files in batch mode")
    
    for i, tiff_file in enumerate(tiff_files):
        print(f"\nProcessing {i+1}/{len(tiff_files)}: {tiff_file.name}")
        
        # Load TIFF data
        data = load_and_process_tiff(tiff_file)
        if data is None:
            continue
            
        # Get random 512x512 crop
        crop = get_random_crop(data, crop_size=4096)
        
        # Handle NaN values by replacing with mean
        if np.any(np.isnan(crop)):
            crop = np.nan_to_num(crop, nan=np.nanmean(crop))
        
        # Apply signed sqrt transform
        transformed_crop = signed_sqrt_transform(crop)
        
        # Min-max normalize both versions
        original_normalized = min_max_normalize(crop)
        transformed_normalized = min_max_normalize(transformed_crop)
        
        # Print some statistics
        print(f"  Original range: [{np.min(crop):.2f}, {np.max(crop):.2f}]")
        print(f"  Transformed range: [{np.min(transformed_crop):.2f}, {np.max(transformed_crop):.2f}]")
        
        # Visualize comparison
        visualize_comparison(original_normalized, transformed_normalized, tiff_file.name)

if __name__ == "__main__":
    main()
