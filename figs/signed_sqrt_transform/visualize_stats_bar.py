#!/usr/bin/env python3
"""
Script to visualize the effect of signed-sqrt transform on variance by elevation.
Loads 512x512 terrain samples from dataset.h5, performs laplacian decoding to get full images,
bins elevations into 50 buckets based on raw elevation, and creates a bar chart showing
median variance for raw and transformed elevation in each bucket.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/ntfs2/shared/terrain-diffusion')
from terrain_diffusion.data.laplacian_encoder import laplacian_decode

GAMMA = 0.5

def signed_sqrt_transform(values):
    """Apply signed sqrt transform to elevation data"""
    return np.sign(values) * np.abs(values)**GAMMA

def inverse_signed_sqrt_transform(values):
    """Invert the signed sqrt transform to get back original elevation"""
    return np.sign(values) * np.abs(values)**(1/GAMMA)

def load_samples_from_h5(h5_file, num_samples=100, crop_size=512, min_pct_land=0.8):
    """Load random samples from the HDF5 dataset and reconstruct full terrain"""
    samples_transformed = []  # Data as stored (already in signed-sqrt space)
    samples_original = []     # Inverse transformed back to original elevation
    
    DOWNSAMPLE_FACTOR = 8  # lowfreq is 8x smaller than residual
    
    with h5py.File(h5_file, 'r') as f:
        # Collect all available keys with their shapes
        all_keys = []
        for res in f.keys():
            res_group = f[res]
            for chunk_id in res_group.keys():
                chunk_group = res_group[chunk_id]
                for subchunk_id in chunk_group.keys():
                    subchunk_group = chunk_group[subchunk_id]
                    if 'residual' in subchunk_group and 'lowfreq' in subchunk_group:
                        # Check if mostly land
                        pct_land = subchunk_group['residual'].attrs.get('pct_land', 0)
                        if pct_land >= min_pct_land:
                            residual_shape = subchunk_group['residual'].shape
                            all_keys.append((res, chunk_id, subchunk_id, residual_shape))
        
        print(f"Found {len(all_keys)} samples in dataset")
        
        # Randomly sample keys
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_keys), min(num_samples, len(all_keys)), replace=False)
        
        for idx in selected_indices:
            res, chunk_id, subchunk_id, residual_shape = all_keys[idx]
            h, w = residual_shape
            
            # Get random crop position if data is larger than crop_size
            if h >= crop_size and w >= crop_size:
                # Ensure crop positions are multiples of DOWNSAMPLE_FACTOR for alignment
                max_top = (h - crop_size) // DOWNSAMPLE_FACTOR
                max_left = (w - crop_size) // DOWNSAMPLE_FACTOR
                
                lowfreq_top = np.random.randint(0, max_top + 1)
                lowfreq_left = np.random.randint(0, max_left + 1)
                
                # Convert to residual coordinates (multiply by 8)
                top = lowfreq_top * DOWNSAMPLE_FACTOR
                left = lowfreq_left * DOWNSAMPLE_FACTOR
                
                # Read aligned crops
                residual_crop = f[res][chunk_id][subchunk_id]['residual'][top:top+crop_size, left:left+crop_size]
                
                lowfreq_size = crop_size // DOWNSAMPLE_FACTOR
                lowfreq_crop = f[res][chunk_id][subchunk_id]['lowfreq'][lowfreq_top:lowfreq_top+lowfreq_size, lowfreq_left:lowfreq_left+lowfreq_size]
            else:
                # Use the full data if it's smaller
                residual_crop = f[res][chunk_id][subchunk_id]['residual'][:]
                lowfreq_crop = f[res][chunk_id][subchunk_id]['lowfreq'][:]
            
            # Reconstruct full terrain using laplacian decode
            crop_transformed = laplacian_decode(residual_crop, lowfreq_crop)
            crop_transformed[crop_transformed < 0] = 0
            
            # Handle NaN values
            if np.any(np.isnan(crop_transformed)):
                crop_transformed = np.nan_to_num(crop_transformed, nan=np.nanmean(crop_transformed))
            
            # Inverse transform to get original elevation
            crop_original = np.sign(crop_transformed) * np.square(crop_transformed)
            crop_transformed = signed_sqrt_transform(crop_original)
            
            samples_transformed.append(crop_transformed)
            samples_original.append(crop_original)
    
    return samples_original, samples_transformed

def compute_stats(samples):
    """Compute mean and standard deviation for each sample"""
    means = []
    stds = []
    variances = []
    
    for sample in samples:
        means.append(np.mean(sample))
        stds.append(np.std(sample))
        variances.append(np.var(sample))
    
    return np.array(means), np.array(stds), np.array(variances)

def main():
    # Path to dataset
    h5_file = Path("./data/dataset.h5")
    
    if not h5_file.exists():
        print(f"Error: Dataset not found at {h5_file}")
        return
    
    print("Loading samples from dataset and reconstructing full terrain...")
    samples_original, samples_transformed = load_samples_from_h5(h5_file, num_samples=1000, crop_size=512)
    print(f"Loaded {len(samples_original)} samples")
    
    # Compute stats for original data (without transform)
    print("Computing statistics for original elevation data (no transform)...")
    original_means, original_stds, _ = compute_stats(samples_original)
    
    # Compute stats for transformed data (with signed-sqrt transform)
    print("Computing statistics for transformed data (signed-sqrt)...")
    transformed_means, transformed_stds, _ = compute_stats(samples_transformed)
    
    # Bin based on raw elevation means (50 buckets)
    print("Binning data into 50 buckets based on raw elevation...")
    num_buckets = 8
    bin_edges = np.linspace(original_means.min(), original_means.max(), num_buckets + 1)
    bin_indices = np.digitize(original_means, bin_edges) - 1
    # Handle edge case: values at max go to last bucket
    bin_indices[bin_indices == num_buckets] = num_buckets - 1
    
    # Compute median standard deviation for each bucket
    median_std_original = []
    median_std_transformed = []
    bin_centers = []
    
    for i in range(num_buckets):
        mask = bin_indices == i
        if np.any(mask):
            median_std_original.append(np.median(original_stds[mask]))
            median_std_transformed.append(np.median(transformed_stds[mask]))
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        else:
            median_std_original.append(np.nan)
            median_std_transformed.append(np.nan)
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    median_std_original = np.array(median_std_original)
    median_std_transformed = np.array(median_std_transformed)
    bin_centers = np.array(bin_centers)
    
    # Create bar chart with dual y-axes
    print("Creating bar chart...")
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()  # Create second y-axis
    
    x = np.arange(len(bin_centers))
    width = 0.35
    
    # Filter out NaN values for plotting
    valid_mask = ~(np.isnan(median_std_original) | np.isnan(median_std_transformed))
    x_valid = x[valid_mask]
    bin_centers_valid = bin_centers[valid_mask]
    median_std_orig_valid = median_std_original[valid_mask]
    median_std_trans_valid = median_std_transformed[valid_mask]
    
    # Raw elevation bars on left y-axis
    bars1 = ax1.bar(x_valid - width/2, median_std_orig_valid, width, label='Raw Elevation', 
                   color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Transformed elevation bars on right y-axis
    bars2 = ax2.bar(x_valid + width/2, median_std_trans_valid, width, label='Transformed Elevation', 
                   color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Configure left axis (raw elevation)
    ax1.set_xlabel('Mean Elevation (m) - Raw', fontsize=15)
    ax1.set_ylabel('Median Standard Deviation (Raw Elevation)', fontsize=15, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Configure right axis (transformed elevation)
    ax2.set_ylabel('Median Standard Deviation (Transformed Elevation)', fontsize=15, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=12)
    
    # Configure x-axis
    ax1.set_xticks(x_valid[::max(1, len(x_valid)//20)])  # Show every Nth tick to avoid crowding
    ax1.set_xticklabels([f'{val:.0f}' for val in bin_centers_valid[::max(1, len(x_valid)//20)]], 
                       rotation=45, ha='right', fontsize=10)
    
    # Title
    ax1.set_title('Median Standard Deviation by Elevation Bucket\n(Raw vs Signed-Sqrt Transformed)', 
                 fontsize=18, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=13, loc='upper left')
    
    plt.tight_layout()
    
    # Print some statistics
    print("\n=== Statistics Summary ===")
    print(f"Number of buckets: {num_buckets}")
    print(f"Buckets with data: {np.sum(valid_mask)}")
    print(f"Raw elevation range: [{original_means.min():.2f}, {original_means.max():.2f}] meters")
    print(f"Median standard deviation (raw): {np.nanmedian(median_std_orig_valid):.2e}")
    print(f"Median standard deviation (transformed): {np.nanmedian(median_std_trans_valid):.2e}")
    
    plt.show()

if __name__ == "__main__":
    main()

