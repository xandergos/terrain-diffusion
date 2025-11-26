#!/usr/bin/env python3
"""
Script to visualize the effect of signed-sqrt transform on standard deviation vs mean relationship.
Loads 512x512 terrain samples from dataset.h5, performs laplacian decoding to get full images,
and creates scatter plots comparing the statistics before and after transformation.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats
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
    
    for sample in samples:
        means.append(np.mean(sample))
        stds.append(np.std(sample))
    
    return np.array(means), np.array(stds)

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
    original_means, original_stds = compute_stats(samples_original)
    original_means = np.sign(original_means) * np.sqrt(np.abs(original_means))
    
    # Compute stats for transformed data (with signed-sqrt transform)
    print("Computing statistics for transformed data (signed-sqrt)...")
    transformed_means, transformed_stds = compute_stats(samples_transformed)
    
    # Create plots
    print("Creating plots...")

    # Compute log standard deviations
    original_log_stds = np.log(original_stds)
    transformed_log_stds = np.log(transformed_stds)

    # Compute log-log correlation: sign(x) * ln(|x|+1) vs log(std)
    original_log_means = np.sign(original_means) * np.log(np.abs(original_means) + 1)
    transformed_log_means = np.sign(transformed_means) * np.log(np.abs(transformed_means) + 1)
    r_loglog_orig = np.corrcoef(original_log_means, original_log_stds)[0, 1]
    r_loglog_trans = np.corrcoef(transformed_log_means, transformed_log_stds)[0, 1]

    # Linear regression on (mean, std)
    slope_std_orig, intercept_std_orig, r_std_orig, _, _ = stats.linregress(original_means, original_stds)
    slope_std_trans, intercept_std_trans, r_std_trans, _, _ = stats.linregress(transformed_means, transformed_stds)

    # Linear regression on (mean, log(std))
    slope_log_orig, intercept_log_orig, r_log_orig, _, _ = stats.linregress(original_means, original_log_stds)
    slope_log_trans, intercept_log_trans, r_log_trans, _, _ = stats.linregress(transformed_means, transformed_log_stds)

    # Figure 1: Mean vs Std (Original and Transformed)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original data (no transform)
    ax1.scatter(original_means, original_stds, alpha=0.3, s=40, color='blue', edgecolors='black', linewidth=1.0)
    x_range_orig = np.linspace(original_means.min(), original_means.max(), 100)
    y_fit_orig = slope_std_orig * x_range_orig + intercept_std_orig
    ax1.plot(x_range_orig, y_fit_orig, 'r-', linewidth=4, label=f'Linear fit')
    ax1.set_xlabel('Mean Elevation (m)', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Standard deviation (m)', fontsize=22, fontweight='bold')
    ax1.set_title(f'Original Elevation Data\nCorr = {r_std_orig:.4f}', fontsize=22, fontweight='bold')
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=20)

    # Transformed data (with signed-sqrt)
    ax2.scatter(transformed_means, transformed_stds, alpha=0.6, s=40, color='orange', edgecolors='black', linewidth=1.0)
    x_range_trans = np.linspace(transformed_means.min(), transformed_means.max(), 100)
    y_fit_trans = slope_std_trans * x_range_trans + intercept_std_trans
    ax2.plot(x_range_trans, y_fit_trans, 'r-', linewidth=4, label=f'Linear fit')
    ax2.set_xlabel('Mean (signed-sqrt m)', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Standard deviation (signed-sqrt m)', fontsize=22, fontweight='bold')
    ax2.set_title(f'With Signed-Sqrt Transform\nCorr = {r_std_trans:.4f}', fontsize=22, fontweight='bold')
    ax2.tick_params(labelsize=16)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=20)

    plt.tight_layout()

    # Figure 2: Mean vs log(Std) (Original and Transformed)
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original data (no transform)
    bx1.scatter(original_means, original_log_stds, alpha=0.6, s=40, color='blue', edgecolors='black', linewidth=1.0)
    y_fit_log_orig = slope_log_orig * x_range_orig + intercept_log_orig
    bx1.plot(x_range_orig, y_fit_log_orig, 'r-', linewidth=4, label=f'Linear fit')
    bx1.set_xlabel('Mean (sqrt m)', fontsize=22, fontweight='bold')
    bx1.set_ylabel('log(σ)', fontsize=22, fontweight='bold')
    bx1.set_title(f'Original Elevation\nCorr = {r_log_orig:.4f}', fontsize=22, fontweight='bold')
    bx1.tick_params(labelsize=16)
    bx1.grid(True, alpha=0.3)
    bx1.set_axisbelow(True)
    bx1.legend(fontsize=20)

    # Transformed data (with signed-sqrt)
    bx2.scatter(transformed_means, transformed_log_stds, alpha=0.6, s=40, color='orange', edgecolors='black', linewidth=1.0)
    y_fit_log_trans = slope_log_trans * x_range_trans + intercept_log_trans
    bx2.plot(x_range_trans, y_fit_log_trans, 'r-', linewidth=4, label=f'Linear fit')
    bx2.set_xlabel('Mean (sqrt m)', fontsize=22, fontweight='bold')
    bx2.set_ylabel('log(σ) (signed-sqrt)', fontsize=22, fontweight='bold')
    bx2.set_title(f'With Signed-Sqrt Transform\nCorr = {r_log_trans:.4f}', fontsize=22, fontweight='bold')
    bx2.tick_params(labelsize=16)
    bx2.grid(True, alpha=0.3)
    bx2.set_axisbelow(True)
    bx2.legend(fontsize=20)

    plt.tight_layout()
    
    # Print some statistics
    print("\n=== Statistics Summary ===")
    print(f"Original elevation data (no transform):")
    print(f"  Mean range: [{original_means.min():.2f}, {original_means.max():.2f}] meters")
    print(f"  Std range: [{original_stds.min():.2e}, {original_stds.max():.2e}] m")
    print(f"  Mean-std correlation: {np.corrcoef(original_means, original_stds)[0, 1]:.4f}")
    print(f"  Mean-log(std) correlation: {r_log_orig:.4f}")
    print(f"  Log(mean)-log(std) correlation: {r_loglog_orig:.4f}")
    
    print(f"\nTransformed data (signed-sqrt):")
    print(f"  Mean range: [{transformed_means.min():.2f}, {transformed_means.max():.2f}]")
    print(f"  Std range: [{transformed_stds.min():.2e}, {transformed_stds.max():.2e}]")
    print(f"  Mean-std correlation: {np.corrcoef(transformed_means, transformed_stds)[0, 1]:.4f}")
    print(f"  Mean-log(std) correlation: {r_log_trans:.4f}")
    print(f"  Log(mean)-log(std) correlation: {r_loglog_trans:.4f}")
    
    print(f"\nStandard deviation stabilization effect:")
    print(f"  Correlation reduction (linear): {np.corrcoef(original_means, original_stds)[0, 1] - np.corrcoef(transformed_means, transformed_stds)[0, 1]:.4f}")
    print(f"  Correlation reduction (mean-log(std)): {r_log_orig - r_log_trans:.4f}")
    print(f"  Correlation reduction (log-log): {r_loglog_orig - r_loglog_trans:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()

