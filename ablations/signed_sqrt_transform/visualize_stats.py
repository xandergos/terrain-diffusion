#!/usr/bin/env python3
"""
Script to visualize the effect of signed-sqrt transform on variance vs mean relationship.
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

def signed_sqrt_transform(values):
    """Apply signed sqrt transform to elevation data"""
    return np.sign(values) * np.sqrt(np.abs(values))

def inverse_signed_sqrt_transform(values):
    """Invert the signed sqrt transform to get back original elevation"""
    return np.sign(values) * values * values

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
            
            # Handle NaN values
            if np.any(np.isnan(crop_transformed)):
                crop_transformed = np.nan_to_num(crop_transformed, nan=np.nanmean(crop_transformed))
            
            # Inverse transform to get original elevation
            crop_original = inverse_signed_sqrt_transform(crop_transformed)
            
            samples_transformed.append(crop_transformed)
            samples_original.append(crop_original)
    
    return samples_original, samples_transformed

def compute_stats(samples):
    """Compute mean and variance for each sample"""
    means = []
    variances = []
    
    for sample in samples:
        means.append(np.mean(sample))
        variances.append(np.var(sample))
    
    return np.array(means), np.array(variances)

def main():
    # Path to dataset
    h5_file = Path("/mnt/ntfs2/shared/terrain-diffusion/data/dataset.h5")
    
    if not h5_file.exists():
        print(f"Error: Dataset not found at {h5_file}")
        return
    
    print("Loading samples from dataset and reconstructing full terrain...")
    samples_original, samples_transformed = load_samples_from_h5(h5_file, num_samples=200, crop_size=512)
    print(f"Loaded {len(samples_original)} samples")
    
    # Compute stats for original data (without transform)
    print("Computing statistics for original elevation data (no transform)...")
    original_means, original_vars = compute_stats(samples_original)
    
    # Compute stats for transformed data (with signed-sqrt transform)
    print("Computing statistics for transformed data (signed-sqrt)...")
    transformed_means, transformed_vars = compute_stats(samples_transformed)
    
    # Create scatter plot
    print("Creating scatter plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Compute log variances
    original_log_vars = np.log(original_vars)
    transformed_log_vars = np.log(transformed_vars)
    
    # Compute log-log correlation: sign(x) * ln(|x|+1) vs log(variance)
    original_log_means = np.sign(original_means) * np.log(np.abs(original_means) + 1)
    transformed_log_means = np.sign(transformed_means) * np.log(np.abs(transformed_means) + 1)
    r_loglog_orig = np.corrcoef(original_log_means, original_log_vars)[0, 1]
    r_loglog_trans = np.corrcoef(transformed_log_means, transformed_log_vars)[0, 1]
    
    # Linear regression on (mean, log(variance))
    slope_orig, intercept_orig, r_orig, p_orig, se_orig = stats.linregress(original_means, original_log_vars)
    slope_trans, intercept_trans, r_trans, p_trans, se_trans = stats.linregress(transformed_means, transformed_log_vars)
    
    # Original data (no transform)
    ax1.scatter(original_means, original_vars, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
    
    # Add regression line (convert back from log space for plotting)
    x_range_orig = np.linspace(original_means.min(), original_means.max(), 100)
    y_fit_orig = np.exp(slope_orig * x_range_orig + intercept_orig)
    ax1.plot(x_range_orig, y_fit_orig, 'r-', linewidth=2, label=f'Linear fit (log var)')
    
    ax1.set_xlabel('Mean Elevation (m)', fontsize=12)
    ax1.set_ylabel('Variance (m²)', fontsize=12)
    ax1.set_title(f'Original Elevation Data\n(No Transform)\nCorr(mean, log(var)) = {r_orig:.4f}', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10)
    
    # Transformed data (with signed-sqrt)
    ax2.scatter(transformed_means, transformed_vars, alpha=0.6, s=50, color='orange', edgecolors='black', linewidth=0.5)
    
    # Add regression line (convert back from log space for plotting)
    x_range_trans = np.linspace(transformed_means.min(), transformed_means.max(), 100)
    y_fit_trans = np.exp(slope_trans * x_range_trans + intercept_trans)
    ax2.plot(x_range_trans, y_fit_trans, 'r-', linewidth=2, label=f'Linear fit (log var)')
    
    ax2.set_xlabel('Mean (signed-sqrt m)', fontsize=12)
    ax2.set_ylabel('Variance ((signed-sqrt m)²)', fontsize=12)
    ax2.set_title(f'With Signed-Sqrt Transform\n(Dataset Format)\nCorr(mean, log(var)) = {r_trans:.4f}', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path("variance_vs_mean.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Print some statistics
    print("\n=== Statistics Summary ===")
    print(f"Original elevation data (no transform):")
    print(f"  Mean range: [{original_means.min():.2f}, {original_means.max():.2f}] meters")
    print(f"  Variance range: [{original_vars.min():.2e}, {original_vars.max():.2e}] m²")
    print(f"  Mean-variance correlation: {np.corrcoef(original_means, original_vars)[0, 1]:.4f}")
    print(f"  Mean-log(variance) correlation: {r_orig:.4f}")
    print(f"  Log(mean)-log(variance) correlation: {r_loglog_orig:.4f}")
    
    print(f"\nTransformed data (signed-sqrt):")
    print(f"  Mean range: [{transformed_means.min():.2f}, {transformed_means.max():.2f}]")
    print(f"  Variance range: [{transformed_vars.min():.2e}, {transformed_vars.max():.2e}]")
    print(f"  Mean-variance correlation: {np.corrcoef(transformed_means, transformed_vars)[0, 1]:.4f}")
    print(f"  Mean-log(variance) correlation: {r_trans:.4f}")
    print(f"  Log(mean)-log(variance) correlation: {r_loglog_trans:.4f}")
    
    print(f"\nVariance stabilization effect:")
    print(f"  Correlation reduction (linear): {np.corrcoef(original_means, original_vars)[0, 1] - np.corrcoef(transformed_means, transformed_vars)[0, 1]:.4f}")
    print(f"  Correlation reduction (mean-log(var)): {r_orig - r_trans:.4f}")
    print(f"  Correlation reduction (log-log): {r_loglog_orig - r_loglog_trans:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()

