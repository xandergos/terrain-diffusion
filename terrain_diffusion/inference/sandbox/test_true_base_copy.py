import os
import torch
import h5py
from terrain_diffusion.training.unet import EDMAutoencoder
from terrain_diffusion.data.laplacian_encoder import *
import matplotlib.pyplot as plt
import numpy as np
import random

device = 'cuda'

# Enable parallel processing on CPU
torch.set_num_threads(16)

# Load autoencoder for decoding
autoencoder = EDMAutoencoder.from_pretrained('checkpoints/models/autoencoder').to(device)

torch.set_grad_enabled(False)

# Open HDF5 file directly
h5_file = 'dataset.h5'
resolution = 90  # The resolution we want to use

samples_to_visualize = 9  # Number of samples to visualize
samples = []

with h5py.File(h5_file, 'r') as f:
    res_group = f[str(resolution)]
    residual_mean = res_group.attrs['residual_mean']
    residual_std = res_group.attrs['residual_std']
    
    # Collect a list of valid chunk/subchunk combinations with pct_land in (0.4, 0.6)
    valid_samples = []
    
    for chunk_id in res_group:
        chunk_group = res_group[chunk_id]
        for subchunk_id in chunk_group:
            subchunk_group = chunk_group[subchunk_id]
            # Check if it has both latent and residual data
            if 'latent' in subchunk_group and 'residual' in subchunk_group:
                # Filter by pct_land in range (0.4, 0.6)
                if 'pct_land' in subchunk_group['residual'].attrs:
                    pct_land = subchunk_group['residual'].attrs['pct_land']
                    if 0.3 <= pct_land <= 0.5:
                        valid_samples.append((chunk_id, subchunk_id))
    
    # Randomly sample from valid samples
    if len(valid_samples) > samples_to_visualize:
        selected_samples = random.sample(valid_samples, samples_to_visualize)
    else:
        selected_samples = valid_samples[:samples_to_visualize]
    
    print(f"Found {len(valid_samples)} samples with pct_land in (0.4, 0.6)")
    print(f"Selected {len(selected_samples)} samples for visualization")
    
    for chunk_id, subchunk_id in selected_samples:
        group_path = f"{resolution}/{chunk_id}/{subchunk_id}"
        
        # 1. Load latent data (use transform_idx=0 for no transformation)
        latent_data = torch.from_numpy(f[f"{group_path}/latent"][0]).to(device)  # Use first transform
        latent_channels = latent_data.shape[0]
        means, logvars = latent_data[:latent_channels//2], latent_data[latent_channels//2:]
        
        # Sample from distribution (this is what the diffusion model would predict)
        sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
        
        # 2. Load the true residual data
        true_residual = torch.from_numpy(f[f"{group_path}/residual"][:]).to(device)[None, None]
        
        # 3. Load the lowfreq data
        lowfreq = torch.from_numpy(f[f"{group_path}/lowfreq"][:]).to(device)[None, None]
        
        # Get pct_land for display
        pct_land = f[f"{group_path}/residual"].attrs['pct_land']
        
        samples.append({
            'latent': sampled_latent, 
            'true_residual': true_residual,
            'lowfreq': lowfreq,
            'path': group_path,
            'pct_land': pct_land
        })

# Process all samples
for sample in samples:
    # Decode the latent to get residual
    sample['decoded'] = autoencoder.decode(sample['latent'][None])
    decoded_residual, watercover = sample['decoded'][:, :1], sample['decoded'][:, 1:2]
    sample['decoded_residual'] = decoded_residual
    sample['watercover'] = torch.sigmoid(watercover)

# Prepare visualization
fig, axs = plt.subplots(4, len(samples), figsize=(3*len(samples), 12))

# Row titles
row_labels = ['True Residual', 'Decoded Residual', 
             'Terrain from True Residual', 'Terrain from Decoded Residual']

for i, sample in enumerate(samples):
    # Get sample data
    true_residual = sample['true_residual']
    decoded_residual = sample['decoded_residual']
    lowfreq = sample['lowfreq']
    pct_land = sample['pct_land']
    
    # Denormalize residuals and lowfreq for laplacian decoding
    denorm_true_residual = true_residual * residual_std + residual_mean
    denorm_decoded_residual = decoded_residual * residual_std + residual_mean
    denorm_lowfreq = lowfreq
    
    # Generate laplacian decoded terrain
    true_terrain = laplacian_decode(denorm_true_residual, denorm_lowfreq)
    decoded_terrain = laplacian_decode(denorm_decoded_residual, denorm_lowfreq)
    
    # Resize if necessary (some might be too large for display)
    # For very large images, we'll center crop to 512x512
    max_display_size = 512
    
    def center_crop(tensor, size):
        h, w = tensor.shape[-2:]
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        return tensor[..., start_h:start_h+size, start_w:start_w+size]
    
    if denorm_true_residual.shape[-1] > max_display_size:
        denorm_true_residual = center_crop(denorm_true_residual, max_display_size)
        denorm_decoded_residual = center_crop(denorm_decoded_residual, max_display_size)
        true_terrain = center_crop(true_terrain, max_display_size)
        decoded_terrain = center_crop(decoded_terrain, max_display_size)
    
    # Row 1: True residual
    img = denorm_true_residual[0, 0].cpu().numpy()
    im = axs[0, i].imshow(img, cmap='viridis')
    axs[0, i].set_title(f"Sample {i+1}\npct_land: {pct_land:.2f}")  # Add pct_land to title
    axs[0, i].axis('off')
    plt.colorbar(im, ax=axs[0, i])
    
    # Row 2: Decoded residual
    img = denorm_decoded_residual[0, 0].cpu().numpy()
    im = axs[1, i].imshow(img, cmap='viridis')
    axs[1, i].axis('off')
    plt.colorbar(im, ax=axs[1, i])
    
    # Row 3: Terrain from true residual
    img = true_terrain[0, 0].cpu().numpy()
    im = axs[2, i].imshow(img, cmap='terrain')
    axs[2, i].axis('off')
    plt.colorbar(im, ax=axs[2, i])
    
    # Row 4: Terrain from decoded residual
    img = decoded_terrain[0, 0].cpu().numpy()
    im = axs[3, i].imshow(img, cmap='terrain')
    axs[3, i].axis('off')
    plt.colorbar(im, ax=axs[3, i])

# Add row labels
for idx, label in enumerate(row_labels):
    fig.text(0.01, 0.75 - idx * 0.25, label, va='center', ha='left', fontsize=14, rotation=90)

plt.tight_layout()
plt.savefig('residual_comparison.png', dpi=300, bbox_inches='tight')
plt.show()