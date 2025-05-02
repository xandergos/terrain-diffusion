import torch
from tqdm import tqdm
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.training.datasets.datasets import H5AutoencoderDataset
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMAutoencoder
from ema_pytorch import PostHocEMA
from safetensors.torch import load_model
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Results:
# 0: 0.146
# 0.05: 0.5
# 0.1: 4
# 0.01

device = 'cuda'

dataset = H5AutoencoderDataset('dataset.h5', 512, [[0, 0.01], [0.01, 1]], [90, 90], subset_weights=[0.01, 1], split='train')

model = EDMAutoencoder.from_pretrained('/mnt/ntfs2/shared/terrain-diffusion/checkpoints/models/autoencoder').to(device)
model = model.to(device)

mode = 'plot'

mses = []
latents = []
if mode == 'stats':
    repeats = 4
elif mode == 'plot':
    repeats = 1
else:
    repeats = 50
dataloader = DataLoader(dataset, batch_size=1 if mode == 'plot' else 8, shuffle=False)
for i in range(repeats):
    for sample in dataloader:
        images = sample['image'].to(device)
        cond_img = sample.get('cond_img')
        conditional_inputs = sample.get('cond_inputs')
        if cond_img is not None:
            cond_img = cond_img.to(device)
        if conditional_inputs is not None:
            conditional_inputs = conditional_inputs.to(device)
        
        with torch.no_grad():
            if cond_img is not None:
                x = torch.cat([images, cond_img[None]], dim=1)
            else:
                x = images
            enc_mean, enc_logvar = model.preencode(x, conditional_inputs)
            z = model.postencode(enc_mean, enc_logvar, use_mode=False)
            print(enc_mean.shape)
            decoded_x = model.decode(z)
            
        enc_residual, enc_lowfreq = decoded_x[:, :1], decoded_x[:, 1:2]
        enc_residual = dataset.denormalize_residual(enc_residual, 90)
        enc_lowfreq = dataset.denormalize_lowfreq(enc_lowfreq, 90)
        enc_residual, enc_lowfreq = laplacian_denoise(enc_residual, enc_lowfreq, 5.0)
        decoded_terrain = laplacian_decode(enc_residual, enc_lowfreq)
        #decoded_terrain = decoded_x[:, 3:4]
        
        true_residual, true_lowfreq = images[:, :1], images[:, 1:2]
        true_residual = dataset.denormalize_residual(true_residual, 90)
        true_lowfreq = dataset.denormalize_lowfreq(true_lowfreq, 90)
        true_residual, true_lowfreq = laplacian_denoise(true_residual, true_lowfreq, 5.0)
        true_terrain = laplacian_decode(true_residual, true_lowfreq)
        #true_terrain = images[:, 3:4]
        
        if mode == 'plot':
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5))
            
            vmin = true_terrain[0].min().item()
            vmax = true_terrain[0].max().item()
            
            ax1.imshow(true_terrain[0].permute(1, 2, 0).cpu().numpy())
            ax1.set_title('Original')
            ax1.axis('off')
            
            ax2.imshow(decoded_terrain[0].permute(1, 2, 0).cpu().numpy())
            ax2.set_title('Decoded')
            ax2.axis('off')
            
            # Plot the difference
            # Apply mean pooling to the clean image
            pooled = torch.nn.functional.avg_pool2d(true_terrain, kernel_size=8, stride=8)
            # Upsample back to original size
            pooled = torch.nn.functional.interpolate(pooled, size=true_terrain.shape[-2:], mode='nearest')
            ax3.imshow(pooled[0].permute(1, 2, 0).cpu().numpy())
            ax3.set_title('Mean Pooled Original')
            ax3.axis('off')
            
            # Calculate MSE between clean and pooled
            mse_pooled = F.mse_loss(true_terrain, pooled).item()
            
            # Calculate MSE between clean and decoded
            mse_decoded = F.mse_loss(true_terrain, decoded_terrain).item()
            
            print(f"MSE between original and mean pooled: {mse_pooled:.4f}")
            print(f"MSE between original and decoded: {mse_decoded:.4f}")
            
            # Plot the latent vector z (first three channels)
            z_normalized = (z[0, :3] - z[0, :3].amin(dim=(1, 2), keepdim=True)) / (z[0, :3].amax(dim=(1, 2), keepdim=True) - z[0, :3].amin(dim=(1, 2), keepdim=True))
            ax4.imshow(z_normalized.permute(1, 2, 0).cpu().numpy())
            ax4.set_title('Latent Vector (First 3 Channels)')
            ax4.axis('off')
            
            # Adjust the figure size to accommodate the new subplots
            fig.set_size_inches(15, 10)
            
            plt.tight_layout()
            plt.show()
        elif mode == 'evaluate':
            # Calculate MSE between original and decoded images
            mse = F.mse_loss(true_terrain, decoded_terrain).item()
            mses.append(mse)
            print(f"Average MSE: {torch.tensor(mses).mean().item():.4f}")
        elif mode == 'stats':
            latents.append(z)

if mode == 'evaluate':
    print(f"Average MSE: {torch.tensor(mses).mean().item():.4f}")
elif mode == 'stats':
    latents = torch.cat(latents, dim=0)
    print(f"Latent vector mean: {latents.mean(dim=[0, 2, 3]).cpu().numpy()}")
    print(f"Latent vector std: {latents.std(dim=[0, 2, 3]).cpu().numpy()}")