import torch
from tqdm import tqdm
from diffusion.datasets.datasets import H5AutoencoderDataset
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import DiffusionAutoencoder, EDMAutoencoder
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

dataset = H5AutoencoderDataset('dataset_full.h5', 512, [0.9999, 1.0], '240m', eval_dataset=False)

model_cfg = EDMAutoencoder.load_config('checkpoints/autoencoder_x8-plain_ft/configs/model_config_latest')
model = EDMAutoencoder.from_config(model_cfg)
load_model(model, 'checkpoints/autoencoder_x8-plain_ft/latest_checkpoint/model.safetensors')

model.encoder.save_pretrained('checkpoints/models/encoder_512')

ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], update_every=20, checkpoint_every_num_steps=12800, allow_different_devices=True,
                 checkpoint_folder='checkpoints/autoencoder_x8-plain/phema')
ema.load_state_dict(torch.load('checkpoints/autoencoder_x8-plain_ft/latest_checkpoint/phema.pt'))
ema.synthesize_ema_model(sigma_rel=0.15).copy_params_from_ema_to_model()

model = model.to(device)

mode = 'stats'

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
        
        scaled_clean_images = images / 0.5
        with torch.no_grad():
            if cond_img is not None:
                x = torch.cat([scaled_clean_images, cond_img[None]], dim=1)
            else:
                x = scaled_clean_images
            enc_mean, enc_logvar = model.preencode(x, conditional_inputs)
            z = model.postencode(enc_mean, enc_logvar, use_mode=True)
            decoded_x = model.decode(z)
        
        if mode == 'plot':
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5))
            
            vmin = scaled_clean_images[0].min().item()
            vmax = scaled_clean_images[0].max().item()
            
            ax1.imshow(scaled_clean_images[0].permute(1, 2, 0).cpu().numpy(), vmin=vmin, vmax=vmax)
            ax1.set_title('Original')
            ax1.axis('off')
            
            ax2.imshow(decoded_x[0].permute(1, 2, 0).cpu().numpy(), vmin=vmin, vmax=vmax)
            ax2.set_title('Decoded')
            ax2.axis('off')
            
            # Plot the difference
            # Apply mean pooling to the clean image
            pooled = torch.nn.functional.avg_pool2d(scaled_clean_images, kernel_size=8, stride=8)
            # Upsample back to original size
            pooled = torch.nn.functional.interpolate(pooled, size=scaled_clean_images.shape[-2:], mode='nearest')
            ax3.imshow(pooled[0].permute(1, 2, 0).cpu().numpy())
            ax3.set_title('Mean Pooled Original')
            ax3.axis('off')
            
            # Calculate MSE between clean and pooled
            mse_pooled = F.mse_loss(scaled_clean_images, pooled).item()
            
            # Calculate MSE between clean and decoded
            mse_decoded = F.mse_loss(scaled_clean_images, decoded_x).item()
            
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
            mse = 4 * F.mse_loss(scaled_clean_images, decoded_x).item()
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