import numpy as np
import torch
from tqdm import tqdm
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from diffusion.datasets.datasets import H5SuperresTerrainDataset, LongDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

scheduler = EDMDPMSolverMultistepScheduler(0.002, 80, 0.5)


device = 'cuda'

def get_model(channels, layers, tag, sigma_rel=None, ema_step=None, sigma_rels=[0.04, 0.09]):
    model = EDMUnet2D(
        image_size=512,
        in_channels=5,
        out_channels=1,
        model_channels=channels,
        model_channel_mults=[1, 2, 3, 4],
        layers_per_block=layers,
        attn_resolutions=[],
        midblock_attention=False,
        concat_balance=0.5,
        conditional_inputs=[],
        fourier_scale=1.0
    )
    load_model(model, f'checkpoints/diffusion_x8-{tag}/latest_checkpoint/model.safetensors')

    if sigma_rel is not None:
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], update_every=1, checkpoint_every_num_steps=12800, allow_different_devices=True,
                        checkpoint_folder=f'checkpoints/diffusion_x8-{tag}/phema')
        ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/latest_checkpoint/phema.pt'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()

    return model

model_m = get_model(64, 3, '64x3', 0.05, fs='pos').to(device)
model_g = get_model(32, 2, '32x2', 0.05, ema_step=2048*4, fs='pos').to(device)
guidance_scale = 2.0

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 256, [0.9999, 1], '480m', eval_dataset=False,
                                   latents_mean=[0, 0.07, 0.12, 0.07],
                                   latents_std=[1.4127, 0.8170, 0.8386, 0.8414])

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

torch.set_grad_enabled(False)

# Create output directories
output_dir = Path("outputs/fid_images_64x3_ema005_step192k")
orig_dir = output_dir / "original"
recon_dir = output_dir / "reconstructed"
orig_dir.mkdir(parents=True, exist_ok=False)
recon_dir.mkdir(parents=True, exist_ok=False)

total_images = 0
print("Generating and saving images...")
pbar = tqdm(total=10000)
while pbar.n < pbar.total:
    for batch in dataloader:
        if pbar.n >= pbar.total:
            break
            
        images = batch['image'].to(device)
        cond_img = batch.get('cond_img').to(device)
        
        # Generate reconstruction
        scheduler.set_timesteps(20)
        samples = torch.randn(images.shape, device=device) * scheduler.sigmas[0]
        
        for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
            sigma, t = sigma.to(device), t.to(device)
            scaled_input = scheduler.precondition_inputs(samples, sigma)
            cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1))
            
            x = torch.cat([scaled_input, cond_img], dim=1)
            if guidance_scale == 1.0:
                model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
                model_output = model_output_m
            else:
                model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
                model_output_g = model_g(x, noise_labels=cnoise, conditional_inputs=[])
            
                # Combine predictions using autoguidance
                model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
            
            samples = scheduler.step(model_output, t, samples).prev_sample
        
        # Process each image in the batch individually
        images_np = images.cpu().numpy()
        recon_np = samples.cpu().numpy()
        
        for i in range(images_np.shape[0]):
            orig_img = images_np[i, 0]  # Remove channel dimension
            recon_img = recon_np[i, 0]  # Remove channel dimension
            
            # Normalize using original image's min/max
            img_min = orig_img.min()
            img_max = orig_img.max()
            
            # Normalize both images using the same constants
            norm_orig = (orig_img - img_min) / (img_max - img_min)
            norm_recon = (recon_img - img_min) / (img_max - img_min)
            
            # Clip normalized images to [0,1] range
            norm_orig = np.clip(norm_orig, 0, 1)
            norm_recon = np.clip(norm_recon, 0, 1)
            
            # Save images as PNG with explicit vmin/vmax
            plt.imsave(orig_dir / f"img_{total_images:05d}.png", norm_orig, cmap='gray', vmin=0, vmax=1)
            plt.imsave(recon_dir / f"img_{total_images:05d}.png", norm_recon, cmap='gray', vmin=0, vmax=1)
            
            total_images += 1
            
        pbar.update(len(images))

print(f"Generated {total_images} image pairs")
        