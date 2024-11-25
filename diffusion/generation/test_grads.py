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

device = 'cuda'

def get_model(channels, layers, tag, sigma_rel=None, ema_step=None):
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

model = get_model(64, 3, '64x3', None).to(device)
model.eval()

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 256, [0.9999, 1], '480m', eval_dataset=False,
                                   latents_mean=[0, 0.07, 0.12, 0.07],
                                   latents_std=[1.4127, 0.8170, 0.8386, 0.8414])

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

dataloader_iter = iter(dataloader)
batch = next(dataloader_iter)
images = batch['image'].to(device)
cond_img = batch.get('cond_img').to(device)
conditional_inputs = batch.get('cond_inputs')

norms0 = []
norms1 = []
n = 100
for t in np.linspace(0, np.pi/2, n):
    t = torch.tensor(t, device=device, requires_grad=True)
    
    sigma_data = 0.5
    sigma = sigma_data * torch.tan(t)
    
    z = torch.randn_like(images) * sigma_data
    x_t = torch.cos(t) * images + torch.sin(t) * z
    
    dxt_dt = torch.cos(t) * z - torch.sin(t) * images

    x_t.requires_grad_(True)  # Enable gradients for JVP
    def model_wrapper(scaled_x_t, t):
        if cond_img is not None:
            scaled_x_t = torch.cat([scaled_x_t, cond_img], dim=1)
        pred = model(scaled_x_t, noise_labels=t.flatten(), conditional_inputs=conditional_inputs)
        
        # Calculate f
        f = torch.cos(t) * x_t + torch.sin(t) * sigma_data * pred
        return f

    v_x = torch.zeros_like(dxt_dt)
    v_t = torch.tensor(1.0, device=device, dtype=torch.float32)
    pred, F_theta_grad = torch.func.jvp(
        model_wrapper, 
        (x_t / sigma_data, t),
        (v_x, v_t)
    )
    norms0.append(torch.linalg.vector_norm(F_theta_grad).item())
    
    v_x = dxt_dt
    v_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    pred, F_theta_grad = torch.func.jvp(
        model_wrapper, 
        (x_t / sigma_data, t),
        (v_x, v_t)
    )
    norms1.append(torch.linalg.vector_norm(F_theta_grad).item())
    
    print(torch.linalg.vector_norm(dxt_dt))
    

plt.plot(np.linspace(0, np.pi/2, n), norms0)
plt.plot(np.linspace(0, np.pi/2, n), norms1)
plt.show()
