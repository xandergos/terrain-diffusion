import torch
from tqdm import tqdm
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from diffusion.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

scheduler = EDMDPMSolverMultistepScheduler(0.002, 10.0, 0.5)

device = 'cpu'

def get_model(channels, layers, tag, sigma_rel=None, ema_step=None, fs=1.0, checkpoint='latest_checkpoint', sigma_rels=[0.05, 0.1]):
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
        fourier_scale=fs
    )
    load_model(model, f'checkpoints/diffusion_x8-{tag}/{checkpoint}/model.safetensors')

    if sigma_rel is not None:
        ema = PostHocEMA(model, sigma_rels=sigma_rels, update_every=1, checkpoint_every_num_steps=12800, allow_different_devices=True,
                        checkpoint_folder=f'checkpoints/diffusion_x8-{tag}/phema').to(device)
        ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/{checkpoint}/phema.pt', map_location='cpu'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()

    return model

models = [
    get_model(32, 2, '32x2', 0.05, fs='pos').to(device),
    get_model(64, 3, '64x3', 0.1, fs='pos').to(device)
]

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 128, [0.9999, 1], '480m', eval_dataset=True,
                                   latents_mean=[0, 0.07, 0.12, 0.07],
                                   latents_std=[1.4127, 0.8170, 0.8386, 0.8414])

dataloader = DataLoader(dataset, batch_size=4)

torch.set_grad_enabled(False)

sigma_data = 0.5

# Generate log-spaced sigma values from 0.002 to 80
sigmas = torch.logspace(np.log10(0.002), np.log10(80), 10)
mse_values = {i: [] for i in range(len(models))}

batch = next(iter(dataloader))
images = batch['image'].to(device)
cond_img = batch.get('cond_img').to(device)

for sigma in tqdm(sigmas):
    sigma = sigma.to(device)
    t = torch.atan(sigma / sigma_data)
    cnoise = t.flatten()
    
    # Add noise to images
    noise = torch.randn_like(images) * sigma_data
    x_t = torch.cos(t) * images + torch.sin(t) * noise
    
    # Get model predictions
    scaled_input = x_t / sigma_data
    x = torch.cat([scaled_input, cond_img], dim=1)
    
    for i, model in enumerate(models):
        model_output = model(x, noise_labels=cnoise, conditional_inputs=[])
        pred_v_t = -sigma_data * model_output
        
        # Calculate MSE
        v_t = torch.cos(t) * noise - torch.sin(t) * images
        mse = (1 / sigma_data ** 2) * ((pred_v_t - v_t) ** 2).mean().item()
        mse_values[i].append(mse)

# Plot MSE vs sigma
plt.figure(figsize=(10, 6))
for i in range(len(models)):
    plt.loglog(sigmas, mse_values[i], label=f'Model {i+1}')
plt.grid(True)
plt.xlabel('Sigma')
plt.ylabel('MSE')
plt.title('Average MSE vs Noise Level')
plt.legend()
plt.show()