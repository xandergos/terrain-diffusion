import os
import torch
from tqdm import tqdm
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import H5LatentsDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from terrain_diffusion.training.utils import recursive_to

scheduler = EDMDPMSolverMultistepScheduler(0.002, 10.0, 0.5)

device = 'cuda'

def get_model(checkpoint_path, sigma_rel=None, ema_step=None):
    config_path = os.path.join(checkpoint_path, 'model_config')
    model = EDMUnet2D.from_config(EDMUnet2D.load_config(config_path))

    if sigma_rel is not None:
        # sigma_rels are placeholders since we dont use them
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], checkpoint_folder=os.path.join(checkpoint_path, '..', 'phema')).to(device)
        #ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/{checkpoint}/phema.pt', map_location='cpu'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))

    return model

models = [
    get_model('checkpoints/diffusion_base-128x3/latest_checkpoint', sigma_rel=0.05).to(device)
]

# Enable parallel processing on CPU
torch.set_num_threads(16)

dataset = H5LatentsDataset('dataset.h5', 64, [[0, 1], [0, 1], [0, 1]], [90, 180, 360], [0, 0, 1], [0, 1, 2], eval_dataset=False,
                                   latents_mean=[0, 0, 0, 0],
                                   latents_std=[1, 1, 1, 1])

dataloader = DataLoader(dataset, batch_size=64)

torch.set_grad_enabled(False)

sigma_data = 0.5

# Generate log-spaced sigma values from 0.002 to 80
sigmas = torch.logspace(np.log10(0.002), np.log10(80), 30)
mse_values = {i: {j: [] for j in range(dataloader.batch_size)} for i in range(len(models))}

batch = next(iter(dataloader))
images = batch['image'].to(device)
cond_img = batch.get('cond_img').to(device)
cond_inputs = recursive_to(batch.get('cond_inputs'), device)

image_std_ratio = torch.std(images, dim=(1, 2, 3), keepdim=True) / sigma_data
step = 0
for sigma in tqdm(sigmas):
    sigma = sigma.to(device)
    sigma = sigma.expand(images.shape[0]).view(-1, 1, 1, 1)
    
    t = torch.atan(sigma / sigma_data)
    cnoise = t.flatten()
    
    # Add noise to images
    noise = torch.randn_like(images) * sigma_data
    x_t = torch.cos(t) * images + torch.sin(t) * noise
    
    # Get model predictions
    scaled_input = x_t / sigma_data
    x = torch.cat([scaled_input, cond_img], dim=1)
    
    for i, model in enumerate(models):
        model_output = model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
        pred_v_t = -sigma_data * model_output
        
        # Calculate MSE for each sample
        v_t = torch.cos(t) * noise - torch.sin(t) * images
        mse = (1 / sigma_data ** 2) * ((pred_v_t - v_t) ** 2).mean(dim=(1,2,3))
        
        # Store MSE for each sample
        for j in range(images.shape[0]):
            mse_values[i][j].append(mse[j].item())
        
    step += 1

# Calculate mean MSE across samples for each model and sigma
mean_mse_values = {}
for model_idx in mse_values:
    # For each model, average MSE across all samples at each sigma point
    mean_mse = np.array([np.mean([mse_values[model_idx][j][sigma_idx] 
                                 for j in range(dataloader.batch_size)]) 
                        for sigma_idx in range(len(sigmas))])
    mean_mse_values[model_idx] = mean_mse

# Create the line plot
plt.figure(figsize=(10, 6))
for model_idx, mean_mse in mean_mse_values.items():
    plt.plot(sigmas.cpu().numpy(), mean_mse, label=f'Model {model_idx}')

plt.xscale('log')  # Use log scale for sigma values
plt.yscale('log')  # Use log scale for MSE values
plt.xlabel('Sigma')
plt.ylabel('Mean MSE')
plt.title('Mean MSE vs Sigma')
plt.legend()
plt.grid(True)
plt.show()

