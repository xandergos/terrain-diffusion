import os
import torch
from tqdm import tqdm
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMUnet2D, EDMAutoencoder
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

autoencoder = EDMAutoencoder.from_pretrained('checkpoints/models/encoder').to(device)
model = get_model('checkpoints/diffusion_base-128x3/latest_checkpoint', sigma_rel=0.05).to(device)

# Enable parallel processing on CPU
torch.set_num_threads(16)

dataset = H5LatentsDataset('dataset.h5', 64, [[0, 1], [0, 1], [0, 1]], [90, 180, 360], [0, 0, 1], [0, 1, 2], eval_dataset=False,
                                   latents_mean=[0, 0, 0, 0],
                                   latents_std=[1, 1, 1, 1],
                                   cond_p_mean=0,
                                   cond_p_std=0)

dataloader = DataLoader(dataset, batch_size=9)

torch.set_grad_enabled(False)

for batch in dataloader:
    # Experiment with different guidance scales
    images = batch['image'].to(device)
    cond_img = batch.get('cond_img').to(device)
    conditional_inputs = recursive_to(batch.get('cond_inputs'), device)
    images_np = images.squeeze().cpu().numpy()
    
    scheduler.set_timesteps(15)
    samples = torch.randn(images.shape, device=device) * scheduler.sigmas[0]
    sigma_data = scheduler.config.sigma_data
    
    i = 0
    for t, sigma in tqdm(zip(scheduler.timesteps, scheduler.sigmas)):
        sigma, t = sigma.to(device), t.to(device)
        scaled_input = scheduler.precondition_inputs(samples, sigma)
        cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1)).expand(samples.shape[0])
        
        # Get predictions from both models
        x = torch.cat([scaled_input, cond_img], dim=1)
        model_output = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs)
        
        samples = scheduler.step(model_output, t, samples).prev_sample
        i += 1
    
    samples = samples / 0.5
    lowfreq = samples[:, -1] * 2353 - 2128
    decoded = autoencoder.decode(samples[:, :-1])[:, 0] * 401 + 88.8
    interp_lowfreq = torch.nn.functional.interpolate(lowfreq.unsqueeze(1), size=decoded.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
    decoded = interp_lowfreq + decoded
    decoded = decoded.clamp(0, 10000)
    
    # Plot decoded images
    import matplotlib.pyplot as plt

    # Ensure the decoded images are on CPU and converted to numpy
    decoded_np = decoded.squeeze().cpu().numpy()

    # Create a grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle('Decoded Diffusion Samples')

    # Plot each decoded image
    for i in range(min(len(decoded_np), 9)):
        img = decoded_np[i]
        
        # Handle single channel images
        if img.shape[-1] == 1:
            img = img.squeeze()
        
        axs[i//3, i%3].imshow(img, cmap='viridis' if img.ndim == 2 else None)
        axs[i//3, i%3].axis('off')

    plt.tight_layout()
    plt.show()