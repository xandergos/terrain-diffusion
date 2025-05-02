import os
import torch
from tqdm import tqdm
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMUnet2D, EDMAutoencoder
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import *

scheduler = EDMDPMSolverMultistepScheduler(0.002, 80.0, 0.5)

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

autoencoder = EDMAutoencoder.from_pretrained('checkpoints/models/autoencoder').to(device)
model = get_model('checkpoints/diffusion_base-256x3-simple/latest_checkpoint', sigma_rel=0.05).to(device)

# Enable parallel processing on CPU
torch.set_num_threads(16)

dataset = H5LatentsSimpleDataset('dataset.h5', 64, [[0.3, 0.5]], [90], [1], eval_dataset=False,
                                   latents_mean=[0, 0, 0, 0],
                                   latents_std=[1, 1, 1, 1],
                                   sigma_data=0.5,
                                   split="train",
                                   beauty_dist=[[1, 1, 1, 1, 1]])

dataloader = DataLoader(dataset, batch_size=9)

torch.set_grad_enabled(False)

for batch in dataloader:
    # Experiment with different guidance scales
    images = batch['image'].to(device)
    cond_img = recursive_to(batch.get('cond_img'), device)
    conditional_inputs = recursive_to(batch.get('cond_inputs'), device)
    images_np = images.squeeze().cpu().numpy()
    
    print("c: ", conditional_inputs)
    
    scheduler.set_timesteps(32)
    samples = torch.randn(images.shape, device=device) * scheduler.sigmas[0]
    sigma_data = scheduler.config.sigma_data
    
    # Create a list to store pred_x0 at each timestep
    pred_x0_history = []
    
    i = 0
    for t, sigma in tqdm(zip(scheduler.timesteps, scheduler.sigmas)):
        sigma, t = sigma.to(device), t.to(device)
        scaled_input = scheduler.precondition_inputs(samples, sigma)
        cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1)).expand(samples.shape[0])
        
        # Get predictions from both models
        x = scaled_input
        if cond_img is not None:
            x = torch.cat([scaled_input, cond_img], dim=1)
        model_output = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs)
        
        # Calculate pred_x0 using the formula: sin(cnoise) * samples + cos(cnoise) * model_output
        pred_x0 = torch.cos(cnoise.view(-1, 1, 1, 1)) * samples + torch.sin(cnoise.view(-1, 1, 1, 1)) * model_output * 0.5
        
        # Store pred_x0 for visualization
        pred_x0_history.append(pred_x0.detach().cpu().clone())
        
        samples = scheduler.step(model_output, t, samples).prev_sample
        i += 1
        
    pred_x0_history = torch.stack(pred_x0_history, dim=0)
    
    pred_x0_history = pred_x0_history * 2
    latent = pred_x0_history[:, :, :4]
    lowfreq = pred_x0_history[:, :, 4:5]
    residuals = []
    watercovers = []
    for i in range(pred_x0_history.shape[0]):
        decoded = autoencoder.decode(latent[i].cuda()).cpu()
        residual, watercover = decoded[:, :1], decoded[:, 1:2]
        residuals.append(residual)
        watercovers.append(watercover)
    residual = torch.stack(residuals, dim=0)
    watercover = torch.stack(watercovers, dim=0)
    watercover = torch.sigmoid(watercover)
    residual = dataset.denormalize_residual(residual, 90)
    lowfreq = dataset.denormalize_lowfreq(lowfreq, 90)
    decoded_terrain = laplacian_decode(residual.view(-1, 1, 512, 512), lowfreq.view(-1, 1, 64, 64)).view(residual.shape)
    
    # Plot the decoded terrain with a slider for the time dimension
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    # Get dimensions
    num_timesteps, batch_size, channels, height, width = decoded_terrain.shape
    
    # Create a figure with subplots for each batch
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)  # Make room for the slider
    fig.suptitle('Decoded Terrain Evolution Over Time')
    
    # Flatten the axes array for easier indexing
    axs = axs.flatten()
    
    # Create a list to store image objects for updates
    im_list = []
    
    # Initial timestep to display
    time_index = 0
    
    # Plot each batch at the initial timestep
    for i in range(min(batch_size, 9)):
        ax = axs[i]
        img = decoded_terrain[time_index, i, 0].cpu().numpy()
        clim = (img.min(), img.max())
        im = ax.imshow(img, cmap='terrain', vmin=clim[0], vmax=clim[1])
        ax.set_title(f'Batch {i+1}')
        ax.axis('off')
        im_list.append(im)
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.6])
    fig.colorbar(im_list[0], cax=cbar_ax)
    
    # Create a slider for time dimension
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Timestep', 0, num_timesteps - 1, valinit=0, valstep=1)
    
    # Update function for the slider
    def update(val):
        time_idx = int(slider.val)
        for i in range(min(batch_size, 9)):
            img = decoded_terrain[time_idx, i, 0].cpu().numpy()
            clim = (img.min(), img.max())
            im_list[i].set_data(img)
            im_list[i].set_clim(clim[0], clim[1])
        fig.canvas.draw_idle()
    
    # Register the update function with the slider
    slider.on_changed(update)
    
    plt.show()