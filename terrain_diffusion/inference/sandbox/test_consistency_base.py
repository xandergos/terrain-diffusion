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
model = get_model('checkpoints/consistency_base-192x3/latest_checkpoint', sigma_rel=0.1).to(device)

# Enable parallel processing on CPU
torch.set_num_threads(16)

dataset = H5LatentsSimpleDataset('dataset.h5', 64, [[0.1, 1.0]], [90], [1], eval_dataset=False,
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
    
    sigma_data = 0.5
    pred_x0 = torch.zeros_like(images)
    noise = torch.randn_like(images) * sigma_data
    
    i = 0
    for t in [np.arctan(160.0), 1.55, 1.5, 1.3, 1.1]:
        t = torch.tensor(t).to(device).float()
        xt = torch.cos(t) * pred_x0 + torch.sin(t) * noise
        scaled_input = xt / sigma_data
        
        # Get predictions from both models
        x = scaled_input
        if cond_img is not None:
            x = torch.cat([scaled_input, cond_img], dim=1)
        model_output = model(x, noise_labels=t.view(1).expand(x.shape[0]), conditional_inputs=conditional_inputs)
        
        pred_x0 = torch.cos(t) * xt + torch.sin(t) * model_output * sigma_data
        noise = torch.randn_like(images) * sigma_data
        i += 1
    samples = pred_x0
    
    print(torch.std(samples).item())
    samples = samples * 2
    latent = samples[:, :4]
    lowfreq = samples[:, 4:5]
    decoded = autoencoder.decode(latent)
    residual, watercover = decoded[:, :1], decoded[:, 1:2]
    watercover = torch.sigmoid(watercover)
    residual = dataset.denormalize_residual(residual, 90)
    lowfreq = dataset.denormalize_lowfreq(lowfreq, 90)
    residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
    decoded_terrain = laplacian_decode(residual, lowfreq)
    #decoded_terrain = lowfreq
    #decoded_terrain = torch.sign(decoded_terrain) * decoded_terrain**2
    
    plot_images = torch.cat([decoded_terrain, watercover], dim=1)
    
    # Plot all channels interactively using a slider to select the channel.
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    # Assume that plot_images has shape: (batch_size, num_channels, height, width)
    # Convert plot_images from tensor to a numpy array.
    # If necessary, squeeze batch dimensions only if batch_size==1. Here, we assume batch_size == 9.
    plot_images_np = plot_images.cpu().numpy()  # shape: (9, num_channels, H, W)
    
    # Create an initial grid of subplots for the 9 images and reserve space for the slider.
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    # Adjust the layout to make room for the slider.
    plt.subplots_adjust(bottom=0.25)
    fig.suptitle('Decoded Diffusion Samples - Channel Slider')
    
    # Create a list to store the image objects for later updates.
    im_list = []
    # We start with channel 0.
    channel_index = 0
    batch_size, num_channels, H, W = plot_images_np.shape
    
    # Plot each image in the grid using the selected channel.
    for i in range(min(batch_size, 9)):
        ax = axs[i // 3, i % 3]
        # For each image, pick the current channel.
        img = plot_images_np[i, channel_index]
        # Compute the min and max to update the color limits.
        clim = (img.min(), img.max())
        im = ax.imshow(img, cmap='viridis', vmin=clim[0], vmax=clim[1])
        ax.axis('off')
        im_list.append(im)
    
    # Create a slider axis below the subplots.
    axslider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height] in figure coordinates.
    # Slider ranges from 0 to num_channels-1 with an integer step.
    slider = Slider(axslider, 'Channel', 0, num_channels - 1, valinit=0, valstep=1)
    
    def update(val):
        # Get the current channel from the slider and update each image.
        ch = int(slider.val)
        for i in range(min(batch_size, 9)):
            img = plot_images_np[i, ch]
            clim = (img.min(), img.max())
            im_list[i].set_data(img)
            # Update the color limits so the contrast adjusts per channel.
            im_list[i].set_clim(clim[0], clim[1])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.show()