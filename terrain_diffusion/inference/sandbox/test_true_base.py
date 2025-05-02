import os
import torch
from terrain_diffusion.training.unet import EDMAutoencoder
from safetensors.torch import load_model
from terrain_diffusion.training.datasets.datasets import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import *

device = 'cuda'

# Enable parallel processing on CPU
torch.set_num_threads(16)

# Load autoencoder for decoding
autoencoder = EDMAutoencoder.from_pretrained('checkpoints/models/autoencoder').to(device)

dataset = H5LatentsSimpleDataset('dataset.h5', 64, [[0.3, 0.5]], [90], [1], eval_dataset=False,
                                 latents_mean=[0, 0, 0, 0],
                                 latents_std=[1, 1, 1, 1],
                                 sigma_data=0.5,
                                 split="train",
                                 beauty_dist=[[1, 1, 1, 1, 1]])

dataloader = DataLoader(dataset, batch_size=9)

torch.set_grad_enabled(False)

for batch in dataloader:
    # Get images directly from the dataset
    print(batch['path'])
    images = batch['image'].to(device)
    cond_img = recursive_to(batch.get('cond_img'), device)
    conditional_inputs = recursive_to(batch.get('cond_inputs'), device)
    
    print("c: ", conditional_inputs)
    
    # Extract latent and lowfreq directly from the dataset images
    samples = images / 0.5  # Undo the normalization that was in the original code
    latent = samples[:, :4]
    lowfreq = samples[:, 4:5]
    
    # Decode the latent using the autoencoder
    decoded = autoencoder.decode(latent)
    residual, watercover = decoded[:, :1], decoded[:, 1:2]
    watercover = torch.sigmoid(watercover)
    residual = dataset.denormalize_residual(residual, 90)
    lowfreq = dataset.denormalize_lowfreq(lowfreq, 90)
    #residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
    decoded_terrain = laplacian_decode(residual, lowfreq)
    
    plot_images = torch.cat([decoded_terrain, watercover], dim=1)
    
    # Plot all channels interactively using a slider to select the channel.
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    # Convert plot_images from tensor to a numpy array
    plot_images_np = plot_images.cpu().numpy()  # shape: (9, num_channels, H, W)
    
    # Create an initial grid of subplots for the 9 images and reserve space for the slider
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    # Adjust the layout to make room for the slider
    plt.subplots_adjust(bottom=0.25)
    fig.suptitle('Original Dataset Samples - Channel Slider')
    
    # Create a list to store the image objects for later updates
    im_list = []
    # We start with channel 0
    channel_index = 0
    batch_size, num_channels, H, W = plot_images_np.shape
    
    # Plot each image in the grid using the selected channel
    for i in range(min(batch_size, 9)):
        ax = axs[i // 3, i % 3]
        # For each image, pick the current channel
        img = plot_images_np[i, channel_index]
        # Compute the min and max to update the color limits
        clim = (img.min(), img.max())
        im = ax.imshow(img, cmap='viridis', vmin=clim[0], vmax=clim[1])
        ax.axis('off')
        im_list.append(im)
    
    # Create a slider axis below the subplots
    axslider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height] in figure coordinates
    # Slider ranges from 0 to num_channels-1 with an integer step
    slider = Slider(axslider, 'Channel', 0, num_channels - 1, valinit=0, valstep=1)
    
    def update(val):
        # Get the current channel from the slider and update each image
        ch = int(slider.val)
        for i in range(min(batch_size, 9)):
            img = plot_images_np[i, ch]
            clim = (img.min(), img.max())
            im_list[i].set_data(img)
            # Update the color limits so the contrast adjusts per channel
            im_list[i].set_clim(clim[0], clim[1])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.show()
    
    # Break after the first batch to match the original behavior
    break