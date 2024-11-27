import os
import json
import random
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from diffusion.datasets.datasets import BaseTerrainDataset, SuperresTerrainDataset
from torchvision.transforms.v2 import functional as TF

folder = '/mnt/ntfs2/shared/data/terrain/datasets/generative_land_data_composite_elv/'
paths = [os.path.join(folder, p) for p in os.listdir(folder)]
random.seed(385)
random.shuffle(paths)
train_paths = paths

if os.path.exists('paths_land.json'):
    with open('paths_land.json') as f:
        paths_land = json.load(f)
else:
    paths_land = []
    for path in tqdm(paths):
        im = np.array(Image.open(path)).astype(np.float32)
        if np.count_nonzero(im > 0) > 0.1 * 1024 ** 2:
            paths_land.append(os.path.basename(path))
    with open('paths_land.json', 'w') as f:
        json.dump(paths_land, f)
paths_land = [os.path.join(folder, p) for p in paths_land]
# 1024 @ 1s: [37.4, 62.8, 192.2, 1085]
# 1024 @ 2s: [58.2, 84, 284, 1085]
# 1024 @ 0s: [23.4, 50.4, 147.9, 1085]
# 256 @ 0s: [52, 152, 1151]
# 64 @ 0s: [155, 1151]
dataset = SuperresTerrainDataset(paths_land, 256, 64, [128], [0], [0, -2651], [155, 2420], upsample_factor=4, 
                                 eval_dataset=False, noise_scale=0.1)

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def visualize_terrains(dataset, num_samples=10):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            break
        
        encoded_image = sample['cond_img']
        image = sample['image'][0, 0]
        
        # Plot the original image and both channels of the encoded image
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Remove axes and margins
        for ax in axs:
            ax.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        
        # Plot original image
        axs[0].imshow(image.cpu().numpy(), cmap='gray')
        
        # Plot encoded image channels
        for j in range(2):
            axs[j+1].imshow(encoded_image[0, j].cpu().numpy(), cmap='gray')
        
        plt.show()
        
        # Ensure the image is in the correct range [0, 1]
        def plot_images():
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            samples_np = image.cpu().numpy()

            def normalize_data(x):
                return np.sign(x) * np.sqrt(np.abs(x))

            # Normalize the data
            data = samples_np
            normalized_data = normalize_data(data)
            
            # Solving min + t * (max - min) = 0 for t
            min_height = -9000
            max_height = 8000
            center = -normalize_data(min_height) / (normalize_data(max_height) - normalize_data(min_height))
            
            # Create a colorscale that sharply transitions from yellow to blue at 0
            colors_below = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffd9', '#addd8e', '#8c6d31', '#969696', '#ffffff']
            ticks = [t / 4 * center for t in range(5)] + [t / 4 * (1 - center) + center for t in range(5)]

            # Create custom colormap
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", list(zip(ticks, colors_below)))

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create the heatmap
            im = ax.imshow(normalized_data, cmap=custom_cmap,
                           vmin=normalize_data(min_height), vmax=normalize_data(max_height))

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label("Elevation (Nonlinear)", rotation=270, labelpad=15)

            # Set colorbar ticks
            cbar.set_ticks([normalized_data.min(), 0, normalized_data.max()])
            cbar.set_ticklabels([f" {int(data.min())}", "0", f" {int(data.max())}"])

            # Update the layout for a more appealing look
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            plt.tight_layout()
            #plt.show()
            
            # Save the figure without the colorbar, ensuring only the image is saved
            cbar.remove()  # Remove the colorbar from the figure
            ax.set_frame_on(False)  # Remove the axes frame
            plt.axis('off')  # Turn off the axis
            plt.savefig(f"outputs/real_samples/image{i}.png", bbox_inches='tight', pad_inches=0, transparent=True)
            
        #plot_images()

# Visualize 10 random terrain samples
visualize_terrains(dataset, num_samples=36)
