import os
import json
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from diffusion.datasets.datasets import BaseTerrainDataset, H5BaseTerrainDataset
from torchvision.transforms.v2 import functional as TF

from diffusion.encoder import LaplacianPyramidEncoder

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
#dataset = H5BaseTerrainDataset('dataset64.h5', 64, [0.99999, 1], '256', 1, eval_dataset=True)
dataset = BaseTerrainDataset(paths_land, 512, 512, [1], 40, [0, -2651], [80, 2420], eval_dataset=True,
                             root_dir='/mnt/ntfs2/shared/data/terrain/datasets/generative_land_data_composite_elv/')

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

encoder = LaplacianPyramidEncoder([1], 5, [0, -2651], [160, 2420])

def visualize_terrains(dataset, num_samples=10):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            break
        
        encoded_image = sample['image']
        cond_img = sample.get('cond_img', None)
        cond_inputs = sample.get('cond_inputs', None)
        image = encoder.decode(encoded_image)[0, 0]
        
        # Plot the original image, encoded image channels, and conditional image channels
        num_plots = 1 + encoded_image.shape[1] + (cond_img.shape[1] if cond_img is not None else 0)
        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        # Remove axes and margins
        for ax in axs:
            ax.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0, left=0, right=1, top=1, bottom=0)
        
        # Plot original image
        axs[0].imshow(image.cpu().numpy(), cmap='viridis')
        axs[0].set_title('Original Image')
        
        # Plot encoded image channels
        for j in range(encoded_image.shape[1]):
            axs[j+1].imshow(encoded_image[0, j].cpu().numpy(), cmap='viridis')
            axs[j+1].set_title(f'Encoded Channel {j}')
        
        # Plot conditional image channels if available
        if cond_img is not None:
            for j in range(cond_img.shape[1]):
                axs[encoded_image.shape[1]+j+1].imshow(cond_img[0, j].cpu().numpy(), cmap='viridis')
                axs[encoded_image.shape[1]+j+1].set_title(f'Cond Image Channel {j}')
        
        # Print conditional inputs if available
        if cond_inputs is not None:
            print("Conditional Inputs:", cond_inputs)
        plt.show()

# Visualize 10 random terrain samples
visualize_terrains(dataset, num_samples=500)
