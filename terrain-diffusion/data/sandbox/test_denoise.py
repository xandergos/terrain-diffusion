import json
import os
from functools import partial
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


import numpy as np

from diffusion.datasets.datasets import BaseTerrainDataset
import scipy.optimize

from diffusion.encoder import denoise_pyramid, denoise_pyramid_layer


if __name__ == "__main__":
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
    dataset = BaseTerrainDataset(paths_land, 64, 64, [1], [5], [0, 0], [0.5, 0.5], eval_dataset=True)

    for i in range(len(dataset)):
        # Get the first image from the dataset
        first_image = dataset[i]['image']
        
        # Extract the second channel (index 1)
        second_channel = first_image[1].unsqueeze(0)  # Add channel dimension
        
        # Add noise to the image
        noise = torch.randn_like(second_channel) * 0.001 * (2420 / 0.5)
        noisy_image = second_channel + noise
        
        full_noisy_image = torch.cat([first_image[0].unsqueeze(0), noisy_image], dim=0)

        encoder = dataset.encoder
        
        start_time = time.time()
        optimized_x = denoise_pyramid(full_noisy_image, encoder)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

        # Compute the final loss
        final_loss = torch.nn.functional.mse_loss(optimized_x[1], noisy_image)

        print(f"Final loss: {final_loss.item()}")
        
        import matplotlib.pyplot as plt
        
        # Decode the images
        decoded_noisy = encoder.decode(full_noisy_image).squeeze().cpu().numpy()
        decoded_optimized = encoder.decode(optimized_x).squeeze().cpu().numpy()
        
        # Compute the difference
        difference = decoded_optimized - encoder.decode(first_image).squeeze().cpu().numpy()
        
        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot the original decoded noisy image
        im1 = ax1.imshow(decoded_noisy, cmap='viridis')
        ax1.set_title('Original Noisy Image')
        fig.colorbar(im1, ax=ax1)
        
        # Plot the decoded optimized image
        im2 = ax2.imshow(decoded_optimized, cmap='viridis')
        ax2.set_title('Optimized Image')
        fig.colorbar(im2, ax=ax2)
        
        # Plot the difference
        im3 = ax3.imshow(difference, cmap='bwr')
        ax3.set_title('Difference')
        fig.colorbar(im3, ax=ax3)
        
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        
