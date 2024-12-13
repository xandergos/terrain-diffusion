import json
import os
from functools import partial
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

import torchvision.transforms.v2 as T


import numpy as np

from diffusion.datasets.datasets import BaseTerrainDataset

def has_water(dataset, batch_size=32, num_workers=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    water_percentages = []

    for batch in tqdm(dataloader, desc="Calculating water percentages"):
        # Assuming the elevation data is in the first channel
        batch = batch['image']
        elevation = batch[:, 0, :, :]
        
        # Count pixels less than 0 (water) for each image in the batch
        water_pixels = (elevation < 0).sum(dim=(1, 2))
        total_pixels = elevation.shape[1] * elevation.shape[2]
        
        # Calculate percentage of water for each image
        batch_percentages = (water_pixels / total_pixels * 100).tolist()
        water_percentages.extend(batch_percentages)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(water_percentages, bins=10, edgecolor='black')
    plt.title('Histogram of Water Percentage in Images')
    plt.xlabel('Percentage of Water')
    plt.ylabel('Number of Images')
    plt.savefig('water_percentage_histogram.png')
    plt.close()

    return f"Histogram of water percentages saved as 'water_percentage_histogram.png'"
    
def calculate_mean_histogram(dataset, batch_size=32, num_workers=16, num_bins=50):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    mean_values = []

    for batch in tqdm(dataloader, desc="Calculating mean values"):
        # Assuming all channels are relevant for the mean calculation
        batch = (batch['image'] + 2561) / 2420 / 2
        
        # Calculate mean for each image in the batch
        batch_means = batch.mean(dim=(1, 2, 3)).tolist()
        mean_values.extend(batch_means)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(mean_values, bins=num_bins, edgecolor='black')
    plt.title('Histogram of Mean Pixel Values in Images')
    plt.xlabel('Mean Pixel Value')
    plt.ylabel('Number of Images')
    plt.savefig('mean_values_histogram.png')
    plt.close()

    print(np.mean(mean_values), np.std(mean_values))
    return f"Histogram of mean pixel values saved as 'mean_values_histogram.png'"


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
    dataset = BaseTerrainDataset(paths_land, 64, 64, [], 0, [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], eval_dataset=True)

    print(calculate_mean_histogram(dataset))
