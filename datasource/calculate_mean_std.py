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

def calculate_mean_std(dataset, batch_size=32, num_workers=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    channels = dataset[0]['image'].shape[-3]

    # Initialize aggregates for each channel
    count = 0
    mean = np.zeros(channels)
    M2 = np.zeros(channels)

    for batch in tqdm(dataloader, desc="Calculating mean and std"):
        # Combine all three channels into a single array
        batch = batch['image']
        batch_np = batch.numpy()
        batch_size, num_channels, height, width = batch_np.shape
        
        # Reshape to (batch_size * height * width, num_channels)
        batch_flat = batch_np.transpose(0, 2, 3, 1).reshape(-1, channels)
        
        # Update aggregates
        count += batch_flat.shape[0]
        delta = batch_flat - mean
        mean += np.sum(delta, axis=0) / count
        delta2 = batch_flat - mean
        M2 += np.sum(delta * delta2, axis=0)

    # Calculate final statistics
    mean_result = mean
    std_result = np.sqrt(M2 / (count - 1))

    return mean_result, std_result
    

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
    
    # 1024 @ 4x2s: [57, 1110]
    # 256 @ 4x2s: [108, 1166]
    # 64 @ 4x2s: [202, 1130]

    # 1024 @ 0s: [23.4, 50.4, 147.9, 1085]
    # 256 @ 0s: [52, 152, 1151]
    # 64 @ 0s: [155, 1151]
    dataset = BaseTerrainDataset(paths_land, 256, 64, [1], [5], [0, 0], [0.5, 0.5], eval_dataset=True)

    print(calculate_mean_std(dataset))
