import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from training.datasets.datasets import FileGANDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Create dataset
dataset = FileGANDataset(
    dataset_names=[
        "data/wc2.1_10m_elev.tif",
        "data/wc2.1_10m_bio_1.tif", 
        "data/wc2.1_10m_bio_4.tif",
        "data/wc2.1_10m_bio_12.tif",
        "data/wc2.1_10m_bio_15.tif"
    ],
    crop_size=(30, 30),
    resize_size=(12, 12)
)

# Load 1000 samples and create histogram
print("Loading 1000 samples for histogram...")
all_values = []
for i in tqdm(range(1000)):
    sample = dataset[i]['image'][0]
    all_values.extend(sample.numpy().flatten())

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(all_values, bins=100, density=True)
plt.title('Distribution of Values Across 1000 Samples')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Print some statistics
all_values = np.array(all_values)
print(f"\nData Statistics:")
print(f"Mean: {np.mean(all_values):.3f}")
print(f"Std: {np.std(all_values):.3f}")
print(f"Min: {np.min(all_values):.3f}")
print(f"Max: {np.max(all_values):.3f}")
print(f"25th percentile: {np.percentile(all_values, 25):.3f}")
print(f"Median: {np.median(all_values):.3f}")
print(f"75th percentile: {np.percentile(all_values, 75):.3f}")

while True:
    # Visualize a few samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(2):
        for j in range(5):
            sample = dataset[i*5 + j]['image']
            axes[i,j].imshow(sample[0].numpy())
            axes[i,j].axis('off')
            axes[i,j].set_title(f'Sample {i*5 + j}')
    plt.tight_layout()
    plt.show()
