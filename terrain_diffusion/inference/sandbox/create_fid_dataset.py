import os
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
from diffusion.encoder import LaplacianPyramidEncoder
from PIL import Image

folder = '/mnt/ntfs2/shared/data/terrain/datasets/generative_land_data_composite_elv/'

land_weight = 10
ocean_weight = 1

land_files = []
ocean_files = []

for file in tqdm(os.listdir(folder)):
    img = tiff.imread(os.path.join(folder, file)).astype(np.float32)
    img = F.adaptive_avg_pool2d(torch.from_numpy(img)[None], (64, 64))
    pct_land = torch.mean((img > 0).float()).item()
    if pct_land > 0.9999:
        land_files.append(file)
    else:
        ocean_files.append(file)
        
# Process and save each augmented image
# Create a directory to store the augmented images if it doesn't exist
output_dir = 'fid/true_64'
for prev_file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, prev_file))
os.makedirs(output_dir, exist_ok=True)

for file in tqdm(land_files + ocean_files[:round(len(land_files) * ((land_weight + ocean_weight) / land_weight - 1))]):
    img = tiff.imread(os.path.join(folder, file)).astype(np.float32)
    img = F.adaptive_avg_pool2d(torch.from_numpy(img)[None], (64, 64))
    # Create a list to store all augmented images
    augmented_images = []
    flipped = torch.flip(img, [1])
    for k in range(4):
        rotated = torch.rot90(img, k, [1, 2])
        augmented_images.append(rotated)
        
        rotated_flipped = torch.rot90(flipped, k, [1, 2])
        augmented_images.append(rotated_flipped)


    # Process and save each augmented image
    for i, augmented_img in enumerate(augmented_images):
        # Normalize the image to be in the range 0-1
        normalized_img = (augmented_img - augmented_img.min()) / (augmented_img.max() - augmented_img.min())
        
        # Convert to numpy array and transpose dimensions for PIL
        img_np = normalized_img.numpy()[0]
        
        # Convert to PIL Image
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        # Save as PNG
        output_filename = f"{os.path.splitext(file)[0]}_aug_{i}.png"
        img_pil.save(os.path.join(output_dir, output_filename))
    

print(len(land_files), len(ocean_files))
