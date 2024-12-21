"""Shared utilities for preprocessing elevation data."""

import torch
import torch.nn.functional as F
import tifffile as tiff
import os
import numpy as np

def preprocess_elevation(img, image_size):
    """Preprocess elevation data by resizing and handling ocean/land masks."""
    if len(img.shape) == 2:
        img = torch.from_numpy(img)
        img = F.interpolate(img[None, None], (image_size, image_size), mode='area')[0, 0]
        return img.numpy()
    ocean = torch.from_numpy(img[:, :, 1]) if img.shape[2] == 2 else torch.from_numpy(np.zeros_like(img[:, :, 0]))
    land = torch.from_numpy(img[:, :, 0])

    land = F.interpolate(land[None, None], (image_size, image_size), mode='area')[0, 0]
    ocean = F.interpolate(ocean[None, None], (image_size, image_size), mode='area')[0, 0]
    ocean = torch.minimum(ocean, torch.tensor(-1.0))

    land_mask = land > 0
    
    ocean = F.adaptive_avg_pool2d(ocean[None], (256, 256))
    ocean = F.interpolate(ocean[None], (image_size, image_size), mode='bicubic')[0, 0]
    ocean = torch.minimum(ocean, torch.tensor(-1.0))
    
    img = land * land_mask.float() + ocean * (1 - land_mask.float())
    
    return img.numpy()

def process_single_file_base(file, elevation_folder, image_size, base_resolution, target_resolution, num_chunks, encoder):
    """
    Process a single elevation file and return the preprocessed chunks.
    
    Args:
        file (str): Filename to process
        elevation_folder (str): Path to folder containing elevation files
        image_size (int): Size to resize images to
        base_resolution (int): Resolution of input images in meters
        target_resolution (int): Target resolution in meters
        num_chunks (int): Number of chunks to split the image into
        encoder: LaplacianPyramidEncoder instance
    
    Returns:
        list: List of dictionaries containing preprocessed chunks and metadata
    """
    img = tiff.imread(os.path.join(elevation_folder, file)).astype(np.float32)
    if len(img.shape) == 2 and np.any(img <= 0):
        return []
        
    img = preprocess_elevation(img, image_size)
    
    downsample_factor = target_resolution // base_resolution
    img = F.adaptive_avg_pool2d(torch.from_numpy(img)[None], 
                              (img.shape[0] // downsample_factor, img.shape[1] // downsample_factor))
    
    h, w = img.shape[-2:]
    chunk_size = h // num_chunks
    chunks_data = []
    
    for chunk_h in range((h + chunk_size - 1) // chunk_size):
        for chunk_w in range((w + chunk_size - 1) // chunk_size):
            start_h = chunk_h * chunk_size
            start_w = chunk_w * chunk_size
            end_h = min(start_h + chunk_size, h)
            end_w = min(start_w + chunk_size, w)
            
            img_chunk = img[..., start_h:end_h, start_w:end_w]
            pct_land = torch.mean((img_chunk > 0).float()).item()
            
            encoded, downsampled_encoding = encoder.encode(img_chunk, return_downsampled=True)
            
            chunks_data.append({
                'highfreq': encoded[:1],
                'lowfreq': downsampled_encoding[-1],
                'pct_land': pct_land,
                'filename': file,
                'chunk_id': f'chunk_{chunk_h}_{chunk_w}'
            })
    
    return chunks_data

class ElevationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_files, elevation_folder, image_size, base_resolution, target_resolution, num_chunks, encoder):
        self.dataset_files = dataset_files
        self.elevation_folder = elevation_folder
        self.image_size = image_size
        self.base_resolution = base_resolution
        self.target_resolution = target_resolution
        self.num_chunks = num_chunks
        self.encoder = encoder
    
    def __len__(self):
        return len(self.dataset_files)
    
    def __getitem__(self, idx):
        file = self.dataset_files[idx]
        return process_single_file_base(file, 
                                        self.elevation_folder, 
                                        self.image_size, 
                                        self.base_resolution, 
                                        self.target_resolution, 
                                        self.num_chunks, 
                                        self.encoder)