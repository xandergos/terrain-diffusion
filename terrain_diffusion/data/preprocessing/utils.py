"""Shared utilities for preprocessing elevation data."""

import skimage
import torch
import torch.nn.functional as F
import tifffile as tiff
import os
import numpy as np
import rasterio
import scipy.interpolate
from terrain_diffusion.data.laplacian_encoder import laplacian_encode

def read_raster(file):
    with rasterio.open(file) as src:
        data = src.read(1)
        return data

def process_single_file_base(
    chunk_id, 
    highres_elevation_folder, 
    lowres_elevation_folder, 
    highres_size, 
    lowres_size,
    lowres_sigma,
    num_chunks=1,
    landcover_folder=None,
    watercover_folder=None,
):
    """
    Process a single elevation file and return the preprocessed chunks.
    
    Args:
        chunk_id (int|str): The chunk id to process.
        highres_elevation_folder (str): Path to the folder containing high-resolution elevation files.
        lowres_elevation_folder (str): Path to the folder containing low-resolution elevation files.
        highres_size (int): The size of the high-resolution images.
        lowres_size (int): The size of the low-resolution images.
        num_chunks (int): The number of chunks to divide the image into for processing. (Default: 1)
        landcover_folder (str): Path to the folder containing land cover data. (Optional)
        watercover_folder (str): Path to the folder containing water cover data. (Optional)
    
    Returns:
        list: List of dictionaries containing preprocessed chunks and metadata
    """
    file = chunk_id + '.tif'
    highres_dem = read_raster(os.path.join(highres_elevation_folder, file)).astype(np.float32)
    if not np.isnan(highres_dem).all():
        highres_dem = skimage.transform.resize(highres_dem, (highres_size, highres_size), order=0, preserve_range=True)
    else:
        highres_dem = np.nan * np.ones((highres_size, highres_size), dtype=np.float32)
    
    base_lowres_dem = read_raster(os.path.join(lowres_elevation_folder, file)).astype(np.float32)
    base_lowres_dem = skimage.transform.resize(base_lowres_dem, (lowres_size, lowres_size), order=1, preserve_range=True)
    base_lowres_dem = skimage.filters.gaussian(base_lowres_dem, sigma=lowres_sigma)
    
    scaled_base_lowres_dem = skimage.transform.resize(base_lowres_dem, (highres_size, highres_size), order=1, preserve_range=True)
    
    # Replace NaN values over 10 pixels from non-NaN pixels with 0
    if np.isnan(highres_dem).all():
        highres_dem = np.zeros_like(highres_dem)
    elif np.isnan(highres_dem).any():
        # Create a mask of non-NaN pixels
        nan_mask = np.isnan(highres_dem)
        distance = scipy.ndimage.distance_transform_edt(nan_mask)
        alpha = np.minimum(1, distance[nan_mask]/32)
        highres_dem[nan_mask] = scaled_base_lowres_dem[nan_mask] * alpha

    if landcover_folder is not None:
        try:
            landcover = read_raster(os.path.join(landcover_folder, file)).astype(np.float32)
            landcover = skimage.transform.resize(landcover, (highres_size, highres_size), order=0, preserve_range=True)
        except Exception:
            print(f"Error reading landcover file {file}")
            landcover = None
    else:
        landcover = None
    
    if watercover_folder is not None:
        try:
            watercover = read_raster(os.path.join(watercover_folder, file)).astype(np.float32)
            watercover = skimage.transform.resize(watercover, (highres_size, highres_size), order=0, preserve_range=True)
        except Exception:
            print(f"Error reading watercover file {file}")
            watercover = None
    else:
        watercover = None
        
    residual, lowres_dem = laplacian_encode(highres_dem, lowres_size, lowres_sigma)
        
    highres_chunk_size = highres_size // num_chunks
    lowres_chunk_size = lowres_size // num_chunks
    chunks_data = []
    
    num_chunks = (highres_size + highres_chunk_size - 1) // highres_chunk_size
    for chunk_h in range(num_chunks):
        for chunk_w in range(num_chunks):
            highres_start_h = chunk_h * highres_chunk_size
            highres_start_w = chunk_w * highres_chunk_size
            highres_end_h = min(highres_start_h + highres_chunk_size, highres_size)
            highres_end_w = min(highres_start_w + highres_chunk_size, highres_size)
            
            lowres_start_h = chunk_h * lowres_chunk_size
            lowres_start_w = chunk_w * lowres_chunk_size
            lowres_end_h = min(lowres_start_h + lowres_chunk_size, lowres_size)
            lowres_end_w = min(lowres_start_w + lowres_chunk_size, lowres_size)
            
            pct_land = np.mean(lowres_dem[..., lowres_start_h:lowres_end_h, lowres_start_w:lowres_end_w] > 0)
            
            chunks_data.append({
                'residual': residual[..., highres_start_h:highres_end_h, highres_start_w:highres_end_w],
                'highfreq': highres_dem[..., highres_start_h:highres_end_h, highres_start_w:highres_end_w],
                'lowfreq': lowres_dem[..., lowres_start_h:lowres_end_h, lowres_start_w:lowres_end_w],
                'landcover': landcover[..., highres_start_h:highres_end_h, highres_start_w:highres_end_w] if landcover is not None else None,
                'watercover': watercover[..., highres_start_h:highres_end_h, highres_start_w:highres_end_w] if watercover is not None else None,
                'pct_land': pct_land,
                'chunk_id': chunk_id,
                'subchunk_id': f'chunk_{chunk_h}_{chunk_w}'
            })
    
    return chunks_data


class ElevationDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 highres_elevation_folder, 
                 lowres_elevation_folder, 
                 highres_size, 
                 lowres_size,
                 lowres_sigma,
                 num_chunks=1,
                 landcover_folder=None,
                 watercover_folder=None,
                 skip_chunk_ids=None):
        self.chunk_ids = list(map(lambda x: x.split('.')[0], os.listdir(highres_elevation_folder)))
        self.highres_elevation_folder = highres_elevation_folder
        self.lowres_elevation_folder = lowres_elevation_folder
        self.highres_size = highres_size
        self.lowres_size = lowres_size
        self.lowres_sigma = lowres_sigma
        self.num_chunks = num_chunks
        self.landcover_folder = landcover_folder
        self.watercover_folder = watercover_folder
        if skip_chunk_ids is not None:
            skip_chunk_ids = set(str(x) for x in skip_chunk_ids)
            self.chunk_ids = [f for f in self.chunk_ids if str(f) not in skip_chunk_ids]
    
    def __len__(self):
        return len(self.chunk_ids)
    
    def __getitem__(self, idx):
        cid = self.chunk_ids[idx]
        return process_single_file_base(cid, 
                                        self.highres_elevation_folder, 
                                        self.lowres_elevation_folder, 
                                        self.highres_size, 
                                        self.lowres_size,
                                        self.lowres_sigma,
                                        self.num_chunks, 
                                        self.landcover_folder, 
                                        self.watercover_folder)