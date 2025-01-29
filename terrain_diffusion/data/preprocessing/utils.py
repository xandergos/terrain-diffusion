"""Shared utilities for preprocessing elevation data."""

from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt
import skimage
import torch
import torch.nn.functional as F
import tifffile as tiff
import os
import numpy as np
import rasterio
import scipy.interpolate
from terrain_diffusion.data.downloading.world_grid import create_equal_area_grid
from terrain_diffusion.data.laplacian_encoder import laplacian_encode
from rasterio.merge import merge
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
    
def read_raster(file, include_bounds=False):
    with rasterio.open(file) as src:
        assert src.crs == 'EPSG:4326', "Raster file is not in WGS84. Use extract_mask_from_tiffs instead."
        data = src.read(1)
        if include_bounds:
            bounds = src.bounds
            # Fix for precision issues with longitude
            if (bounds[0] + bounds[2]) / 2 > 180:
                bounds = (bounds[0] - 360, bounds[1], bounds[2] - 360, bounds[3])
            elif (bounds[0] + bounds[2]) / 2 < -180:
                bounds = (bounds[0] + 360, bounds[1], bounds[2] + 360, bounds[3])
            return data, bounds
        else:
            return data
    
def extract_mask_from_tiffs(
    tiff_path: str, 
    bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Extract a mask from TIFF file(s) covering a specific geographical extent.
    Automatically handles CRS transformations to/from WGS84 (EPSG:4326).
    
    Args:
        tiff_path: Path to either a TIFF file or a folder containing TIFF files
        bounds: Tuple of (lon_min, lat_min, lon_max, lat_max) coordinates in WGS84
    
    Returns:
        np.ndarray: The extracted mask as a numpy array
    
    Example:
        bounds = (-90, -180, 90, 180)  # Whole globe (lat_min, lon_min, lat_max, lon_max)
        mask = extract_mask_from_tiffs("path/to/tiffs", bounds)
    """
    path = Path(tiff_path)
    
    # Handle single file case
    if path.is_file():
        with rasterio.open(path) as src:
            # Transform bounds from WGS84 to the source CRS if needed
            if src.crs != 'EPSG:4326':
                transformed_bounds = transform_bounds('EPSG:4326', src.crs, 
                                                   bounds[0], bounds[1], 
                                                   bounds[2], bounds[3])
                window = src.window(*transformed_bounds)
            else:
                window = src.window(bounds[0], bounds[1], 
                                    bounds[2], bounds[3])
            
            data = src.read(1, window=window)
            
            # If the source CRS isn't WGS84, reproject the extracted data
            if src.crs != 'EPSG:4326':
                # Calculate the transform for the window
                window_transform = src.window_transform(window)
                
                # Calculate the target transform and dimensions
                try:
                    dst_crs = 'EPSG:4326'
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, dst_crs, data.shape[1], data.shape[0],
                        *transformed_bounds
                    )
                except Exception:
                    raise
                
                # Initialize the destination array
                dst_data = np.zeros((dst_height, dst_width), dtype=data.dtype)
                
                # Perform the reprojection
                reproject(
                    source=data,
                    destination=dst_data,
                    src_transform=window_transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
                
                if dst_data.dtype == np.float32:
                    dst_data[np.abs(dst_data) > 1e10] = np.nan
                    assert np.all(np.isnan(dst_data)) or np.nanmax(np.abs(dst_data)) <= 1e10, f"Data is out of range: {np.nanmax(np.abs(dst_data))}"
                return dst_data
            
            if data.dtype == np.float32:
                data[np.abs(data) > 1e10] = np.nan
                assert np.all(np.isnan(data)) or np.nanmax(np.abs(data)) <= 1e10, f"Data is out of range: {np.nanmax(np.abs(data))}"
            return data
    elif path.is_dir():
        tiff_files = list(path.glob("*.tif*"))
        src_files_to_merge = []
        
        for tiff in tiff_files:
            with rasterio.open(tiff) as src:
                # Transform bounds to source CRS for intersection check
                if src.crs != 'EPSG:4326':
                    transformed_bounds = transform_bounds('EPSG:4326', src.crs, 
                                                          bounds[0], bounds[1], 
                                                          bounds[2], bounds[3])
                    check_bounds = transformed_bounds
                else:
                    check_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
                
                # Check if the file intersects with our area of interest
                if src.bounds.left < check_bounds[2] and src.bounds.right > check_bounds[0] and \
                   src.bounds.bottom < check_bounds[3] and src.bounds.top > check_bounds[1]:
                    src_files_to_merge.append(src)
        
        if not src_files_to_merge:
            raise ValueError(f"No TIFF files found that intersect with the specified bounds in {tiff_path}")
        
        # Merge the relevant files
        mosaic, out_transform = merge(src_files_to_merge, bounds=bounds)
        
        # If the merged result isn't in WGS84, reproject it
        if src_files_to_merge[0].crs != 'EPSG:4326':
            # Calculate the target transform and dimensions
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_files_to_merge[0].crs, 'EPSG:4326',
                mosaic.shape[2], mosaic.shape[1],
                *transformed_bounds
            )
            
            # Initialize the destination array
            dst_data = np.zeros((dst_height, dst_width), dtype=mosaic.dtype)
            
            # Perform the reprojection
            reproject(
                source=mosaic[0],
                destination=dst_data,
                src_transform=out_transform,
                src_crs=src_files_to_merge[0].crs,
                dst_transform=dst_transform,
                dst_crs='EPSG:4326',
                resampling=Resampling.nearest
            )
            
            return dst_data
            
        return mosaic[0]
    else:
        raise ValueError(f"Input path must be either a TIFF file or a directory containing TIFFs. Got {tiff_path}")
    
def process_single_file_base(
    chunk_id, 
    bounds,
    highres_elevation_folder, 
    lowres_elevation_file, 
    highres_size, 
    lowres_size,
    lowres_sigma,
    num_chunks=1,
    landcover_folder=None,
    watercover_folder=None,
    koppen_geiger_file=None,
    climate_folder=None,
    edge_margin=0,
):
    """
    Process a single elevation file and return the preprocessed chunks.
    
    Args:
        chunk_id (int|str): The chunk id to process.
        grid_cell (Tuple[float, float, float, float]): The grid cell to process.
        highres_elevation_folder (str): Path to the folder containing high-resolution elevation files.
        lowres_elevation_file (str): Path to the file containing low-resolution elevation data.
        highres_size (int): The size of the high-resolution images.
        lowres_size (int): The size of the low-resolution images.
        num_chunks (int): The number of chunks to divide the image into for processing. (Default: 1)
        landcover_folder (str): Path to the folder containing land cover data. (Optional)
        watercover_folder (str): Path to the folder containing water cover data. (Optional)
        koppen_geiger_file (str): Path to the file containing Koppen-Geiger climate data. (Optional)
        climate_folder (str): Path to the folder containing WorldClim climate data. (Optional)
        edge_margin (int): The number of pixels to remove from the edges of the lowres image, automatically scaled up for the highres image.
    
    Returns:
        list: List of dictionaries containing preprocessed chunks and metadata
    """
    file = chunk_id + '.tif'
    
    highres_margin = edge_margin * highres_size // lowres_size
    
    highres_path = os.path.join(highres_elevation_folder, file)
    if os.path.exists(highres_path):
        highres_dem, new_bounds = read_raster(highres_path, include_bounds=True)
        bounds = new_bounds
        highres_dem = np.where(highres_dem == 0.0, np.nan, highres_dem)
        highres_dem = highres_dem.astype(np.float32)
        if not np.isnan(highres_dem).all():
            highres_dem = skimage.transform.resize(highres_dem, (highres_size, highres_size), order=0, preserve_range=True)
        else:
            highres_dem = np.nan * np.ones((highres_size, highres_size), dtype=np.float32)
    else:
        highres_dem = np.nan * np.ones((highres_size, highres_size), dtype=np.float32)
        assert np.isnan(highres_dem).all()
    if highres_margin > 0:
        highres_dem = highres_dem[highres_margin:-highres_margin, highres_margin:-highres_margin]
    
    base_lowres_dem = extract_mask_from_tiffs(lowres_elevation_file, bounds).astype(np.float32)
    base_lowres_dem[base_lowres_dem > -1] = -1
    base_lowres_dem = skimage.transform.resize(base_lowres_dem, (lowres_size, lowres_size), order=1, preserve_range=True)
    base_lowres_dem = skimage.filters.gaussian(base_lowres_dem, sigma=lowres_sigma)
    assert (~np.isnan(base_lowres_dem)).all()
    
    scaled_base_lowres_dem = skimage.transform.resize(base_lowres_dem, (highres_size, highres_size), order=1, preserve_range=True)
    if highres_margin > 0:
        scaled_base_lowres_dem = scaled_base_lowres_dem[highres_margin:-highres_margin, highres_margin:-highres_margin]
    
    # Replace NaN values over 10 pixels from non-NaN pixels with 0
    if np.isnan(highres_dem).all():
        highres_dem = scaled_base_lowres_dem
    elif np.isnan(highres_dem).any():
        # Create a mask of non-NaN pixels
        nan_mask = np.isnan(highres_dem)
        distance = scipy.ndimage.distance_transform_edt(nan_mask)
        alpha = np.minimum(1, distance[nan_mask]/32)
        highres_dem[nan_mask] = scaled_base_lowres_dem[nan_mask] * alpha
    assert (~np.isnan(highres_dem)).all()
        
    os.makedirs('debug', exist_ok=True)
    plt.imsave(os.path.join('debug', f'highres_dem_{chunk_id}.png'), highres_dem, cmap='gray')
    
    if landcover_folder is not None:
        try:
            landcover = read_raster(os.path.join(landcover_folder, file)).astype(np.int16)
            assert not np.isnan(landcover).all()
            landcover = skimage.transform.resize(landcover, (highres_size, highres_size), order=0, preserve_range=True)
            if highres_margin > 0:
                landcover = landcover[..., highres_margin:-highres_margin, highres_margin:-highres_margin]
        except Exception:
            landcover = None
    else:
        landcover = None
    
    if watercover_folder is not None:
        try:
            watercover = read_raster(os.path.join(watercover_folder, file)).astype(np.float32)
            assert not np.isnan(watercover).all()
            watercover = skimage.transform.resize(watercover, (highres_size, highres_size), order=0, preserve_range=True)
            if highres_margin > 0:
                watercover = watercover[..., highres_margin:-highres_margin, highres_margin:-highres_margin]
        except Exception:
            watercover = None
    else:
        watercover = None
        
    if koppen_geiger_file is not None:
        koppen_geiger = extract_mask_from_tiffs(koppen_geiger_file, bounds).astype(np.int8)
        assert not np.isnan(koppen_geiger).all()
        koppen_geiger = skimage.transform.resize(koppen_geiger, (lowres_size, lowres_size), order=0, preserve_range=True)
        if highres_margin > 0:
            koppen_geiger = koppen_geiger[..., highres_margin:-highres_margin, highres_margin:-highres_margin]
    else:
        koppen_geiger = None
        
    if climate_folder is not None:
        # Initialize list to store climate data layers
        climate_layers = []
        
        # Extract and process each climate variable
        for i in range(1, 20):
            fname = f'wc2.1_30s_bio_{i}.tif'
            climate_data = extract_mask_from_tiffs(os.path.join(climate_folder, fname), bounds).astype(np.float32)
            if np.isnan(climate_data).all():
                climate_data = np.nan * np.ones((lowres_size, lowres_size), dtype=np.float32)
            else:
                climate_data = skimage.transform.resize(climate_data, (lowres_size, lowres_size), order=0, preserve_range=True)
            climate_layers.append(climate_data)
        
        # Stack all layers along new axis
        climate = np.stack(climate_layers, axis=0)
    else:
        climate = None
        
    residual, lowres_dem = laplacian_encode(highres_dem, lowres_size, lowres_sigma)
        
    highres_chunk_size = (highres_size - highres_margin * 2) // num_chunks
    lowres_chunk_size = (lowres_size - edge_margin * 2) // num_chunks
    chunks_data = []
    
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
                'koppen_geiger': koppen_geiger[..., lowres_start_h:lowres_end_h, lowres_start_w:lowres_end_w] if koppen_geiger is not None else None,
                'climate': climate[..., lowres_start_h:lowres_end_h, lowres_start_w:lowres_end_w] if climate is not None else None,
                'pct_land': pct_land,
                'chunk_id': chunk_id,
                'subchunk_id': f'chunk_{chunk_h}_{chunk_w}'
            })
    
    return chunks_data


class ElevationDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 highres_elevation_folder, 
                 lowres_elevation_file, 
                 resolution,
                 highres_size, 
                 lowres_size,
                 lowres_sigma,
                 num_chunks=1,
                 landcover_folder=None,
                 watercover_folder=None,
                 koppen_geiger_folder=None,
                 climate_folder=None,
                 skip_chunk_ids=None,
                 edge_margin=0):
        self.grid_cells = create_equal_area_grid((highres_size*resolution, highres_size*resolution))
        self.chunk_ids = [str(i) for i in range(len(self.grid_cells))]
        
        self.highres_elevation_folder = highres_elevation_folder
        self.lowres_elevation_file = lowres_elevation_file
        self.highres_size = highres_size
        self.lowres_size = lowres_size
        self.lowres_sigma = lowres_sigma
        self.num_chunks = num_chunks
        self.landcover_folder = landcover_folder
        self.watercover_folder = watercover_folder
        self.koppen_geiger_folder = koppen_geiger_folder
        self.climate_folder = climate_folder
        if skip_chunk_ids is not None:
            skip_chunk_ids = set(str(x) for x in skip_chunk_ids)
            self.chunk_ids = [f for f in self.chunk_ids if str(f) not in skip_chunk_ids]
        self.edge_margin = edge_margin
        
    def __len__(self):
        return len(self.chunk_ids)
    
    def __getitem__(self, idx):
        cid = self.chunk_ids[idx]
        return process_single_file_base(cid, 
                                        self.grid_cells[int(cid)],
                                        self.highres_elevation_folder, 
                                        self.lowres_elevation_file, 
                                        self.highres_size, 
                                        self.lowres_size,
                                        self.lowres_sigma,
                                        self.num_chunks, 
                                        self.landcover_folder, 
                                        self.watercover_folder,
                                        self.koppen_geiger_folder,
                                        self.climate_folder,
                                        self.edge_margin)