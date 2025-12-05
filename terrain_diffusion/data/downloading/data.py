import time
import click
import ee
import os
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

from tqdm import tqdm
from terrain_diffusion.data.downloading.world_grid import create_equal_area_grid
import numpy as np
from global_land_mask import globe

def initialize_ee():
    """
    Initialize the Earth Engine API. Assumes you have already authenticated.
    """
    ee.Authenticate()
    ee.Initialize(
        project='generative-land',
        opt_url='https://earthengine-highvolume.googleapis.com'
    )

def download_cell_data(image, cell: Tuple[float, float, float, float], output_dir: str, cell_index: int, image_name: str = None, scale: float = None, max_retries: int = 3):
    """
    Download Earth Engine data for a specific grid cell directly to local filesystem.
    
    Args:
        image: Earth Engine image to download
        cell: Tuple of (min_lon, min_lat, max_lon, max_lat) defining the cell bounds
        output_dir: Directory to save the downloaded data
        cell_index: Index of the cell for unique naming
        image_name: Name prefix for the saved file
        scale: Optional scale in meters per pixel (required for mosaicked ImageCollections)
        max_retries: Number of retry attempts on failure
    """
    min_lon, min_lat, max_lon, max_lat = cell
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    # Save to local file
    filename = f"{image_name}_{cell_index}.tif" if image_name else f"cell_{cell_index}.tif"
    filepath = os.path.join(output_dir, filename)
    temp_filepath = filepath + ".tmp"
    
    if os.path.exists(filepath):
        return filepath
    
    for attempt in range(max_retries):
        try:
            # MERIT elevations fit safely in Int16; export native grid without resampling.
            # Do any resampling/reprojection offline after download.
            image_to_export = image
            
            # Get download URL
            params = {
                'region': region,
                'format': 'GEO_TIFF',
                'filePerBand': False,
                'maxPixels': 1e9,
                'formatOptions': {
                    'cloudOptimized': False,
                    'compression': 'LZW'
                }
            }
            if scale is not None:
                params['scale'] = scale
            url = image_to_export.getDownloadURL(params)
            
            # Download the file to temp location
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Move temp file to final location
            os.rename(temp_filepath, filepath)
                    
            return filepath
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Error downloading cell {cell_index} after {max_retries} attempts: {e}")
                return None

def calculate_land_percentage(cell: Tuple[float, float, float, float], resolution: float = 0.1) -> float:
    """
    Calculate the percentage of land in a cell using global_land_mask.
    
    Args:
        cell: Tuple of (min_lon, min_lat, max_lon, max_lat) defining the cell bounds
        resolution: Grid resolution in degrees for sampling points
    
    Returns:
        float: Percentage of land coverage (0-100)
    """
    min_lon, min_lat, max_lon, max_lat = cell
    
    # Create grid of points within the cell
    lats = np.arange(min_lat, max_lat, resolution)
    lons = np.arange(min_lon, max_lon, resolution)
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    
    # Get land mask for all points
    is_land = globe.is_land(lat_grid, lon_grid)
    
    # Calculate percentage
    return 100 * np.mean(is_land)

@click.command()
@click.option('--image', type=str, help='Image to export. Options: "dem", "copernicus", "landcover_class", "landcover_water"')
@click.option('--output_dir', type=str, default="terrain_data", help='Directory where the exported data will be saved')
@click.option('--output_size', type=int, default=4096, help='Output size of the image in pixels')
@click.option('--output_resolution', type=float, default=90, help='Output resolution of the image in meters')
@click.option('--land_threshold', type=float, default=0.1, help='Required land coverage percentage per export cell (Default 0.1%)')
@click.option('--num_workers', type=int, default=0, help='Number of parallel download workers (0 for sequential)')
def download_data_cli(image, output_dir, output_size, output_resolution, land_threshold, num_workers):
    """
    Download terrain data for all grid cells using Earth Engine directly to local filesystem.
    Only processes cells with land coverage above the specified threshold.
    
    Args:
        image: Type of image to download (dem, landcover_class, landcover_water)
        output_dir: Directory where the downloaded data will be saved
        output_size: Output size of the image in pixels
        output_resolution: Output resolution of the image in meters
        land_threshold: Required land coverage percentage per cell
        num_workers: Number of parallel download workers (0 for sequential)
    """
    # Initialize Earth Engine
    initialize_ee()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get grid cells
    grid_cells = create_equal_area_grid((output_size*output_resolution, output_size*output_resolution))
        
    # native_scale is only needed for mosaicked ImageCollections which lose projection info
    if image == "dem":
        export_image = ee.Image('MERIT/DEM/v1_0_3')
        image_name = "dem"
        native_scale = None
    elif image == "copernicus":
        export_image = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
        image_name = "copernicus"
        native_scale = 30.75  # GLO30 is 1 arc-second (~30.75m)
    elif image == "landcover_class":
        export_image = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global") \
            .select('discrete_classification') \
            .mosaic()
        image_name = "landcover_class"
        native_scale = 100
    elif image == "landcover_water":
        # Use JRC Global Surface Water for comprehensive water detection
        export_image = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        image_name = "landcover_water"
        native_scale = None
    else:
        raise ValueError(f"Invalid image option: {image}. Please choose 'dem', 'copernicus', 'landcover_class', or 'landcover_water'.")
        
    # Filter cells by land percentage first
    filtered_cells = []
    for i, cell in enumerate(grid_cells):
        land_percentage = calculate_land_percentage(cell)
        if land_percentage > land_threshold:
            filtered_cells.append((i, cell))
    
    print(f"Found {len(filtered_cells)} cells with >{land_threshold}% land coverage")
    
    # Ask for confirmation
    confirmation = input("Do you want to proceed with the download? (y/n): ")
    if confirmation.lower() != 'y':
        print("Download cancelled")
        return []
    
    # Randomize download order (seeded for reproducibility)
    random.seed(42)
    random.shuffle(filtered_cells)
        
    # Download files
    downloaded_files = []
    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(download_cell_data, export_image, cell, output_dir, i, image_name, native_scale): i
                for i, cell in filtered_cells
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading cells"):
                filepath = future.result()
                if filepath:
                    downloaded_files.append(filepath)
    else:
        for i, cell in tqdm(filtered_cells, desc="Downloading cells"):
            filepath = download_cell_data(export_image, cell, output_dir, i, image_name, scale=native_scale)
            if filepath:
                downloaded_files.append(filepath)
        
    print(f"Downloaded {len(downloaded_files)} files to {output_dir}")
    return downloaded_files
    
if __name__ == "__main__":
    download_data_cli()