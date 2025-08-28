import time
import click
import ee
import os
import requests
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

def download_cell_data(image, cell: Tuple[float, float, float, float], output_dir: str, cell_index: int, output_size: int = 1024, image_name: str = None):
    """
    Download Earth Engine data for a specific grid cell directly to local filesystem.
    
    Args:
        image: Earth Engine image to download
        cell: Tuple of (min_lon, min_lat, max_lon, max_lat) defining the cell bounds
        output_dir: Directory to save the downloaded data
        cell_index: Index of the cell for unique naming
        output_size: Desired width and height of the output image in pixels
        image_name: Name prefix for the saved file
    """
    min_lon, min_lat, max_lon, max_lat = cell
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    
    # Calculate scales to get exactly output_size pixels
    x_scale = (max_lon - min_lon) / output_size  # degrees per pixel
    y_scale = (max_lat - min_lat) / output_size  # degrees per pixel
    
    crs_transform = [
        x_scale,  # xScale: size of pixel in degrees
        0,        # xShear
        min_lon,  # xTranslation: longitude of the top-left corner
        0,        # yShear
        -y_scale, # yScale: negative because latitude decreases as we go down
        max_lat   # yTranslation: latitude of the top-left corner
    ]
    
    try:
        # Save to local file
        filename = f"{image_name}_{cell_index}.tif" if image_name else f"cell_{cell_index}.tif"
        filepath = os.path.join(output_dir, filename)
        temp_filepath = filepath + ".tmp"
        
        if os.path.exists(filepath):
            print(f"File {filepath} already exists")
            return filepath
        
        # Get download URL
        url = image.getDownloadURL({
            'region': region,
            'crs': 'EPSG:4326',
            'crs_transform': crs_transform,  # snake_case here
            'format': 'GEO_TIFF',
            'filePerBand': False,
            'maxPixels': 1e9
        })
        
        # Download the file to temp location
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        try:
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Error downloading cell {cell_index}. Retrying in 60 seconds: {e}")
            time.sleep(60)
            raise
        
        # Move temp file to final location
        os.rename(temp_filepath, filepath)
                
        return filepath
        
    except Exception as e:
        print(f"Error downloading cell {cell_index}: {e}")
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
@click.option('--image', type=str, help='Image to export. Options: "dem" or "landcover" or "gtopo')
@click.option('--output_dir', type=str, default="terrain_data", help='Directory where the exported data will be saved')
@click.option('--output_size', type=int, default=4096, help='Output size of the image in pixels')
@click.option('--output_resolution', type=int, default=90, help='Output resolution of the image in meters')
@click.option('--land_threshold', type=float, default=0.1, help='Required land coverage percentage per export cell (Default 0.1%)')
def download_data_cli(image, output_dir, output_size, output_resolution, land_threshold):
    """
    Download terrain data for all grid cells using Earth Engine directly to local filesystem.
    Only processes cells with land coverage above the specified threshold.
    
    Args:
        image: Type of image to download (dem, landcover_class, landcover_water)
        output_dir: Directory where the downloaded data will be saved
        output_size: Output size of the image in pixels
        output_resolution: Output resolution of the image in meters
        land_threshold: Required land coverage percentage per cell
    """
    # Initialize Earth Engine
    initialize_ee()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get grid cells
    grid_cells = create_equal_area_grid((output_size*output_resolution, output_size*output_resolution))
        
    if image == "dem":
        export_image = ee.Image('MERIT/DEM/v1_0_3')
        image_name = "dem"
    elif image == "landcover_class":
        export_image = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global") \
            .select('discrete_classification') \
            .mosaic()
        image_name = "landcover_class"
    elif image == "landcover_water":
        # Use JRC Global Surface Water for comprehensive water detection
        export_image = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        #export_image = jrc_water.gte(1).uint8()
        image_name = "landcover_water"
    else:
        raise ValueError(f"Invalid image option: {image}. Please choose 'dem' or 'landcover'.")
        
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
        
    # Download files directly
    downloaded_files = []
    for i, cell in tqdm(filtered_cells, desc="Downloading cells"):
        filepath = download_cell_data(export_image, cell, output_dir, i, output_size, image_name)
        if filepath:
            downloaded_files.append(filepath)
        
    print(f"Downloaded {len(downloaded_files)} files to {output_dir}")
    return downloaded_files
    
if __name__ == "__main__":
    download_data_cli()