import click
import ee
import os
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

def export_cell_data(image, cell: Tuple[float, float, float, float], output_dir: str, cell_index: int, output_size: int = 1024, image_name = None):
    """
    Export Earth Engine data for a specific grid cell with exact pixel dimensions.
    
    Args:
        cell: Tuple of (min_lon, min_lat, max_lon, max_lat) defining the cell bounds
        output_dir: Directory to save the exported data
        cell_index: Index of the cell for unique naming
        output_size: Desired width and height of the output image in pixels
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
    
    # Configure export parameters
    export_params = {
        'image': image,
        'description': f'{cell_index}',
        'folder': os.path.basename(output_dir),
        'region': region,
        'crs': 'EPSG:4326',
        'crsTransform': crs_transform,
        'maxPixels': 1e9
    }
    
    # Start the export task
    task = ee.batch.Export.image.toDrive(**export_params)
    task.start()
    return task

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
    Download terrain data for all grid cells using Earth Engine.
    Only processes cells with >50% land coverage.
    
    Args:
        output_dir: Directory where the exported data will be saved
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
        export_image = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global") \
            .select('water-permanent-coverfraction') \
            .mosaic()
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
    confirmation = input("Do you want to proceed with the export? (y/n): ")
    if confirmation.lower() != 'y':
        print("Export cancelled")
        return []
        
    # Create export tasks
    tasks = []
    for i, cell in tqdm(filtered_cells, desc="Exporting cells"):
        task = export_cell_data(export_image, cell, output_dir, i, output_size, image_name)
        tasks.append(task)
        
    print(f"Started {len(tasks)} export tasks. Check your Google Drive for the results.")
    return tasks
    
if __name__ == "__main__":
    download_data_cli()