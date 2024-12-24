import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from terrain_diffusion.data.downloading.world_grid import create_equal_area_grid

def extract_grid_data(tiff_path: str, grid_cells: List[Tuple[float, float, float, float]],
                      output_dir, output_prefix) -> List[Optional[np.ndarray]]:
    """
    Extracts data from a GeoTIFF file for each grid cell in the provided list,
    applying Gaussian blur preprocessing to the entire raster first.
    
    Args:
        tiff_path: Path to the GeoTIFF file
        grid_cells: List of tuples (min_lon, min_lat, max_lon, max_lat) defining each grid cell
        output_dir: Directory to save the processed grid cells
        output_prefix: Prefix for output filenames
    
    Returns:
        List of numpy arrays containing the processed data for each grid cell, or None if the cell
        contains no valid data
    """
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(tiff_path) as src:
        # Read the entire raster
        data = src.read(1)
        
        # Create a mask for nodata values
        nodata_mask = data == src.nodata
        
        # Apply gaussian blur to valid data only
        valid_data = np.ma.masked_where(nodata_mask, data)
        blurred_data = gaussian_filter(valid_data, sigma=10)
        
        # Restore nodata values
        blurred_data[nodata_mask] = src.nodata
        
        for idx, cell in tqdm(enumerate(grid_cells)):
            min_lon, min_lat, max_lon, max_lat = cell
            
            # Create a window for the grid cell
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
            
            # Extract the blurred data for this window
            cell_data = blurred_data[window.toslices()]
            
            # If the window is empty or contains only nodata values, skip
            if cell_data.size != 0 and np.any(cell_data != src.nodata):
                # Create output path
                output_path = f"{output_dir}/{output_prefix}_cell_{idx}.tif"
                
                # Get the transform for this window
                window_transform = rasterio.windows.transform(window, src.transform)
                
                # Save the data as a new GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=cell_data.shape[0],
                    width=cell_data.shape[1],
                    count=1,
                    dtype=cell_data.dtype,
                    crs=src.crs,
                    transform=window_transform,
                    nodata=src.nodata
                ) as dst:
                    dst.write(cell_data, 1)


# Example usage:
if __name__ == "__main__":
    # Create grid cells
    grid_cells = create_equal_area_grid((4096*90, 4096*90))
    
    # Extract data for each cell
    tiff_path = "/home/agos/Downloads/ETOPO_2022_v1_60s_N90W180_bed.tif"
    cell_data = extract_grid_data(tiff_path, grid_cells, output_dir="/mnt/ntfs2/shared/data/terrain/etopo", output_prefix="etopo")