import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from global_land_mask import globe

def create_equal_area_grid(target_size: Tuple[float, float]) -> List[Tuple[float, float, float, float]]:
    """
    Creates a grid of approximately equal-area squares between latitudes -60° to 60°.
    
    Args:
        target_size: Tuple of (width, height) in meters for desired grid cell size.
    
    Returns:
        List of tuples (min_lon, min_lat, max_lon, max_lat) defining each grid cell.
    """
    EARTH_RADIUS = 6378000  # Earth's radius in meters
    MIN_LAT, MAX_LAT = -60, 60
    
    # Calculate base degree spacing at equator
    base_lon_spacing = np.degrees(target_size[0] / (EARTH_RADIUS * np.cos(np.radians(0))))
    base_lat_spacing = np.degrees(target_size[1] / EARTH_RADIUS)
    
    grid_cells = []
    
    # Generate latitude bands, stopping before exceeding MAX_LAT
    current_lat = MIN_LAT
    while current_lat + base_lat_spacing < MAX_LAT:
        # Ensure we don't exceed MAX_LAT
        next_lat = current_lat + base_lat_spacing
            
        # Adjust longitude spacing based on latitude to maintain equal area
        cos_lat = np.cos(np.radians(current_lat + (next_lat - current_lat)/2))
        adjusted_lon_spacing = base_lon_spacing / cos_lat
        
        # Generate longitude divisions for this latitude band
        current_lon = -180
        while current_lon < 180:
            next_lon = min(current_lon + adjusted_lon_spacing, 180)
            if next_lon - current_lon < adjusted_lon_spacing * 0.5:  # Skip if resulting cell would be too small
                break
                
            cell = (current_lon, current_lat, next_lon, next_lat)
            grid_cells.append(cell)
            current_lon += adjusted_lon_spacing
            
        current_lat = next_lat
    
    return grid_cells

def plot_equal_area_grid():
    """
    Plots the equal-area grid over a map of Earth using an equal-area projection.
    Uses the Cylindrical Equal Area projection for accurate area representation.
    Cells are colored based on whether they contain water, land, or both.
    """
    # Create figure with equal-area projection
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    
    # Add coastlines and other features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    
    # Get grid cells
    grid_cells = create_equal_area_grid((4096*90*4, 4096*90*4))
    
    # Initialize counters
    water_only_cells = 0
    land_only_cells = 0
    border_cells = 0

    # Plot each grid cell
    for cell in grid_cells:
        min_lon, min_lat, max_lon, max_lat = cell
        
        # Sample points within the cell to determine if it contains water
        sample_lons = np.linspace(min_lon, max_lon, 10)
        sample_lats = np.linspace(min_lat, max_lat, 10)
        lon_grid, lat_grid = np.meshgrid(sample_lons, sample_lats)
        
        # Check land/water composition
        is_land = globe.is_land(lat_grid, lon_grid)
        has_water = not np.all(is_land)
        has_land = np.any(is_land)
        
        # Create box coordinates
        lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
        lats = [min_lat, min_lat, max_lat, max_lat, min_lat]
        
        # Determine cell type and color
        if has_water and has_land:
            color = 'red'
            border_cells += 1
        elif has_water:
            color = 'blue'
            water_only_cells += 1
        else:
            color = 'green'
            land_only_cells += 1
            
        ax.plot(lons, lats, color=color, alpha=0.5, linewidth=0.5,
                transform=ccrs.PlateCarree())

    print(f"Pure water cells: {water_only_cells}")
    print(f"Pure land cells: {land_only_cells}")
    print(f"Border cells (containing both): {border_cells}")
    print(f"Total cells: {water_only_cells + land_only_cells + border_cells}")

    # Set map bounds
    ax.set_global()
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    plt.title('Equal-Area Grid Cells (491520m x 491520m)')
    plt.show()

if __name__ == "__main__":
    plot_equal_area_grid()
