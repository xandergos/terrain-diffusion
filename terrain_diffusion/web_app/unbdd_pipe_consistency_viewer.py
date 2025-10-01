from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import base64
import io
from PIL import Image
import sys
import os
import torch
import rasterio
from rasterio.transform import from_bounds
import tempfile
from terrain_diffusion.web_app.unbdd_pipe_consistency import create_unbounded_pipe

app = Flask(__name__)

# Initialize the infinite terrain (this takes time on first load)
print("Loading infinite terrain generator...")
terrain = create_unbounded_pipe(sigmas=[80, 5], cond_input_scaling=1/16) # Shape = (Infinite, Infinite), accessed as terrain[x1:x2, y1:y2]
print("Terrain loaded!")

def signed_sqrt_to_elevation(values):
    """Convert signed sqrt values back to elevation in meters"""
    return np.sign(values) * values * values

def temperature_to_color(temperature_data, elevation_data, water_data=None):
    """Map temperature data as overlay over terrain.
    
    Temperature is already in Celsius.
    Creates a semi-transparent overlay with cold (blue) to hot (red) gradient
    """
    h, w = temperature_data.shape
    
    # First, get the base terrain image
    terrain_image = elevation_to_color(elevation_data, water_data)
    
    # Temperature is already in Celsius
    temp_celsius = temperature_data
    
    # Get the actual temperature range from the data for adaptive scaling
    temp_min = np.min(temp_celsius)
    temp_max = np.max(temp_celsius)
    
    # Expand range slightly for better visualization
    temp_range = temp_max - temp_min
    if temp_range > 0:
        temp_norm = (temp_celsius - temp_min) / temp_range
    else:
        temp_norm = np.zeros_like(temp_celsius)
    
    temp_norm = np.clip(temp_norm, 0, 1)
    
    # Create temperature overlay colors
    overlay_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create color gradient: blue (cold) to red (hot)
    # Blue for cold temperatures
    blue_component = (1 - temp_norm) * 255
    # Red for hot temperatures  
    red_component = temp_norm * 255
    # Green component peaks in the middle
    green_component = (1 - np.abs(temp_norm - 0.5) * 2) * 255
    
    overlay_image[:, :, 0] = red_component.astype(np.uint8)
    overlay_image[:, :, 1] = green_component.astype(np.uint8)
    overlay_image[:, :, 2] = blue_component.astype(np.uint8)
    
    # Blend the overlay with the terrain (60% terrain, 40% temperature overlay)
    alpha = 0.4  # Overlay strength
    blended_image = (
        terrain_image.astype(np.float32) * (1 - alpha) + 
        overlay_image.astype(np.float32) * alpha
    ).astype(np.uint8)
    
    return blended_image

def precipitation_to_color(precipitation_data, elevation_data, water_data=None):
    """Map precipitation data as overlay over terrain.
    
    Precipitation is in mm/year.
    Creates a semi-transparent overlay with dry (brown/yellow) to wet (blue/green) gradient
    """
    h, w = precipitation_data.shape
    
    # First, get the base terrain image
    terrain_image = elevation_to_color(elevation_data, water_data)
    
    # Precipitation is already in mm/year, ensure non-negative
    precip_mm_year = np.clip(precipitation_data, 0, None)
    
    # Normalize using the actual data range for adaptive scaling
    precip_max = np.max(precip_mm_year)
    if precip_max > 0:
        precip_norm = precip_mm_year / precip_max
    else:
        precip_norm = np.zeros_like(precip_mm_year)
    
    # Create precipitation overlay colors
    overlay_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create color gradient: brown/yellow (dry) to blue/green (wet)
    # Dry areas: brown/yellow
    dry_color = np.array([139, 115, 85])  # Brown
    # Wet areas: blue/green
    wet_color = np.array([65, 105, 225])  # Royal blue
    
    # Interpolate between dry and wet colors
    for i in range(3):
        overlay_image[:, :, i] = (
            dry_color[i] * (1 - precip_norm) + wet_color[i] * precip_norm
        ).astype(np.uint8)
    
    # Blend the overlay with the terrain (65% terrain, 35% precipitation overlay)
    alpha = 0.35  # Overlay strength
    blended_image = (
        terrain_image.astype(np.float32) * (1 - alpha) + 
        overlay_image.astype(np.float32) * alpha
    ).astype(np.uint8)
    
    return blended_image

def variation_to_color(variation_data, elevation_data, water_data=None):
    """Map variation data (standard deviation) as overlay over terrain.
    
    Variation represents standard deviation.
    Creates a semi-transparent overlay with low (transparent) to high (bright) variation gradient
    """
    h, w = variation_data.shape
    
    # First, get the base terrain image
    terrain_image = elevation_to_color(elevation_data, water_data)
    
    # Standard deviation is already in appropriate units, ensure non-negative
    std_dev = np.clip(variation_data, 0, None)
    
    # Normalize using the actual data range for adaptive scaling
    std_max = np.max(std_dev)
    if std_max > 0:
        var_norm = std_dev / std_max
    else:
        var_norm = np.zeros_like(std_dev)
    
    # Create variation overlay colors
    overlay_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create color gradient: transparent (low variation) to bright (high variation)
    # Low variation: use terrain color (will blend to show terrain)
    # High variation: bright purple/magenta
    high_var_color = np.array([255, 100, 255])  # Bright magenta
    
    # For variation overlay, we use variable alpha based on variation intensity
    # Low variation areas will show more terrain, high variation areas will show more overlay
    for i in range(3):
        overlay_image[:, :, i] = high_var_color[i]
    
    # Variable alpha blending: low variation = more terrain, high variation = more overlay
    alpha = var_norm * 0.5  # Maximum 50% overlay strength
    
    # Blend the overlay with the terrain using variable alpha
    blended_image = np.zeros_like(terrain_image)
    for i in range(3):
        blended_image[:, :, i] = (
            terrain_image[:, :, i].astype(np.float32) * (1 - alpha) + 
            overlay_image[:, :, i].astype(np.float32) * alpha
        ).astype(np.uint8)
    
    return blended_image

def elevation_to_color(elevation_data, water_data=None):
    """Map elevation (m) → RGB using adaptive elevation scale.
    
    • Uses maximum elevation in current window as reference
    • Water: light-cyan (shallow) → dark-blue (deep)
    • Beach/shore (0-1m): sand color
    • Land (>1m): black → white gradient, scaled to local max elevation
    • If water_data provided: overlays light blue based on water probability
    """
    h, w = elevation_data.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Earth's elevation extremes for water only
    MAX_DEPTH = 11034  # Mariana Trench depth (positive value)

    # --- Water ----------------------------------------------------------------
    water_mask = elevation_data < 0
    if np.any(water_mask):
        depth = -elevation_data[water_mask]          # positive metres
        t = np.sqrt(depth) / np.sqrt(MAX_DEPTH)      # normalize to sqrt of max depth
        t = np.clip(t, 0, 1)                         # clamp to [0,1]

        shallow = np.array([210, 230, 255])          # light cyan
        deep    = np.array([  0,  40, 130])          # dark blue
        rgb_image[water_mask] = (
            shallow * (1 - t[:, None]) + deep * t[:, None]
        ).astype(np.uint8)

    # --- Land ------------------------------------------------------------------
    land_mask = elevation_data >= 0
    if np.any(land_mask):
        # Beach/shore (0-1m) - sand color
        beach_mask = (elevation_data >= 0) & (elevation_data <= 1.0)
        if np.any(beach_mask):
            rgb_image[beach_mask] = [194, 178, 128]  # sand color
        
        # Higher elevations (>1m)
        higher_mask = elevation_data > 1.0
        if np.any(higher_mask):
            higher_elev = elevation_data[higher_mask]
            # Use local maximum elevation for better contrast
            local_max = np.max(higher_elev)
            t = np.sqrt(higher_elev) / np.sqrt(local_max)   # normalize to sqrt of local max
            t = np.clip(t, 0, 1)                            # clamp to [0,1]
            
            # Create grayscale values (black to white)
            gray_value = (t * 255).astype(np.uint8)
            rgb_image[higher_mask] = np.column_stack([gray_value, gray_value, gray_value])

    # --- Water Data Overlay ---------------------------------------------------
    if water_data is not None:
        # Light blue overlay color
        water_overlay = np.array([180, 220, 255])  # Light blue
        
        # For each pixel, blend the overlay based on water probability
        for i in range(3):  # RGB channels
            rgb_image[..., i] = (
                rgb_image[..., i] * (1 - water_data) + 
                water_overlay[i] * water_data
            ).astype(np.uint8)

    return rgb_image

@app.route('/')
def index():
    return render_template('terrain_viewer.html')

@app.route('/api/terrain')
def get_terrain():
    """Get terrain data for a specific region"""
    try:
        # Get parameters from request
        x = int(request.args.get('x', 0))
        y = int(request.args.get('y', 0))
        window_width = round(float(request.args.get('window_width', 1920)))
        window_height = round(float(request.args.get('window_height', 1080)))
        scale = float(request.args.get('scale', 1.0))
        channel = request.args.get('channel', 'terrain')
        
        # Calculate terrain dimensions based on window aspect ratio
        # Long side should be 2048 pixels, scaled by the scale factor
        base_size = int(2048 * scale)
        aspect_ratio = window_width / window_height
        
        if window_width >= window_height:
            # Landscape orientation
            width = base_size
            height = int(base_size / aspect_ratio)
        else:
            # Portrait orientation
            height = base_size
            width = int(base_size * aspect_ratio)
        
        # Clamp dimensions to reasonable limits
        width = min(max(width, 64), 10240)
        height = min(max(height, 64), 10240)
        
        # Extract terrain data - the terrain is accessed as terrain[y1:y2, x1:x2]
        print(f"Generating terrain for region: x={x}, y={y}, width={width}, height={height}, scale={scale}, channel={channel}")
        terrain_data = terrain[:, y:y+height, x:x+width].cpu().numpy()
        
        # Process data based on selected channel
        if channel == 'terrain':
            # Convert signed sqrt values to actual elevation
            elevation_data = signed_sqrt_to_elevation(terrain_data[0])
            water_data = terrain_data[1] if terrain_data.shape[0] > 1 else None
            rgb_image = elevation_to_color(elevation_data, water_data)
            data_for_stats = elevation_data
            stats_label = 'elevation'
            stats_unit = 'm'
        elif channel == 'temperature':
            # Mean temperature (channel 2) - already in Celsius
            temperature_data = terrain_data[2]
            elevation_data = signed_sqrt_to_elevation(terrain_data[0])
            water_data = terrain_data[1] if terrain_data.shape[0] > 1 else None
            rgb_image = temperature_to_color(temperature_data, elevation_data, water_data)
            data_for_stats = temperature_data  # Already in Celsius
            stats_label = 'temperature'
            stats_unit = '°C'
        elif channel == 'temperature_var':
            # Temperature variation (channel 3) - standard deviation in °C (multiplied by 100)
            temp_var_data = terrain_data[3] / 100.0  # Convert back to actual °C
            elevation_data = signed_sqrt_to_elevation(terrain_data[0])
            water_data = terrain_data[1] if terrain_data.shape[0] > 1 else None
            rgb_image = variation_to_color(temp_var_data, elevation_data, water_data)
            data_for_stats = temp_var_data  # Now in actual °C
            stats_label = 'temperature_variation'
            stats_unit = '°C'
        elif channel == 'precipitation':
            # Mean precipitation (channel 4) - already in mm/year
            precipitation_data = terrain_data[4]
            elevation_data = signed_sqrt_to_elevation(terrain_data[0])
            water_data = terrain_data[1] if terrain_data.shape[0] > 1 else None
            rgb_image = precipitation_to_color(precipitation_data, elevation_data, water_data)
            data_for_stats = precipitation_data
            stats_label = 'precipitation'
            stats_unit = 'mm/year'
        elif channel == 'precipitation_var':
            # Precipitation variation (channel 5) - standard deviation in mm/year
            precip_var_data = terrain_data[5]
            elevation_data = signed_sqrt_to_elevation(terrain_data[0])
            water_data = terrain_data[1] if terrain_data.shape[0] > 1 else None
            rgb_image = variation_to_color(precip_var_data, elevation_data, water_data)
            data_for_stats = precip_var_data
            stats_label = 'precipitation_variation'
            stats_unit = 'mm/year'
        else:
            # Default to terrain
            elevation_data = signed_sqrt_to_elevation(terrain_data[0])
            water_data = terrain_data[1] if terrain_data.shape[0] > 1 else None
            rgb_image = elevation_to_color(elevation_data, water_data)
            data_for_stats = elevation_data
            stats_label = 'elevation'
            stats_unit = 'm'
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Calculate statistics for the selected data
        min_value = float(np.min(data_for_stats))
        max_value = float(np.max(data_for_stats))
        mean_value = float(np.mean(data_for_stats))
        
        return jsonify({
            'image': image_base64,
            'data': data_for_stats.tolist(),
            'stats': {
                'min_value': min_value,
                'max_value': max_value,
                'mean_value': mean_value,
                'stats_label': stats_label,
                'stats_unit': stats_unit,
                'channel': channel,
                'width': width,
                'height': height,
                'x': x,
                'y': y,
                'scale': scale,
                'aspect_ratio': aspect_ratio
            }
        })
        
    except Exception as e:
        print(f"Error generating terrain: {e}")
        raise e

@app.route('/api/export')
def export_terrain():
    """Export terrain elevation data as a TIFF file"""
    try:
        # Get parameters from request
        x1 = int(request.args.get('x1', 0))
        y1 = int(request.args.get('y1', 0))
        x2 = int(request.args.get('x2', 1024))
        y2 = int(request.args.get('y2', 1024))
        
        # Ensure coordinates are in the right order (top-left to bottom-right)
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Limit export size to prevent memory issues
        max_size = 16384
        if width > max_size or height > max_size:
            return jsonify({'error': f'Export size too large. Maximum size is {max_size}x{max_size} pixels.'}), 400
        
        print(f"Exporting terrain region: x1={min_x}, y1={min_y}, x2={max_x}, y2={max_y} (size: {width}x{height})")
        
        # Extract terrain data - the terrain is accessed as terrain[y1:y2, x1:x2]
        terrain_data = terrain[0, min_y:max_y, min_x:max_x].cpu().numpy()
        
        # Convert signed sqrt values to actual elevation
        elevation_data = signed_sqrt_to_elevation(terrain_data)
        
        # Create a temporary file for the TIFF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
        temp_file.close()
        
        # Define the spatial extent (assuming 1 meter per pixel)
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        
        # Write the TIFF file
        with rasterio.open(
            temp_file.name,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=elevation_data.dtype,
            crs='EPSG:4326',  # WGS84 coordinate system
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(elevation_data, 1)
            
            # Add metadata
            dst.update_tags(
                AREA_OR_POINT='Point',
                DESCRIPTION='Terrain elevation data exported from Infinite Terrain Explorer',
                UNITS='meters',
                MIN_ELEVATION=str(np.min(elevation_data)),
                MAX_ELEVATION=str(np.max(elevation_data)),
                MEAN_ELEVATION=str(np.mean(elevation_data))
            )
        
        # Generate filename
        filename = f"terrain_{min_x}_{min_y}_to_{max_x}_{max_y}.tif"
        
        def cleanup():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=filename,
            mimetype='image/tiff'
        )
        
    except Exception as e:
        print(f"Error exporting terrain: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    

