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
terrain = create_unbounded_pipe(sigmas=[80, 5], cond_input_scaling=1) # Shape = (Infinite, Infinite), accessed as terrain[x1:x2, y1:y2]
print("Terrain loaded!")

def signed_sqrt_to_elevation(values):
    """Convert signed sqrt values back to elevation in meters"""
    return np.sign(values) * values * values

def elevation_to_color(elevation_data):
    """Map elevation (m) → RGB using absolute Earth elevation scale.
    
    • Uses Mt. Everest (8848m) and Mariana Trench (-11034m) as reference points
    • Gradient is applied to √|elevation| so changes near sea-level are clearer.
    • Water   : light-cyan (shallow) → dark-blue (deep).
    • Beach/shore (0-1m): sand color
    • Low land: lush green → olive.
    • Highlands/mountains: brown → white (snow).
    """
    h, w = elevation_data.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Earth's elevation extremes
    MAX_DEPTH = 11034  # Mariana Trench depth (positive value)
    MAX_ELEVATION = 8848  # Mt. Everest height

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
            t = np.sqrt(higher_elev) / np.sqrt(MAX_ELEVATION)   # normalize to sqrt of max elevation
            t = np.clip(t, 0, 1)                                # clamp to [0,1]

            bp    = np.array([0.0, 0.3, 0.6, 1.0])       # break-points
            cols  = np.array([
                [110, 220, 110],   # lush green lowlands
                [170, 190,  90],   # olive/grass
                [180, 140, 110],   # brown rock
                [255, 255, 255],   # snow
            ])

            land_rgb = np.zeros((t.shape[0], 3))
            for c in range(3):                          # interpolate per channel
                land_rgb[:, c] = np.interp(t, bp, cols[:, c])

            rgb_image[higher_mask] = land_rgb.astype(np.uint8)

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
        window_width = int(request.args.get('window_width', 1920))
        window_height = int(request.args.get('window_height', 1080))
        scale = float(request.args.get('scale', 1.0))
        
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
        print(f"Generating terrain for region: x={x}, y={y}, width={width}, height={height}, scale={scale}")
        terrain_data = terrain[y:y+height, x:x+width].cpu().numpy()
        
        # Convert signed sqrt values to actual elevation
        elevation_data = signed_sqrt_to_elevation(terrain_data)
        
        # Create colored image
        rgb_image = elevation_to_color(elevation_data)
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Also send elevation statistics
        min_elevation = float(np.min(elevation_data))
        max_elevation = float(np.max(elevation_data))
        mean_elevation = float(np.mean(elevation_data))
        
        return jsonify({
            'image': image_base64,
            'elevation_data': elevation_data.tolist(),
            'stats': {
                'min_elevation': min_elevation,
                'max_elevation': max_elevation,
                'mean_elevation': mean_elevation,
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
        terrain_data = terrain[min_y:max_y, min_x:max_x].cpu().numpy()
        
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
    

