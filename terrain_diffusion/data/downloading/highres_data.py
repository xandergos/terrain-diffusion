import json
import math
import os
import random
import time

import ee
from matplotlib import pyplot as plt

import global_land_mask

from tqdm import trange

ee.Authenticate()
ee.Initialize(
    project='generative-land',
    opt_url='https://earthengine-highvolume.googleapis.com'
)

image_collection = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2")
elevation = image_collection.select('DSM').mosaic()

coords = []
dist = 1
for lat in range(-60, 60, dist):
    lon = 0
    spacing = max(1, math.floor(dist * 0.75 / math.cos(math.radians(lat))))
    while lon < 360:
        coords.append((lon, lat))
        lon += spacing

random.seed(32)
random.shuffle(coords)

# filter coords to only include land
coords = [coord for coord in coords if global_land_mask.is_land(coord[1], (coord[0] + 180) % 360 - 180)]
print(f"{len(coords)} coordinates to export.")

def export_data(image, img_name, lon, lat, size, scale):
    """
    Export Earth Engine image data for a given location.
    
    Args:
        image: Earth Engine image to export
        img_name: Name prefix for the exported file
        lon: Longitude of center point
        lat: Latitude of center point 
        size: Radius in meters around center point
        scale: Resolution in meters per pixel
    
    Returns:
        task: Earth Engine export task if started, None if already exported
    """
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(size)
    
    if os.path.exists("exported_tasks.json"):
        with open("exported_tasks.json", "r") as f:
            exported_tasks = json.load(f)
    else:
        exported_tasks = {}
        
    if f"{img_name}_{lon}_{lat}" in exported_tasks:
        return None
        
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=img_name,
        scale=scale,
        region=roi,
        folder=f'generative_land_data_{img_name}',
        fileNamePrefix=f'lon_{lon}-lat_{lat}',
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        skipEmptyTiles=True
    )
    
    task.start()
    exported_tasks[f"{img_name}_{lon}_{lat}"] = task.status()['state']
    
    with open("exported_tasks.json", "w") as f:
        json.dump(exported_tasks, f)
        
    return task

def export_batch(image, img_name, coords, output_size, resolution):
    """
    Export a batch of locations.
    
    Args:
        image: Earth Engine image to export
        img_name: Name prefix for exported files
        coords: List of (lon,lat) coordinates to export
        output_size: Desired output size in pixels
        resolution: Resolution in meters per pixel
    
    Returns:
        tasks: List of started export tasks
    """
    tasks = []
    radius = (output_size/2) * resolution # Convert pixels to meters
    
    for lon, lat in coords:
        task = export_data(image, img_name, lon, lat, radius, resolution)
        if task is not None:
            tasks.append(task)
    return tasks

def count_complete(tasks):
    """Count number of completed tasks."""
    n = 0
    for task in tasks:
        state = task.status()['state']
        if state in ['COMPLETED', 'FAILED']:
            n += 1
    return n

# Parameters
output_size = 2048  # Output image size in pixels
resolution = 30     # Resolution in meters per pixel
batch_size = 50     # Number of locations to process in parallel

# Process all coordinates in batches
for i in (pbar := trange(0, len(coords), batch_size)):
    subcoords = coords[i:i + batch_size]
    tasks = export_batch(elevation, 'high_res_elevation', subcoords, output_size, resolution)
    
    # Wait for batch to complete
    n = 0
    while n != len(tasks):
        pbar.set_postfix({'status': f"{n} / {len(tasks)}"})
        n = count_complete(tasks)
        time.sleep(5)
