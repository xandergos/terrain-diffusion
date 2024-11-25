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

# land cover
lc = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019').select('discrete_classification');

# climate
clim = ee.ImageCollection("WORLDCLIM/V1/MONTHLY").mean()

# Bathymetry for lakes
lakes = ee.Image("projects/sat-io/open-datasets/GLOBathy/GLOBathy_bathymetry").rename(['lake_depth'])

# Import the USGS ground elevation image.
elv = ee.Image("USGS/SRTMGL1_003").unmask(ee.Image.constant(-1000))
backup_elv = ee.Image("NOAA/NGDC/ETOPO1").rename(['bedrock', 'elevation']).select(['elevation'])
composite_elv = elv.gt(0).multiply(elv).add(elv.lte(0).multiply(backup_elv))

both_elevation = ee.Image.cat([elv, backup_elv]).rename(['base_elevation', 'backup_elevation'])

if os.path.exists("exported_tasks.json"):
    with open("exported_tasks.json", "r") as f:
        exported_tasks = json.load(f)
else:
    exported_tasks = {}

scale = 30
def export_data(image, img_name, lon, lat, size):
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(size)
    if f"{img_name}_{lon}_{lat}" in exported_tasks:
        return None
    task = ee.batch.Export.image.toDrive(image=image,
                                         description=img_name,
                                         scale=scale,
                                         region=roi,
                                         folder=f'generative_land_data_{img_name}',
                                         fileNamePrefix=f'lon_{lon}-lat_{lat}',
                                         crs='EPSG:4326',
                                         fileFormat='GeoTIFF',
                                         skipEmptyTiles=True)
    task.start()
    exported_tasks[f"{img_name}_{lon}_{lat}"] = task.status()['state']
    return task


coords = []
dist = 4
for lat in range(-60, 60, dist):
    lon = 0
    spacing = math.floor(dist * 0.75 / math.cos(math.radians(lat)))
    while lon < 360:
        coords.append((lon, lat))
        lon += spacing

random.seed(32)
random.shuffle(coords)

# filter coords to only include land
coords = [coord for coord in coords if global_land_mask.is_land(coord[1], (coord[0] + 180) % 360 - 180)]

print(f"{len(coords)} samples to export.")

def export_batch(image, img_name, coords):
    tasks = []
    for lon, lat in coords:
        task = export_data(image, img_name, lon, lat, scale * 2048)
        if task is not None:
            tasks.append(task)
    return tasks


def count_complete(tasks):
    n = 0
    for task in tasks:
        state = task.status()['state']
        if state == 'COMPLETED':
            n += 1
        elif state == 'FAILED':
            n += 1
    return n


bs = 50
for i in (pbar := trange(0, len(coords), bs)):
    subcoords = coords[i:i + bs]
    tasks = export_batch(elv, 'high_res_elevation', subcoords)
    
    with open('download-logs.txt', 'w') as f:
        f.write(f"Waiting for batch {i} to complete...")
    
    with open("exported_tasks.json", "w") as f:
        json.dump(exported_tasks, f)
        
    n = 0
    while n != len(tasks):
        pbar.set_postfix({'status': f"{n} / {len(tasks)}"})
        n = count_complete(tasks)
        time.sleep(5)
