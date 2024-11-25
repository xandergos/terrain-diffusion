
import os
import random
from PIL import Image
import numpy as np
import json
from tqdm import tqdm


folder = '/mnt/ntfs2/shared/data/terrain/datasets/generative_land_data_composite_elv/'
paths = [os.path.join(folder, p) for p in os.listdir(folder)]
random.seed(385)
random.shuffle(paths)
train_paths = paths

if os.path.exists('paths_land.json'):
    with open('paths_land.json') as f:
        paths_land = json.load(f)
else:
    paths_land = []
    for path in tqdm(paths):
        im = np.array(Image.open(path)).astype(np.float32)
        if np.count_nonzero(im > 0) > 0.1 * 1024 ** 2:
            paths_land.append(os.path.basename(path))
    with open('paths_land.json', 'w') as f:
        json.dump(paths_land, f)
        
paths_ocean = []
for path in tqdm(paths):
    if os.path.basename(path) not in paths_land:
        paths_ocean.append(os.path.basename(path))

with open('paths_ocean.json', 'w') as f:
    json.dump(paths_ocean, f)

print(f"Number of land paths: {len(paths_land)}")
print(f"Number of ocean paths: {len(paths_ocean)}")
print(f"Total paths: {len(paths)}")
