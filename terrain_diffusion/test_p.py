import random
import h5py
import numpy as np
from tqdm import trange

from terrain_diffusion.data.laplacian_encoder import laplacian_decode

h5_file = 'data/dataset.h5'

est_p5 = []

keys = set()
res = 90
with h5py.File(h5_file, 'r') as f:
    if str(res) not in f:
        raise ValueError(f"Resolution {res} not found in {h5_file}")
        
    res_group = f[str(res)]
    for chunk_id in res_group:
        chunk_group = res_group[chunk_id]
        for subchunk_id in chunk_group:
            subchunk_group = chunk_group[subchunk_id]
            if 'latent' not in subchunk_group:
                continue
                
            dset = subchunk_group['latent']
            pct_land_valid = 0.5 <= dset.attrs['pct_land']
            
            if pct_land_valid:
                keys.add((chunk_id, res, subchunk_id))
                
keys = list(keys)

        
import matplotlib.pyplot as plt
for i in trange(1000):
    key = random.choice(keys)
    chunk_id, res, subchunk_id = key
    with h5py.File(h5_file, 'r') as f:
        group_path = f"{res}/{chunk_id}/{subchunk_id}"
        data_residual = f[f"{group_path}/residual"]
        data_lowfreq = f[f"{group_path}/lowfreq"]
        
        i = random.randint(0, data_lowfreq.shape[0] - 64)
        j = random.randint(0, data_lowfreq.shape[1] - 64)
        
        data_lowfreq = data_lowfreq[i:i+64, j:j+64]
        data_residual = data_residual[i*8:i*8+512, j*8:j*8+512]
        blocked = np.percentile(blocked, 50, axis=(1, 3))
        
        blocked = full_terrain.reshape(64, 8, 64, 8)
        p = np.percentile(blocked, 5)
        est = np.mean(full_terrain < p) * 100
        est_p5.append(est)
        if est > 6.5:
            print(est)
            plt.imshow(full_terrain)
            plt.show()
            plt.imshow(full_terrain < p)
            plt.show()
            pass
        
plt.figure(figsize=(10, 6))
plt.hist(est_p5, bins=50, edgecolor='black')
plt.xlabel('Estimated P5')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated P5 Values')
plt.grid(True, alpha=0.3)
plt.show()
        