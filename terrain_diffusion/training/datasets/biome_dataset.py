import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from tqdm import tqdm

class BiomeDataset(Dataset):
    def __init__(self,
                 temp_file,
                 temp_std_file,
                 precip_file,
                 precip_std_file,
                 koppen_file,
                 *,
                 length: int,
                 seed: int = 0,
                 lat_min: float = -60.0,
                 lat_max: float = 60.0,
                 input_dropout: float = 0.0):
        self.input_dropout = input_dropout
        
        print("Loading temperature data...")
        with rasterio.open(temp_file) as src:
            bounds = src.bounds
            height = src.height
            lat_res = (bounds.top - bounds.bottom) / height
            start_row = int((bounds.top - lat_max) / lat_res)
            end_row = int((bounds.top - lat_min) / lat_res)
            start_row = max(0, min(start_row, height))
            end_row = max(0, min(end_row, height))
            if start_row >= end_row:
                start_row, end_row = 0, height
            temp_img = src.read(1)[start_row:end_row, :]
            temp_img[temp_img < -30000] = np.nan
        print("Loading temperature standard deviation data...")
        with rasterio.open(temp_std_file) as src:
            temp_std_img = src.read(1)[start_row:end_row, :]
            temp_std_img[temp_std_img < -30000] = np.nan
        print("Loading precipitation data...")
        with rasterio.open(precip_file) as src:
            precip_img = src.read(1)[start_row:end_row, :]
            precip_img[precip_img < -30000] = np.nan
        print("Loading precipitation standard deviation data...")
        with rasterio.open(precip_std_file) as src:
            precip_std_img = src.read(1)[start_row:end_row, :]
            precip_std_img[precip_std_img < -30000] = np.nan
        print("Loading Koppen-Geiger data...")
        with rasterio.open(koppen_file) as src:
            koppen_img = src.read(1)[start_row:end_row, :]

        # Preallocate and fill with while-loop random sampling (no precomputed mask)
        n_rows, n_cols = temp_img.shape
        rng = np.random.default_rng(seed)
        x = np.empty((int(length), 4), dtype=np.float32)
        y = np.empty((int(length),), dtype=np.int64)

        i = 0
        with tqdm(total=int(length), desc="Sampling dataset") as pbar:
            while i < int(length):
                r = int(rng.integers(0, n_rows))
                c = int(rng.integers(0, n_cols))
                t = temp_img[r, c]
                ts = temp_std_img[r, c]
                p = precip_img[r, c]
                ps = precip_std_img[r, c]
                k = koppen_img[r, c]
                if np.isnan(t) or np.isnan(ts) or np.isnan(p) or np.isnan(ps) or np.isnan(k):
                    continue
                x[i, 0] = t
                x[i, 1] = ts
                x[i, 2] = p
                x[i, 3] = ps
                y[i] = int(k)
                i += 1
                pbar.update(1)

        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        print(f"x_mean: {self.x_mean}, x_std: {self.x_std}")
        self.x = x
        self.y = y
        self.length = int(length)
        self.training = True

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).to(torch.float32) - self.x_mean
        x = x / self.x_std
        if self.input_dropout > 0 and self.training:
            mask = (torch.rand(x.shape) > self.input_dropout).float()
            x = x * mask
        else:
            mask = torch.ones_like(x)
        x = torch.cat([x, mask], dim=0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return {'x': x, 'y': y}
    
    def train(self, mode: bool = True):
        self.training = mode
        return self

if __name__ == "__main__":
    dataset = BiomeDataset(
        temp_file='data/global/wc2.1_30s_bio_1.tif',
        temp_std_file='data/global/wc2.1_30s_bio_4.tif',
        precip_file='data/global/wc2.1_30s_bio_12.tif',
        precip_std_file='data/global/wc2.1_30s_bio_15.tif',
        koppen_file='data/global/koppen_geiger_0p00833333.tif',
        length=1_000_000,
        seed=0,
        input_dropout=0.2,
    )
    print(dataset[0])