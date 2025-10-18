import os
import random
import warnings
import numpy as np
from scipy.ndimage import zoom
from skimage.measure import block_reduce
import torch
from torch.utils.data import Dataset
import rasterio
import h5py
from tqdm import tqdm
import numpy as np
 

def harmonic_relaxation_inpaint_wrapx(out: np.ndarray,
                                      donors: np.ndarray,
                                      donor_values: np.ndarray,
                                      iters: int = 256) -> np.ndarray:
    u = out.copy()
    for _ in range(iters):
        left = np.roll(u, 1, axis=1)
        right = np.roll(u, -1, axis=1)
        up = np.vstack((u[:1, :], u[:-1, :]))
        down = np.vstack((u[1:, :], u[-1:, :]))
        u = 0.25 * (left + right + up + down)
        u[donors] = donor_values[donors]
    return u

def fill_oceans_wrap(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    H, W = a.shape

    donors = ~np.isnan(a)
    a_src = a

    if donors.sum() == 0:
        return a.copy()

    # Wrap 3x horizontally
    a3 = np.concatenate([a_src, a_src, a_src], axis=1)

    # Fill NaNs with 0 for initial seed
    a3_seed = np.nan_to_num(a3, nan=0.0)
    mask_valid = ~np.isnan(a3)
    
    # Apply harmonic relaxation directly at full resolution
    if mask_valid.any() and (~mask_valid).any():
        out3 = harmonic_relaxation_inpaint_wrapx(a3_seed, mask_valid, a3, iters=256)
    else:
        out3 = a3.copy()

    # Extract center portion
    out = out3[:, W:2*W].copy()
    out[donors] = a[donors]
        
    return out

class GANDataset(Dataset):
    """
    A PyTorch dataset class that returns random crops from climate/elevation rasters.
    
    Args:
        dataset_names (list): List of dataset names in desired stacking order
        crop_size (tuple): Size of the crop (height, width)
        resize_size (tuple): Size to resize the crop to
        data_dir (str): Directory containing the TIF files
        
    Attributes:
        data (numpy.ndarray): Array of shape (n_datasets, height, width) containing stacked data
        available_datasets (list): List of available dataset names
    """
    def __init__(self, 
                 h5_file,
                 etopo_file, 
                 mean_temp_file, 
                 std_temp_file, 
                 mean_precip_file, 
                 std_precip_file,
                 crop_size=16,
                 tile_px=26):
        self.h5_file = h5_file
        self.crop_size = crop_size
        if not os.path.exists(h5_file):
            print("Building HDF5 file...")
            with h5py.File(h5_file, 'w') as f:
                band_widths = np.zeros((50,), dtype=int)
                with rasterio.open(etopo_file) as src:
                    bounds = src.bounds
                    
                    # Calculate row indices for -60 to 60 degrees latitude
                    height = src.height
                    lat_res = (bounds.top - bounds.bottom) / height
                    start_row = int((bounds.top - 60) / lat_res)
                    end_row = int((bounds.top + 60) / lat_res)
                        
                    print("Reading ETOPO data...")
                    data = src.read(1)[start_row:end_row, :]
                    print("ETOPO data read")
                    
                    row_indices = np.linspace(0, data.shape[0], 10, dtype=int)
                    for i, (row_top, row_bottom) in tqdm(enumerate(zip(row_indices[:-1], row_indices[1:]))):
                        mid_latitude_deg = bounds.top - (row_top + row_bottom + start_row*2) / 2 * lat_res
                        lat_band = data[row_top:row_bottom, :]
                        lat_scaling = 1 / np.cos(np.deg2rad(mid_latitude_deg))
                        
                        scaled_height = lat_band.shape[0]
                        scaled_width = round(lat_band.shape[1] / lat_scaling)
                        scaled_lat_band = torch.nn.functional.interpolate(
                            torch.from_numpy(lat_band)[None, None, :, :], 
                            size=(scaled_height, scaled_width), 
                            mode='area', 
                            antialias=False)[0, 0].numpy()
                        
                        # Ensure its a multiple of tile_px
                        scaled_lat_band = scaled_lat_band[:scaled_height//tile_px*tile_px, :scaled_width//tile_px*tile_px]
                        tiled_scaled_lat_band = scaled_lat_band.reshape(scaled_height//tile_px, tile_px, scaled_width//tile_px, tile_px)
                        
                        # Aggregate
                        lat_band_means = tiled_scaled_lat_band.mean(axis=(1, 3))
                        lat_band_p5 = np.quantile(tiled_scaled_lat_band, q=0.05, axis=(1, 3))
                        
                        # Create the dataset
                        lat_band = np.zeros((6, lat_band_means.shape[0], lat_band_means.shape[1]))
                        lat_band[0] = lat_band_means
                        lat_band[1] = lat_band_means - lat_band_p5
                        f.create_dataset(f'gan_band_{i}', data=lat_band)
                        band_widths[i] = lat_band.shape[1]
                        
                for file_idx, file in enumerate([mean_temp_file, std_temp_file, mean_precip_file, std_precip_file]):
                    with rasterio.open(file) as src:
                        # Read the data within latitude bounds
                        print(f"Reading {file}...")
                        data = src.read(1)[start_row:end_row, :]
                        print(f"{file} read")
                        
                        if data.dtype == np.int16:
                            data = data.astype(np.float32)
                            data[np.abs(data - 32768) < 1e-6] = np.nan
                        else:
                            data[np.abs(data) > 1e6] = np.nan
                        
                        for i, (row_top, row_bottom) in tqdm(enumerate(zip(row_indices[:-1], row_indices[1:]))):
                            mid_latitude_deg = bounds.top - (row_top + row_bottom + start_row*2) / 2 * lat_res
                            lat_band = data[row_top:row_bottom, :]
                            lat_scaling = 1 / np.cos(np.deg2rad(mid_latitude_deg))
                            
                            scaled_height = lat_band.shape[0]
                            scaled_width = round(lat_band.shape[1] / lat_scaling)
                            scaled_lat_band = torch.nn.functional.interpolate(
                                torch.from_numpy(lat_band)[None, None, :, :], 
                                size=(scaled_height, scaled_width), 
                                mode='area', 
                                antialias=False)[0, 0].numpy()
                            
                            # Ensure its a multiple of tile_px
                            scaled_lat_band = scaled_lat_band[:scaled_height//tile_px*tile_px, :scaled_width//tile_px*tile_px]
                            tiled_scaled_lat_band = scaled_lat_band.reshape(scaled_height//tile_px, tile_px, scaled_width//tile_px, tile_px)
                            lat_band_means = np.nanmean(tiled_scaled_lat_band, axis=(1, 3))
                            lat_band_means = fill_oceans_wrap(lat_band_means)
                            f[f'gan_band_{i}'][file_idx+2] = lat_band_means
                            
                
                # Calculate mean and std across all GAN band datasets
                all_gan_data = []
                for i in range(len(f)):
                    gan_band = np.reshape(f[f'gan_band_{i}'][:], (6, -1))
                    all_gan_data.append(gan_band)
                
                all_gan_data = np.concatenate(all_gan_data, axis=1)
                means = np.mean(all_gan_data, axis=1)
                stds = np.std(all_gan_data, axis=1)
                
                print(f"GAN bands - mean: {means.tolist()}, std: {stds.tolist()}")
                f.attrs['means'] = means
                f.attrs['stds'] = stds
                f.attrs['band_weights'] = band_widths.astype(np.float32) / np.sum(band_widths).astype(np.float32)
        else:
            with h5py.File(self.h5_file, 'r') as f:
                means = f.attrs['means']
                stds = f.attrs['stds']
                print(f"GAN bands - mean: {means.tolist()}, std: {stds.tolist()}")
            
    def __len__(self):
        """Return a fixed length (can be adjusted based on needs)"""
        return 10000  # Arbitrary number of samples per epoch
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            band_weights = f.attrs['band_weights']
            band_idx = np.random.choice(len(band_weights), p=band_weights)
            data_shape = f[f'gan_band_{band_idx}'].shape
            
            # Get random crop indices
            max_i = data_shape[1] - self.crop_size
            max_j = data_shape[2] - self.crop_size
            i = random.randint(0, max_i)
            j = random.randint(0, max_j)
            
            # Load and crop the data
            with h5py.File(self.h5_file, 'r') as f:
                data = f[f'gan_band_{band_idx}'][:, i:i+self.crop_size, j:j+self.crop_size]
                data = (data - f.attrs['means'][:, None, None]) / f.attrs['stds'][:, None, None]
                
            data = torch.from_numpy(data).float()
        return data
    
if __name__ == "__main__":
    dataset = GANDataset(
        h5_file='data/gan.h5',
        etopo_file='data/global/ETOPO_2022_v1_30s_N90W180_bed.tif',
        mean_temp_file='data/global/wc2.1_30s_bio_1.tif',
        std_temp_file='data/global/wc2.1_30s_bio_4.tif',
        mean_precip_file='data/global/wc2.1_30s_bio_12.tif',
        std_precip_file='data/global/wc2.1_30s_bio_15.tif',
        crop_size=60,
    )
    
    import matplotlib.pyplot as plt
    
    # Create a 6x6 grid to show 6 examples with 6 channels each
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    
    for i in range(6):
        sample = dataset[i]
        for ch in range(6):
            ax = axes[i, ch]
            ax.matshow(sample[ch].numpy())
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Ch {ch}', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    print("Saved visualization to gan_dataset_samples.png")
