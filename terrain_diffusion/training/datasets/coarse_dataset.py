from functools import cache
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
from scipy import sparse
from scipy.sparse.linalg import cg

def fill_oceans(
    a: np.ndarray,
    *,
    tol: float = 1e-6,
    maxiter: int | None = None,
    anchor_stride: int | None = None,
    anchor_strength: float = 0.0,
    multires_factor: int = 8,
):
    """
    Fill ocean (NaN) pixels by solving a discrete Laplace equation on the ocean mask,
    using a multiresolution (coarse-to-fine) initialization for faster convergence.

    Steps:
      1) Downsample the temperature grid by `multires_factor` using NaN-aware block mean
         (land values averaged, ocean stays NaN if a coarse block has no land).
      2) Solve Laplace inpainting on the coarse grid.
      3) Upsample the coarse filled field (bilinear) to original size and use it as x0 for CG.
      4) Solve the full-resolution Laplace system with Dirichlet (coast) and Neumann (outer edge).

    Parameters
    ----------
    a : (H, W) np.ndarray
        Land a with ocean marked as np.nan.
    tol : float
        Relative tolerance for CG.
    maxiter : int or None
        Max CG iterations. Default ~ 10 * sqrt(#unknowns).
    anchor_stride : int or None
        Optional weak anchors grid stride (pixels) inside ocean to prevent large-scale drift.
    anchor_strength : float
        Strength of anchor penalty. Typical 1e-3 .. 1e-1. Ignored if anchor_stride is None.
    multires_factor : int
        Integer downsample factor for the coarse solve. Use 8 (default) as requested.

    Returns
    -------
    filled : (H, W) np.ndarray
        Ocean NaNs replaced with smooth, realistic values that match coast a exactly.
    """
    if a.ndim != 2:
        raise ValueError("a must be a 2D array")
    H, W = a.shape
    arr = a.astype(np.float64, copy=True)

    ocean = np.isnan(arr)
    if not np.any(ocean):
        return arr  # nothing to fill

    # ---------- helpers ----------
    def _build_system(arr_in: np.ndarray, ocean_mask: np.ndarray):
        """Build sparse Laplace system A x = b on unknown (ocean) pixels."""
        Hh, Wh = arr_in.shape
        idx_map = -np.ones((Hh, Wh), dtype=np.int64)
        ocean_idx = np.flatnonzero(ocean_mask.ravel())
        idx_map.ravel()[ocean_idx] = np.arange(ocean_idx.size)
        n_unknown = ocean_idx.size

        rows, cols, vals = [], [], []
        b = np.zeros(n_unknown, dtype=np.float64)

        land_vals = arr_in
        land_mask = ~ocean_mask

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        for k, lin in enumerate(ocean_idx):
            i = lin // Wh
            j = lin % Wh
            diag = 0

            for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < Hh and 0 <= nj < Wh:
                    diag += 1
                    if ocean_mask[ni, nj]:
                        n_idx = idx_map[ni, nj]
                        add(k, n_idx, -1.0)
                    else:
                        b[k] += land_vals[ni, nj]
                else:
                    # Neumann at image boundary: omit off-grid neighbor (reduces diag)
                    pass

            add(k, k, float(diag))

        return sparse.csr_matrix((vals, (rows, cols)), shape=(n_unknown, n_unknown)), b, idx_map, ocean_idx

    def _solve(arr_in: np.ndarray, ocean_mask: np.ndarray, *, x0_unknown: np.ndarray | None = None):
        """Solve Laplace with optional anchors and optional initial guess over unknowns."""
        A, b, idx_map, ocean_idx = _build_system(arr_in, ocean_mask)

        # Optional weak anchors (stabilizes vast oceans with near-uniform coasts)
        if anchor_stride is not None and anchor_strength > 0.0:
            land_mask = ~ocean_mask
            # Coastline values = land touching ocean (4-neighborhood)
            touch = np.zeros_like(land_mask, dtype=bool)
            # roll-based test for neighbors that are ocean
            touch |= land_mask & np.roll(ocean_mask,  1, axis=0)
            touch |= land_mask & np.roll(ocean_mask, -1, axis=0)
            touch |= land_mask & np.roll(ocean_mask,  1, axis=1)
            touch |= land_mask & np.roll(ocean_mask, -1, axis=1)
            coast_vals = arr_in[touch]
            target = (np.nanmean(coast_vals) if coast_vals.size > 0
                      else np.nanmean(arr_in[land_mask]))

            # place anchors on a grid
            anchor_mask = np.zeros_like(ocean_mask, dtype=bool)
            anchor_mask[::anchor_stride, ::anchor_stride] = True
            pos = np.flatnonzero(ocean_mask & anchor_mask)
            if pos.size:
                # Add lambda*I and lambda*target to RHS on those rows
                A = A.tolil()
                for lin in pos:
                    r = idx_map.ravel()[lin]
                    if r >= 0:
                        A[r, r] = A[r, r] + anchor_strength
                        b[r] += anchor_strength * target
                A = A.tocsr()

        # Preconditioner (Jacobi)
        M_diag = A.diagonal()
        M_inv = np.where(M_diag > 0, 1.0 / M_diag, 1.0)
        M = sparse.diags(M_inv)

        n_unknown = A.shape[0]
        iters = maxiter if maxiter is not None else max(50, int(10 * np.sqrt(n_unknown)))

        x0 = None
        if x0_unknown is not None:
            # Ensure the initial guess is well-formed
            if x0_unknown.shape != (n_unknown,):
                raise ValueError("x0_unknown has wrong shape")
            x0 = x0_unknown

        x, info = cg(A, b, M=M, atol=tol, maxiter=iters, x0=x0)

        # Map back
        out = arr_in.copy()
        out.ravel()[ocean_idx] = x

        # Protect against degenerate all-zero boundary case
        if np.allclose(out[ocean_mask], 0.0):
            out[ocean_mask] = 1e-9

        return out, info

    def _pad_to_multiple(a: np.ndarray, f: int):
        """Pad with NaNs to make both dims divisible by f."""
        Hh, Wh = a.shape
        pad_h = (f - Hh % f) % f
        pad_w = (f - Wh % f) % f
        if pad_h == 0 and pad_w == 0:
            return a, (0, 0)
        ap = np.full((Hh + pad_h, Wh + pad_w), np.nan, dtype=a.dtype)
        ap[:Hh, :Wh] = a
        return ap, (pad_h, pad_w)

    def _block_nanmean(a: np.ndarray, f: int):
        """Downsample by factor f via NaN-aware block mean; if a block is all-NaN, result is NaN."""
        ap, (ph, pw) = _pad_to_multiple(a, f)
        Hh, Wh = ap.shape
        resh = ap.reshape(Hh // f, f, Wh // f, f)
        # mean over block, ignoring NaNs
        with np.errstate(invalid="ignore"):
            m = np.nanmean(np.nanmean(resh, axis=3), axis=1)
        return m

    # ---------- coarse solve ----------
    f = int(multires_factor)
    if f < 1:
        raise ValueError("multires_factor must be >= 1")

    # Downsample a with NaN-aware block mean
    coarse = _block_nanmean(arr, f)
    # Preserve ocean vs land at coarse scale: a coarse cell is land if ANY land exists in the block
    # Build a coarse mask by checking if *all* values in a block were NaN -> coarse ocean
    ocean_mask_full, (ph, pw) = _pad_to_multiple(ocean.astype(float), f)
    ocean_mask_full = np.isnan(np.where(ocean_mask_full > 0.5, 1.0, np.nan))
    # Equivalent: a block is ocean iff all original were ocean (i.e., no land present)
    # We can compute: block_has_land = nanmean(1-land) < 1  (but simpler to recompute from arr)
    # Let's compute from arr: block is ocean if all NaN in a
    coarse_ocean = np.isnan(coarse)

    # Solve on coarse grid
    coarse_filled, _ = _solve(coarse, coarse_ocean)

    # Upsample coarse_filled back to original size (bilinear)
    zoom_h = H / coarse_filled.shape[0]
    zoom_w = W / coarse_filled.shape[1]
    coarse_up = zoom(coarse_filled, (zoom_h, zoom_w), order=1, mode="nearest")
    # Crop in case of rounding
    coarse_up = coarse_up[:H, :W]

    # Initial guess only for ocean unknowns (copy from coarse_up)
    x0_unknown = coarse_up[ocean].ravel()

    # ---------- fine solve ----------
    filled, info = _solve(arr, ocean, x0_unknown=x0_unknown)

    if info > 0:
        print(f"Warning: CG solver did not converge within {maxiter} iterations")
    return filled

class CoarseDataset(Dataset):
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
                 tile_px=26,
                 sigma_data=0.5):
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.sigma_data = sigma_data
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
                            lat_band_means = fill_oceans(lat_band_means, tol=1e-2, maxiter=None)
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
    
    @cache
    def get_band_weights(self):
        with h5py.File(self.h5_file, 'r') as f:
            return f.attrs['band_weights']
    
    @cache
    def get_band(self, band_idx):
        with h5py.File(self.h5_file, 'r') as f:
            data = f[f'gan_band_{band_idx}']
            data = (data - f.attrs['means'][:, None, None]) / f.attrs['stds'][:, None, None]
            return torch.from_numpy(data).float()

    def __getitem__(self, idx):
        band_weights = self.get_band_weights()
        band_idx = np.random.choice(len(band_weights), p=band_weights)
        
        data = self.get_band(band_idx)
        data_shape = data.shape
        
        # Get random crop indices
        max_i = data_shape[1] - self.crop_size
        max_j = data_shape[2] - self.crop_size
        i = random.randint(0, max_i)
        j = random.randint(0, max_j)
        
        data = data[:, i:i+self.crop_size, j:j+self.crop_size]
        
        if random.random() > 0.5:
            data = torch.flip(data, dims=[-2])
        k = random.randint(0, 3)
        if k > 0:
            data = torch.rot90(data, k=k, dims=[-2, -1])
        data = data * self.sigma_data
        
        t = torch.atan(torch.exp(8 * torch.rand(5) - 3)).view(-1, 1, 1)
        cond_img = data[[0, 2, 3, 4, 5]] 
        cond_img = cond_img * torch.cos(t) + torch.randn_like(cond_img) * torch.sin(t)
        
        return {'image': data, 'cond_img': cond_img, 'cond_inputs': [torch.log(torch.tan(s) / 8) for s in t.flatten()]}
    
if __name__ == "__main__":
    dataset = CoarseDataset(
        h5_file='data/coarse.h5',
        etopo_file='data/global/ETOPO_2022_v1_30s_N90W180_bed.tif',
        mean_temp_file='data/global/wc2.1_30s_bio_1.tif',
        std_temp_file='data/global/wc2.1_30s_bio_4.tif',
        mean_precip_file='data/global/wc2.1_30s_bio_12.tif',
        std_precip_file='data/global/wc2.1_30s_bio_15.tif',
        crop_size=32,
    )
    
    import matplotlib.pyplot as plt
    
    # Create a 6x6 grid to show 6 examples with 6 channels each
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    
    for i in range(6):
        sample = dataset[i]['image']
        for ch in range(6):
            ax = axes[i, ch]
            ax.matshow(sample[ch].numpy())
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Ch {ch}', fontsize=10)
    
    plt.tight_layout()
    plt.show()
