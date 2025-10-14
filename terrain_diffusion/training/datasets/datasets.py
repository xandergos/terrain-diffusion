from functools import lru_cache
import itertools
import json
import math
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from tqdm import tqdm
from .transforms import TupleTransform
import h5py
from torchvision.datasets import CIFAR10
from collections import deque
import torch.nn.functional as F
import rasterio
import glob

class AreaResize(torch.nn.Module):
    """
    Resizes a square image using adaptive mean pooling.
    """
    def __init__(self, output_size):
        """
        Args:
            output_size (int): Desired output size for both height and width.
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, img):
        """
        Args:
            img (Tensor): Square image to be resized.

        Returns:
            Tensor: Resized square image.
        """
        return torch.nn.functional.adaptive_avg_pool2d(img, self.output_size)
    
class H5AutoencoderDataset(Dataset):
    """Dataset for reading high frequency terrain data from an HDF5 file."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 split=None,
                 residual_mean=None,
                 residual_std=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset.
            subset_resolutions (list): Resolutions to filter subsets by.
            subset_weights (list): Weights for each subset. Default is None (uniform sampling).
            subset_class_labels (list): Class labels for each subset. Defaults to None.
            eval_dataset (bool): Whether to use deterministic transforms. Defaults to False.
            split (str): Split to use. Defaults to None (all splits).
            sigma_data (float): Data standard deviation. Defaults to 0.5.
        """
        if subset_weights is None:
            subset_weights = [1] * len(pct_land_ranges)
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges
        self.subset_resolutions = subset_resolutions
        self.subset_weights = subset_weights
        self.subset_class_labels = subset_class_labels
        self.eval_dataset = eval_dataset
        
        # Initialize keys
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        self.keys = [set() for _ in range(num_subsets)]
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'residual' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['residual']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            self.keys[i].add((chunk_id, res, subchunk_id))
        self.keys = [list(keys) for keys in self.keys]
        
        self.residual_mean = residual_mean
        self.residual_std = residual_std
        if self.residual_mean is None or self.residual_std is None:
            self.calculate_stats()
        
        
    def calculate_stats(self, num_samples=10000):
        """Compute per-channel mean and std using a streaming Welford algorithm.

        This avoids stacking samples in memory and works for any number of channels
        returned by __getitem__ under key 'image' with shape [C, H, W].
        """
        torch.set_grad_enabled(False)

        running_count = None
        running_mean = None
        running_m2 = None

        for _ in tqdm(range(num_samples), desc="Calculating stats"):
            sample = self.__getitem__(random.randrange(len(self)))
            x = sample['image']
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
                
            c, h, w = x.shape
            x_flat = x.reshape(c, -1)
            batch_count = x_flat.shape[1]

            # Per-channel batch stats
            batch_mean = x_flat.mean(dim=1)
            batch_m2 = x_flat.var(dim=1, unbiased=False) * batch_count

            if running_count is None:
                running_count = torch.full_like(batch_mean, fill_value=batch_count, dtype=torch.float32)
                running_mean = batch_mean.clone().to(torch.float32)
                running_m2 = batch_m2.clone().to(torch.float32)
                continue

            # Combine current running aggregates with this batch (per-channel)
            total_count = running_count + batch_count
            delta = batch_mean - running_mean
            running_mean = running_mean + delta * (batch_count / total_count)
            running_m2 = running_m2 + batch_m2 + (delta.pow(2) * running_count * batch_count / total_count)
            running_count = total_count

        # Finalize
        variance = running_m2 / running_count
        std = variance.sqrt()

        # Print concise results per channel
        for ch, (m, s) in enumerate(zip(running_mean.tolist(), std.tolist())):
            print(f"Channel {ch}: mean={m:.6f}, std={s:.6f}")
        
        self.channel_means = running_mean
        self.channel_stds = std

        torch.set_grad_enabled(True)

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, index):
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            res_group = f[str(res)]
            
            # Get crop indices
            dset = f[f"{group_path}/residual"]
            data_shape = dset.shape
            
            if not self.eval_dataset:
                # Ensure i and j are multiples of 8 by first dividing by 8, then multiplying back
                max_i = (data_shape[-2] - self.crop_size) // 8
                max_j = (data_shape[-1] - self.crop_size) // 8
                i = random.randint(0, max_i) * 8
                j = random.randint(0, max_j) * 8
            else:
                # Center crop, ensuring it's aligned to 8-pixel boundaries
                i = ((data_shape[-2] - self.crop_size) // 16) * 8
                j = ((data_shape[-1] - self.crop_size) // 16) * 8
            h, w = self.crop_size, self.crop_size
            
            # Load residual data (elevation)
            residual_data = torch.from_numpy(dset[i:i+h, j:j+w])[None]
            residual_data = (residual_data - self.residual_mean) / self.residual_std
            
            data = residual_data.float()
            
        # Apply transforms
        transform_idx = random.randrange(8) if not self.eval_dataset else 0
        flip = (transform_idx // 4) == 1
        rotate_k = transform_idx % 4
            
        if flip:
            data = torch.flip(data, dims=[-1])
        if rotate_k != 0:
            data = torch.rot90(data, k=rotate_k, dims=[-2, -1])
            
        if class_label is not None:
            return {'image': data, 'cond_inputs': [torch.tensor(class_label)]}
        else:
            return {'image': data}
    
    def denormalize_residual(self, residual, resolution):
        with h5py.File(self.h5_file, 'r') as f:
            res_group = f[str(resolution)]
            return residual * res_group.attrs['residual_std'] + res_group.attrs['residual_mean']

class H5DecoderTerrainDataset(Dataset):
    """Dataset for reading terrain data from an HDF5 file with an encoding."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 sigma_data=0.5, 
                 clip_edges=True,
                 split=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset.
            subset_resolutions (list): Resolutions to filter subsets by.
            subset_weights (list): Weights for each subset. Default is None (uniform sampling).
            subset_class_labels (list): Class labels for each subset. Defaults to None.
            eval_dataset (bool): Whether to use deterministic transforms. Defaults to False.
            split (str): Split to use. Defaults to None (all splits).
            sigma_data (float): Data standard deviation. Defaults to 0.5.
        """
        if subset_weights is None:
            subset_weights = [1] * len(pct_land_ranges)
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges
        self.subset_resolutions = subset_resolutions
        self.subset_weights = subset_weights
        self.subset_class_labels = subset_class_labels
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.sigma_data = sigma_data
        
        # Define which climate channels to use and number of landcover classes
        self.climate_channels = [0, 3, 11, 14]
        self.num_landcover_classes = 23
        
        # Initialize keys
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        self.keys = [set() for _ in range(num_subsets)]
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'residual' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['residual']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            self.keys[i].add((chunk_id, res, subchunk_id))
        self.keys = [list(keys) for keys in self.keys]

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, index):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        index = random.randrange(len(self.keys[subset_idx]))
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            res_group = f[str(res)]
            data_latent = f[f"{group_path}/latent"]
            
            latent_shape = data_latent.shape
            residual_shape = f[f"{group_path}/residual"].shape
            
            if self.clip_edges:
                assert residual_shape[1] >= self.crop_size + 2, f"Crop size is larger than image size + 2. Crop size: {self.crop_size}, Image size: {residual_shape[2]}"
            else:
                assert residual_shape[1] >= self.crop_size, f"Crop size is larger than image size. Crop size: {self.crop_size}, Image size: {residual_shape[2]}"
        
            if not self.eval_dataset:
                if self.clip_edges:
                    i = random.randint(1, latent_shape[2] - self.crop_size // 8 - 1)
                    j = random.randint(1, latent_shape[3] - self.crop_size // 8 - 1)
                else:
                    i = random.randint(0, latent_shape[2] - self.crop_size // 8)
                    j = random.randint(0, latent_shape[3] - self.crop_size // 8)
            else:
                i = (latent_shape[2] - self.crop_size // 8) // 2
                j = (latent_shape[3] - self.crop_size // 8) // 2
                
            h = w = self.crop_size // 8
            li, lj, lh, lw = i*8, j*8, h*8, w*8
                
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Adjust lowfreq crop for flips and rotations. Note that we are inverting the transformation so we do it in reverse.
            for _ in range(rotate_k):
                li, lj = lj, residual_shape[1] - li - lh
            if flip:
                lj = residual_shape[1] - lj - lw
            
            # Load latent data
            data_latent = torch.from_numpy(data_latent[transform_idx, :, i:i+h, j:j+w])
            assert data_latent.shape[-1] > 0, f"Latent crop is empty. i: {i}, j: {j}, h: {h}, w: {w}"
            latent_channels = data_latent.shape[0]
            means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
            sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
            
            # Load residual data
            data_residual = torch.from_numpy(f[f"{group_path}/residual"][li:li+lh, lj:lj+lw])[None]
            data_residual = (data_residual - res_group.attrs['residual_mean']) / res_group.attrs['residual_std'] * self.sigma_data
            if flip:
                data_residual = torch.flip(data_residual, dims=[-1])
            if rotate_k != 0:
                data_residual = torch.rot90(data_residual, k=rotate_k, dims=[-2, -1])
            
            # Load watercover data
            try:
                data_watercover = torch.from_numpy(f[f"{group_path}/watercover"][li:li+lh, lj:lj+lw])[None] / 100
                data_watercover = data_watercover * self.sigma_data
                if flip:
                    data_watercover = torch.flip(data_watercover, dims=[-1])
                if rotate_k != 0:
                    data_watercover = torch.rot90(data_watercover, k=rotate_k, dims=[-2, -1])
            except KeyError:
                data_watercover = torch.zeros((1, lh, lw))
                
        image = torch.cat([
            data_residual,
            data_watercover
        ], dim=0)
        
        cond_image = F.interpolate(sampled_latent[None], size=(self.crop_size, self.crop_size), mode='nearest')[0]
        
        # Only include sampled_latent in img tensor instead of concatenating with other data
        if class_label is not None:
            cond_inputs = [torch.tensor(class_label)]
        else:
            cond_inputs = []
            
        return {'image': image.float(), 'cond_img': cond_image, 'cond_inputs': cond_inputs, 'path': group_path}
        

class H5UpsamplingTerrainDataset(Dataset):
    """Dataset for reading terrain data from an HDF5 file for upsampling tasks.
    This dataset uses downsampled versions of terrain as conditioning instead of latent encodings.
    """
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 downsample_factor=2,
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 sigma_data=0.5, 
                 clip_edges=True,
                 split=None,
                 use_watercover=False,
                 require_watercover=False,
                 downsample_sizes=[8]):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset.
            subset_resolutions (list): Resolutions to filter subsets by.
            downsample_factor (int): Factor by which to downsample the terrain (2 or 4).
            subset_weights (list): Weights for each subset for sampling probability.
            subset_class_labels (list): Class labels for each subset.
            eval_dataset (bool): Whether to use deterministic transforms.
            sigma_data (float): Data standard deviation.
            clip_edges (bool): Whether to clip edges when cropping.
            split (str): Dataset split to use.
            use_watercover (bool): Whether to use watercover data.
            require_watercover (bool): Whether to require watercover data.
            downsample_sizes (list): Sizes to downsample the terrain for conditioning (equal weights).
        """
        super().__init__()
        if subset_weights is None:
            subset_weights = [1] * len(pct_land_ranges)
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges
        self.subset_resolutions = subset_resolutions
        self.subset_weights = subset_weights
        self.subset_class_labels = subset_class_labels
        self.sigma_data = sigma_data
        self.clip_edges = clip_edges
        self.eval_dataset = eval_dataset
        self.use_watercover = use_watercover
        self.require_watercover = require_watercover
        self.downsample_sizes = downsample_sizes
        
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        self.keys = [set() for _ in range(num_subsets)]
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'residual' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['residual']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            self.keys[i].add((chunk_id, res, subchunk_id))
        self.keys = [list(keys) for keys in self.keys]

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, index):
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        downsample_size = random.choice(self.downsample_sizes)
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r', rdcc_nbytes=16*1024**2) as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            residual_dset = f[f"{group_path}/residual"]
            
            if not self.eval_dataset:
                if self.clip_edges:
                    i = random.randint(1, residual_dset.shape[0] - self.crop_size - 1)
                    j = random.randint(1, residual_dset.shape[1] - self.crop_size - 1)
                else:
                    i = random.randint(0, residual_dset.shape[0] - self.crop_size)
                    j = random.randint(0, residual_dset.shape[1] - self.crop_size)
            else:
                i = (residual_dset.shape[0] - self.crop_size) // 2
                j = (residual_dset.shape[1] - self.crop_size) // 2
                
            # Extract crop
            data_residual = torch.from_numpy(residual_dset[i:i+self.crop_size, j:j+self.crop_size])[None]
            
            # Normalize residual
            residual_std = f[str(res)].attrs['residual_std']
            residual_mean = f[str(res)].attrs['residual_mean']
            data_residual = (data_residual - residual_mean) / residual_std * self.sigma_data
            
            # Load watercover if requested
            if self.use_watercover:
                try:
                    water_data = torch.from_numpy(f[f"{group_path}/watercover"][i:i+self.crop_size, j:j+self.crop_size])[None] / 100 * self.sigma_data
                except KeyError:
                    water_data = torch.zeros_like(data_residual)
                img = torch.cat([data_residual, water_data], dim=0)
            else:
                img = data_residual
            
            # Apply random transforms
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            if flip:
                img = torch.flip(img, dims=[-1])
            if rotate_k != 0:
                img = torch.rot90(img, k=rotate_k, dims=[-2, -1])
            
            # Create downsampled version for conditioning
            cond_img = torch.nn.functional.adaptive_avg_pool2d(
                img[None],
                output_size=(downsample_size, downsample_size)
            )[0]
            
            # Upsample conditioning image back to original size using nearest neighbor interpolation
            cond_img = torch.nn.functional.interpolate(
                cond_img[None],
                size=(self.crop_size, self.crop_size),
                mode='nearest'
            )[0]
            
            return {'image': img, 'cond_img': cond_img}

class H5LatentsDataset(Dataset):
    """Dataset for training diffusion models to generate terrain latents."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 latents_mean=None, 
                 latents_std=None, 
                 sigma_data=0.5, 
                 clip_edges=True,
                 split=None,
                 beauty_dist=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset. Elements can be None to include everything.
            subset_resolutions (list): Resolutions to filter subsets by. Each subset will only include elevation data with the corresponding resolution. Elements be None to include all resolutions.
            subset_weights (list): Weights for each subset, determining the relative probability of sampling from that subset. Default is None, which results in uniform sampling.
            subset_class_labels (list): Class labels for each subset, determining the class of each subset. Defaults to None, which results in no class labels being returned.
            eval_dataset (bool, optional): Whether the dataset should be transformed deterministically. Defaults to False.
            latents_mean (list, optional): Mean values for normalizing latents. Defaults to None.
            latents_std (list, optional): Standard deviation values for normalizing latents. Defaults to None.
            sigma_data (float, optional): Data standard deviation. Defaults to 0.5.
            clip_edges (bool, optional): Whether to clip edges when cropping. Defaults to True.
            split (str, optional): Split to use. Defaults to None (all splits).
            beauty_dist (list, optional): Weights for sampling beauty scores 1-5. 
                                        Must sum to 1. Defaults to uniform distribution.
        """
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges or [[0, 1]]
        self.subset_resolutions = subset_resolutions or [480]
        self.subset_weights = subset_weights or [1.0]
        self.subset_class_labels = subset_class_labels
        self.latents_mean = torch.tensor(latents_mean).view(-1, 1, 1) if isinstance(latents_mean, list) else torch.clone(latents_mean).view(-1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(-1, 1, 1) if isinstance(latents_std, list) else torch.clone(latents_std).view(-1, 1, 1)
        self.sigma_data = sigma_data
        self.split = split
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.beauty_dist = beauty_dist or None
        
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        if beauty_dist is not None:
            self.keys = [[set() for _ in range(5)] for _ in range(num_subsets)]
        else:
            self.keys = [set() for _ in range(num_subsets)]
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'latent' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['latent']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            if beauty_dist is not None:
                                beauty_score = float(subchunk_group.attrs['beauty_score'])
                                beauty_score = max(1, min(5, round(beauty_score))) - 1
                                self.keys[i][beauty_score].add((chunk_id, res, subchunk_id))
                            else:
                                self.keys[i].add((chunk_id, res, subchunk_id))
        if beauty_dist is not None:
            self.keys = [[list(subkeys) for subkeys in keys] for keys in self.keys]
            print("Using beauty distribution. Have sizes:", [[len(subkeys) for subkeys in keys] for keys in self.keys])
        else:
            self.keys = [list(keys) for keys in self.keys]
            print("Not using beauty distribution.")

    def __len__(self):
        return 100000

    def augment_onehot(self, onehot):
        """Augments onehot in place by randomly adding classes to pixels."""
        # Get indices of present classes
        present_indices = torch.where(onehot.sum(dim=(-2,-1)) > 0)[0]
        
        # Randomly choose number of partitions (k)
        k = random.randint(1, len(present_indices))
        
        # Randomly shuffle and split present indices into k groups
        shuffled_indices = present_indices[torch.randperm(len(present_indices))]
        
        # Create random split sizes that sum to total length
        split_points = np.sort(np.random.choice(len(present_indices)-1, k-1, replace=False) + 1)
        split_sizes = np.diff(np.concatenate([[0], split_points, [len(present_indices)]]))
        
        # Create groups based on random split sizes
        start = 0
        index_groups = []
        for size in split_sizes:
            index_groups.append(shuffled_indices[start:start+size])
            start += size
        
        # For each group, perform separate augmentation
        for group_indices in index_groups:
            # Create mask for pixels belonging to any class in this group
            group_mask = onehot[group_indices].sum(dim=0) > 0
            
            # Set all classes in this group to 1 where group mask is True
            for idx in range(len(group_indices)):
                onehot[group_indices[idx]][group_mask] = 1
            
            # For classes in this group that aren't present in these pixels,
            # randomly decide whether to include additional ones
            zero_indices = torch.where(onehot[:].sum(dim=(-2,-1)) == 0)[0]
            if len(zero_indices) > 0:
                num_to_set = random.randint(0, len(zero_indices))
                indices_to_set = zero_indices[torch.randperm(len(zero_indices))[:num_to_set]]
                
                # Set selected classes to 1 where group mask is True
                for idx in indices_to_set:
                    onehot[idx][group_mask] = 1
                        
    def __getitem__(self, idx):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        if self.beauty_dist is not None:
            beauty_score = random.choices(range(5), weights=self.beauty_dist[subset_idx], k=1)[0]
            index = random.randrange(len(self.keys[subset_idx][beauty_score]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][beauty_score][index]
        else:
            index = random.randrange(len(self.keys[subset_idx]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            res_group = f[str(res)]
            data_latent = f[f"{group_path}/latent"]
            data_lowfreq = f[f"{group_path}/lowfreq"]
            data_climate = f[f"{group_path}/climate"]
            
            shape = data_latent.shape
            assert data_lowfreq.shape == shape[2:]
            assert data_climate.shape[1:] == shape[2:]
            
            if self.clip_edges:
                assert shape[2] >= self.crop_size + 2, f"Crop size is larger than image size + 2. Crop size: {self.crop_size}, Image size: {shape[2]}"
            else:
                assert shape[2] >= self.crop_size, f"Crop size is larger than image size. Crop size: {self.crop_size}, Image size: {shape[2]}"
        
            if not self.eval_dataset:
                if self.clip_edges:
                    i = random.randint(1, shape[2] - self.crop_size - 1)
                    j = random.randint(1, shape[3] - self.crop_size - 1)
                else:
                    i = random.randint(0, shape[2] - self.crop_size)
                    j = random.randint(0, shape[3] - self.crop_size)
            else:
                i = (shape[2] - self.crop_size) // 2
                j = (shape[3] - self.crop_size) // 2
                
            h = w = self.crop_size
            li, lj, lh, lw = i, j, h, w
                
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Adjust lowfreq crop for flips and rotations. Note that we are inverting the transformation so we do it in reverse.
            for _ in range(rotate_k):
                li, lj = lj, shape[2] - li - lh
            if flip:
                lj = shape[2] - lj - lw
            
            # Load latent data
            data_latent = torch.from_numpy(data_latent[transform_idx, :, i:i+h, j:j+w])
            assert data_latent.shape[-1] > 0, f"Latent crop is empty. i: {i}, j: {j}, h: {h}, w: {w}"
            latent_channels = data_latent.shape[0]
            means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
            sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
            sampled_latent = (sampled_latent - self.latents_mean) / self.latents_std * self.sigma_data
            
            # Load lowfreq data
            data_lowfreq = torch.from_numpy(data_lowfreq[li:li+lh, lj:lj+lw])[None]
            if flip:
                data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
            if rotate_k != 0:
                data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
            data_lowfreq = data_lowfreq.float()
            data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD * self.sigma_data
            lowfreq_mean = torch.mean(data_lowfreq) / self.sigma_data
            
            # Load climate data
            # (0=temp, 3=temp seasonality, 11=precip, 14=precip seasonality)
            if f"{group_path}/climate" in f:
                climate_data = f[f"{group_path}/climate"][[0, 3, 11, 14]][:, li:li+lh, lj:lj+lw]
                climate_data = (climate_data - res_group.attrs['climate_mean'][[0, 3, 11, 14], None, None]) / res_group.attrs['climate_std'][[0, 3, 11, 14], None, None] * self.sigma_data
                climate_data = torch.from_numpy(climate_data).float()
                any_nan_climate = torch.isnan(climate_data).all(dim=(-2, -1)).any().item()
                if any_nan_climate:
                    climate_nanmean = torch.zeros(4)
                else:
                    climate_nanmean = torch.nanmean(climate_data, dim=(-2, -1))
                climate_data_mask = torch.isnan(climate_data).any(dim=0).float().unsqueeze(0) * np.sqrt(2) * self.sigma_data
                climate_data = torch.nan_to_num(climate_data, nan=0.0)
            else:
                climate_data = torch.zeros(4, lh, lw)
                climate_data_mask = torch.zeros(1, lh, lw)
                climate_data_mask[0] = np.sqrt(2) * self.sigma_data
                climate_nanmean = torch.zeros(4)
                any_nan_climate = True
            
        image = torch.cat([
            sampled_latent,
            data_lowfreq,
            climate_data,
            climate_data_mask
        ], dim=0)
        
        # Only include sampled_latent in img tensor instead of concatenating with other data
        cond_inputs = [lowfreq_mean.reshape([]).float(),
                       climate_nanmean[0].float().reshape([]),
                       climate_nanmean[1].float().reshape([]),
                       climate_nanmean[2].float().reshape([]),
                       climate_nanmean[3].float().reshape([]),
                       torch.tensor(1 if any_nan_climate else 0, dtype=torch.int64).reshape([])]
        if class_label is not None:
            cond_inputs += [torch.tensor(class_label)]
            
        return {'image': image.float(), 'cond_inputs': cond_inputs, 'path': group_path}

    def denormalize_residual(self, residual, resolution):
        with h5py.File(self.h5_file, 'r') as f:
            res_group = f[str(resolution)]
            return residual * res_group.attrs['residual_std'] + res_group.attrs['residual_mean']
    
    def denormalize_lowfreq(self, lowfreq, resolution=None):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        return lowfreq * LOWFREQ_STD + LOWFREQ_MEAN

class H5LatentsSimpleDataset(Dataset):
    """Simplified dataset for training diffusion models to generate terrain latents without climate/landcover data."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 latents_mean=None, 
                 latents_std=None, 
                 sigma_data=0.5, 
                 clip_edges=True,
                 split=None,
                 beauty_dist=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset.
            subset_resolutions (list): Resolutions to filter subsets by.
            subset_weights (list): Weights for each subset. Default is None (uniform sampling).
            subset_class_labels (list): Class labels for each subset. Defaults to None.
            eval_dataset (bool): Whether dataset should be transformed deterministically. Defaults to False.
            latents_mean (list): Mean values for normalizing latents. Defaults to None.
            latents_std (list): Standard deviation values for normalizing latents. Defaults to None.
            sigma_data (float): Data standard deviation. Defaults to 0.5.
            clip_edges (bool): Whether to clip edges when cropping. Defaults to True.
            split (str): Split to use. Defaults to None (all splits).
            beauty_dist (list): Weights for sampling beauty scores 1-5. Must sum to 1. Defaults to uniform distribution.
        """
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges or [[0, 1]]
        self.subset_resolutions = subset_resolutions or [480]
        self.subset_weights = subset_weights or [1.0]
        self.subset_class_labels = subset_class_labels
        self.latents_mean = torch.tensor(latents_mean).view(-1, 1, 1) if isinstance(latents_mean, list) else torch.clone(latents_mean).view(-1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(-1, 1, 1) if isinstance(latents_std, list) else torch.clone(latents_std).view(-1, 1, 1)
        self.sigma_data = sigma_data
        self.split = split
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.beauty_dist = beauty_dist or None
        
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        # Initialize keys based on whether beauty distribution is used
        if beauty_dist is not None:
            self.keys = [[set() for _ in range(5)] for _ in range(num_subsets)]
        else:
            self.keys = [set() for _ in range(num_subsets)]
            
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'latent' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['latent']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            if beauty_dist is not None:
                                beauty_score = float(subchunk_group.attrs['beauty_score'])
                                beauty_score = max(1, min(5, round(beauty_score))) - 1
                                self.keys[i][beauty_score].add((chunk_id, res, subchunk_id))
                            else:
                                self.keys[i].add((chunk_id, res, subchunk_id))
                                
        if beauty_dist is not None:
            self.keys = [[list(subkeys) for subkeys in keys] for keys in self.keys]
            print("Using beauty distribution. Have sizes:", [[len(subkeys) for subkeys in keys] for keys in self.keys])
        else:
            self.keys = [list(keys) for keys in self.keys]
            print("Not using beauty distribution.")

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        if self.beauty_dist is not None:
            beauty_score = random.choices(range(5), weights=self.beauty_dist[subset_idx], k=1)[0]
            index = random.randrange(len(self.keys[subset_idx][beauty_score]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][beauty_score][index]
        else:
            index = random.randrange(len(self.keys[subset_idx]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            data_latent = f[f"{group_path}/latent"]
            data_lowfreq = f[f"{group_path}/lowfreq"]
            
            shape = data_latent.shape
            assert data_lowfreq.shape == shape[2:]
            
            if self.clip_edges:
                assert shape[2] >= self.crop_size + 2
                i = random.randint(1, shape[2] - self.crop_size - 1) if not self.eval_dataset else (shape[2] - self.crop_size) // 2
                j = random.randint(1, shape[3] - self.crop_size - 1) if not self.eval_dataset else (shape[3] - self.crop_size) // 2
            else:
                assert shape[2] >= self.crop_size
                i = random.randint(0, shape[2] - self.crop_size) if not self.eval_dataset else (shape[2] - self.crop_size) // 2
                j = random.randint(0, shape[3] - self.crop_size) if not self.eval_dataset else (shape[3] - self.crop_size) // 2
                
            h = w = self.crop_size
            
            # Handle transformations
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
                
            # Load and process latent data
            data_latent = torch.from_numpy(data_latent[transform_idx, :, i:i+h, j:j+w])
            latent_channels = data_latent.shape[0]
            means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
            sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
            sampled_latent = (sampled_latent - self.latents_mean) / self.latents_std * self.sigma_data
            
            # Load and process lowfreq data
            li, lj, lh, lw = i, j, h, w
            for _ in range(rotate_k):
                li, lj = lj, shape[2] - li - lh
            if flip:
                lj = shape[2] - lj - lw
                
            data_lowfreq = torch.from_numpy(data_lowfreq[li:li+lh, lj:lj+lw])[None]
            if flip:
                data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
            if rotate_k != 0:
                data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
            data_lowfreq = data_lowfreq.float()
            water_mask = (data_lowfreq < 0).float() * 2 - 1
            data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD * self.sigma_data
            lowfreq_mean = torch.mean(data_lowfreq) / self.sigma_data
            
        image = torch.cat([sampled_latent, data_lowfreq], dim=0)
        #cond_img = torch.cat([data_lowfreq, water_mask], dim=0) / self.sigma_data
        
        cond_inputs = [lowfreq_mean.reshape([]).float()]
        if class_label is not None:
            cond_inputs += [torch.tensor(class_label)]
            
        return {'image': image.float(), 'cond_inputs': cond_inputs, 'path': group_path}

    def denormalize_residual(self, residual, resolution):
        with h5py.File(self.h5_file, 'r') as f:
            res_group = f[str(resolution)]
            return residual * res_group.attrs['residual_std'] + res_group.attrs['residual_mean']
        
    def denormalize_lowfreq(self, lowfreq, resolution=None):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        return lowfreq * LOWFREQ_STD + LOWFREQ_MEAN

class H5DirectDataset(Dataset):
    """Dataset for training diffusion models directly on terrain residuals without using latent encodings.
    This dataset downsamples the residual data to match the latent resolution."""
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 sigma_data=0.5, 
                 clip_edges=True,
                 split=None,
                 beauty_dist=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset.
            subset_resolutions (list): Resolutions to filter subsets by.
            subset_weights (list): Weights for each subset. Default is None (uniform sampling).
            subset_class_labels (list): Class labels for each subset. Defaults to None.
            eval_dataset (bool): Whether dataset should be transformed deterministically. Defaults to False.
            sigma_data (float): Data standard deviation. Defaults to 0.5.
            clip_edges (bool): Whether to clip edges when cropping. Defaults to True.
            split (str): Split to use. Defaults to None (all splits).
            beauty_dist (list): Weights for sampling beauty scores 1-5. Must sum to 1. Defaults to uniform distribution.
        """
        self.h5_file = h5_file
        self.crop_size = crop_size  # This is the size after downsampling (same as latent size)
        self.pct_land_ranges = pct_land_ranges or [[0, 1]]
        self.subset_resolutions = subset_resolutions or [480]
        self.subset_weights = subset_weights or [1.0]
        self.subset_class_labels = subset_class_labels
        self.sigma_data = sigma_data
        self.split = split
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.beauty_dist = beauty_dist or None
        
        num_subsets = len(subset_weights)
        assert len(pct_land_ranges) == len(subset_resolutions) == num_subsets, \
            "Number of subsets must match between pct_land_ranges, dataset_resolutions, and subset_weights."
        assert subset_class_labels is None or len(subset_class_labels) == num_subsets, \
            "Number of subset class labels must match number of subsets."
        
        # Initialize keys based on whether beauty distribution is used
        if beauty_dist is not None:
            self.keys = [[set() for _ in range(5)] for _ in range(num_subsets)]
        else:
            self.keys = [set() for _ in range(num_subsets)]
            
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'residual' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['residual']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            if beauty_dist is not None and 'beauty_score' in subchunk_group.attrs:
                                beauty_score = float(subchunk_group.attrs['beauty_score'])
                                beauty_score = max(1, min(5, round(beauty_score))) - 1
                                self.keys[i][beauty_score].add((chunk_id, res, subchunk_id))
                            else:
                                self.keys[i].add((chunk_id, res, subchunk_id))
                                
        if beauty_dist is not None:
            self.keys = [[list(subkeys) for subkeys in keys] for keys in self.keys]
            print("Using beauty distribution. Have sizes:", [[len(subkeys) for subkeys in keys] for keys in self.keys])
        else:
            self.keys = [list(keys) for keys in self.keys]
            print("Not using beauty distribution.")

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        if self.beauty_dist is not None:
            beauty_score = random.choices(range(5), weights=self.beauty_dist[subset_idx], k=1)[0]
            index = random.randrange(len(self.keys[subset_idx][beauty_score]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][beauty_score][index]
        else:
            index = random.randrange(len(self.keys[subset_idx]))
            chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            res_group = f[str(res)]
            data_residual = f[f"{group_path}/residual"]
            data_lowfreq = f[f"{group_path}/lowfreq"]
            
            residual_shape = data_residual.shape
            lowfreq_shape = data_lowfreq.shape
            
            # Calculate the full-resolution crop size (8x larger)
            full_crop_size = self.crop_size * 8
            
            if self.clip_edges:
                assert lowfreq_shape[0] >= self.crop_size + 2  # Add margin for safety
                i_low = random.randint(1, lowfreq_shape[0] - self.crop_size - 1) if not self.eval_dataset else (lowfreq_shape[0] - self.crop_size) // 2
                j_low = random.randint(1, lowfreq_shape[1] - self.crop_size - 1) if not self.eval_dataset else (lowfreq_shape[1] - self.crop_size) // 2
            else:
                assert lowfreq_shape[0] >= self.crop_size
                i_low = random.randint(0, lowfreq_shape[0] - self.crop_size) if not self.eval_dataset else (lowfreq_shape[0] - self.crop_size) // 2
                j_low = random.randint(0, lowfreq_shape[1] - self.crop_size) if not self.eval_dataset else (lowfreq_shape[1] - self.crop_size) // 2
            
            # Convert to residual space (full resolution)
            i = i_low * 8
            j = j_low * 8
                
            # Handle transformations
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Load and process residual data at full resolution
            data_residual = torch.from_numpy(data_residual[i:i+full_crop_size, j:j+full_crop_size])[None]
            
            # Apply transformations to residual
            if flip:
                data_residual = torch.flip(data_residual, dims=[-1])
            if rotate_k != 0:
                data_residual = torch.rot90(data_residual, k=rotate_k, dims=[-2, -1])
            
            # Downsample residual to match latent resolution (1/8)
            data_residual = F.avg_pool2d(data_residual, kernel_size=8, stride=8)
            
            # Normalize residual
            data_residual = (data_residual - res_group.attrs['residual_mean']) / res_group.attrs['residual_std'] * self.sigma_data
            
            # Calculate corresponding indices for lowfreq (which is already at 1/8 resolution)
            i_low = i // 8
            j_low = j // 8
            h_low = self.crop_size  # Already at target size
            w_low = self.crop_size  # Already at target size
            
            # Load and process lowfreq data (already at 1/8 resolution)
            data_lowfreq = torch.from_numpy(data_lowfreq[i_low:i_low+h_low, j_low:j_low+w_low])[None]
            
            # Apply transformations to lowfreq
            if flip:
                data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
            if rotate_k != 0:
                data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
            
            # Create water mask based on lowfreq (water is typically negative elevation)
            water_mask = (data_lowfreq < 0).float() * 2 - 1
            
            # Normalize lowfreq
            data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD * self.sigma_data
            lowfreq_mean = torch.mean(data_lowfreq) / self.sigma_data
            
        # Combine data for output
        image = torch.cat([data_residual, data_lowfreq], dim=0)
        
        # Prepare conditional inputs
        cond_inputs = [lowfreq_mean.reshape([]).float()]
        if class_label is not None:
            cond_inputs += [torch.tensor(class_label)]
            
        return {'image': image.float(), 'cond_inputs': cond_inputs, 'path': group_path}

    def denormalize_residual(self, residual, resolution):
        with h5py.File(self.h5_file, 'r') as f:
            res_group = f[str(resolution)]
            return residual * res_group.attrs['residual_std'] + res_group.attrs['residual_mean']
        
    def denormalize_lowfreq(self, lowfreq, resolution=None):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        return lowfreq * LOWFREQ_STD + LOWFREQ_MEAN

class H5GANDataset(Dataset):
    def __init__(self, 
                 h5_file, 
                 crop_size, 
                 pct_land_ranges, 
                 subset_resolutions, 
                 subset_weights=None,
                 subset_class_labels=None,
                 eval_dataset=False,
                 latents_mean=None, 
                 latents_std=None, 
                 sigma_data=0.5, 
                 clip_edges=True,
                 split=None,
                 final_size=None,
                 include_climate=False,
                 memory=262144,
                 memory_freq=4):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset. Elements can be None to include everything.
            subset_resolutions (list): Resolutions to filter subsets by. Each subset will only include elevation data with the corresponding resolution. Elements be None to include all resolutions.
            subset_weights (list): Weights for each subset, determining the relative probability of sampling from that subset. Default is None, which results in uniform sampling.
            subset_class_labels (list): Class labels for each subset, determining the class of each subset. Defaults to None, which results in no class labels being returned.
            eval_dataset (bool, optional): Whether the dataset should be transformed deterministically. Defaults to False.
            latents_mean (list, optional): Mean values for normalizing latents. Defaults to None.
            latents_std (list, optional): Standard deviation values for normalizing latents. Defaults to None.
            sigma_data (float, optional): Data standard deviation. Defaults to 0.5.
            clip_edges (bool, optional): Whether to clip edges when cropping. Defaults to True.
            split (str, optional): Split to use. Defaults to None (all splits).
            final_size (int, optional): Size to interpolate output images to. If None, no interpolation is performed.
            include_climate (bool, optional): Whether to include climate data
        """
        self.base_dataset = H5LatentsDataset(h5_file, crop_size, pct_land_ranges, subset_resolutions, 
                                             subset_weights, subset_class_labels, eval_dataset, 
                                             latents_mean, latents_std, sigma_data, clip_edges, split, include_climate)
        self.memory = deque()
        self.memory_size = memory
        self.memory_freq = memory_freq
        self.final_size = final_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if idx % (self.memory_freq + 1) != 0:
            memory_idx = (idx % (self.memory_freq + 1)) * (self.memory_size // self.memory_freq)
            if len(self.memory) > memory_idx:
                return self.memory[memory_idx]
            
        out = self.base_dataset[idx]
        img = out['image'][4:5]
        if self.final_size is not None:
            img = torch.nn.functional.interpolate(
                img[None], 
                size=(self.final_size, self.final_size), 
                mode='nearest'
            )[0]
            
        self.memory.append({'image': img})
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        
        return {'image': img}

class CIFAR10Dataset(Dataset):
    """Dataset wrapper for CIFAR10 that returns images in the same format as H5LatentsDataset.
    
    This wrapper takes a CIFAR10 dataset and returns just the images formatted as a dict with
    'image' and 'cond_inputs' keys to match the H5LatentsDataset format. Images are normalized
    per-channel to have mean 0 and standard deviation 1 using CIFAR10's known statistics.
    """
    def __init__(self, root, train=True, download=True):
        # CIFAR10 per-channel mean and std values
        self.means = torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
        self.stds = torch.tensor([0.2470, 0.2435, 0.2616])[:, None, None]
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=T.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        # Normalize each channel
        img = (img - self.means) / self.stds
        # Convert to grayscale by averaging channels
        img = img.mean(dim=0, keepdim=True)
        return {'image': img}


class LongDataset(Dataset):
    def __init__(self, base_dataset, length=10 ** 12, shuffle=True):
        self.base_dataset = base_dataset
        self.length = length
        self.shuffle = shuffle
        self.order = torch.randperm(len(self.base_dataset)) if shuffle else torch.arange(len(self.base_dataset))

    def __len__(self):
        return self.length

    def base_length(self, batch_size):
        return math.ceil(len(self.base_dataset) / batch_size)

    def __getitem__(self, index):
        if index % len(self.base_dataset) == 0 and self.shuffle:
            self.order = torch.randperm(len(self.base_dataset))
        return self.base_dataset[self.order[index % len(self.base_dataset)]]

def stacking_collate_fn(batch):
    batch_size = len(batch[0])
    stacked_tensors = []
    for i in range(batch_size):
        if isinstance(batch[0][i], torch.Tensor):
            stacked_tensor = torch.stack([item[i] for item in batch])
            stacked_tensors.append(stacked_tensor)
        else:
            stacked_tensors.append(stacking_collate_fn([item[i] for item in batch]))
    return tuple(stacked_tensors)

class GANDataset(Dataset):
    """Dataset for GAN training that outputs the conditional image from H5LatentsDataset without landcover map.
    
    This dataset reads from an HDF5 file containing latent codes, low frequency components and climate data,
    and outputs the conditional image tensor containing the low frequency and climate data.
    """
    def __init__(self, h5_file, crop_size, pct_land_ranges=None, subset_resolutions=None, 
                 subset_weights=None, subset_class_labels=None, split=None, eval_dataset=False,
                 clip_edges=False, use_climate=True):
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.eval_dataset = eval_dataset
        self.clip_edges = clip_edges
        self.subset_weights = subset_weights or [1]
        self.subset_class_labels = subset_class_labels
        self.use_climate = use_climate
        
        if pct_land_ranges is None:
            pct_land_ranges = [[0, 1]]
        if subset_resolutions is None:
            subset_resolutions = [90]
            
        self.keys = [set() for _ in range(len(subset_resolutions))]
        
        with h5py.File(self.h5_file, 'r') as f:
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                    
                if str(res) not in f:
                    continue
                    
                res_group = f[str(res)]
                for chunk_id in res_group:
                    chunk_group = res_group[chunk_id]
                    for subchunk_id in chunk_group:
                        subchunk_group = chunk_group[subchunk_id]
                        if 'latent' not in subchunk_group:
                            continue
                            
                        dset = subchunk_group['latent']
                        pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                        split_valid = split is None or dset.attrs['split'] == split
    
                        if pct_land_valid and split_valid:
                            self.keys[i].add((chunk_id, res, subchunk_id))
        self.keys = [list(keys) for keys in self.keys]

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, idx):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            data_lowfreq = f[f"{group_path}/lowfreq"]
            data_climate = f[f"{group_path}/climate"] if self.use_climate and 'climate' in f[group_path] else None
            
            shape = data_lowfreq.shape
            
            if not self.eval_dataset:
                if self.clip_edges:
                    i = random.randint(1, shape[0] - self.crop_size - 1)
                    j = random.randint(1, shape[1] - self.crop_size - 1)
                else:
                    i = random.randint(0, shape[0] - self.crop_size)
                    j = random.randint(0, shape[1] - self.crop_size)
            else:
                i = (shape[0] - self.crop_size) // 2
                j = (shape[1] - self.crop_size) // 2
                
            h = w = self.crop_size
            
            data_lowfreq = torch.from_numpy(data_lowfreq[i:i+h, j:j+w])[None]
            data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD
            
            if self.use_climate and data_climate is not None:
                data_mean_temp = data_climate[0, i:i+h, j:j+w]
                data_temp_seasonality = data_climate[3, i:i+h, j:j+w]
                data_annual_precip = data_climate[11, i:i+h, j:j+w]
                data_precip_seasonality = data_climate[14, i:i+h, j:j+w]
                
                # Create nan mask (True where any climate variable is nan)
                nan_mask = (np.isnan(data_mean_temp) | 
                          np.isnan(data_temp_seasonality) | 
                          np.isnan(data_annual_precip) | 
                          np.isnan(data_precip_seasonality))
                
                # Normalize climate data by subtracting mean and dividing by std
                climate_mean = f[str(res)].attrs['climate_mean']
                climate_std = f[str(res)].attrs['climate_std']
                
                data_mean_temp = torch.from_numpy(data_mean_temp - climate_mean[0]) / climate_std[0]
                data_temp_seasonality = torch.from_numpy(data_temp_seasonality - climate_mean[3]) / climate_std[3]
                data_annual_precip = torch.from_numpy(data_annual_precip - climate_mean[11]) / climate_std[11]
                data_precip_seasonality = torch.from_numpy(data_precip_seasonality - climate_mean[14]) / climate_std[14]
                
                climate_means = [
                    torch.nanmean(data_mean_temp),
                    torch.nanmean(data_temp_seasonality),
                    torch.nanmean(data_annual_precip),
                    torch.nanmean(data_precip_seasonality)
                ]
                climate_means = torch.as_tensor([torch.nan_to_num(mean, nan=0.0) for mean in climate_means])
            
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Apply transformations to all channels
            if flip:
                data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
            if rotate_k != 0:
                data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
                
            return {'image': data_lowfreq, 'additional_vars': climate_means}

class FileGANDataset(Dataset):
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
    def __init__(self, dataset_names, crop_size=(32, 32), resize_size=(32, 32), 
                 filter_threshold=None, filter_pct=0.5, max_filter_tries=50):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.filter_threshold = filter_threshold
        self.filter_pct = float(filter_pct)
        self.max_filter_tries = int(max_filter_tries)
        
        data_arrays = []
        for tif_file in dataset_names:
            with rasterio.open(tif_file) as src:
                # Get the data bounds
                bounds = src.bounds
                
                # Calculate row indices for -60 to 60 degrees latitude
                height = src.height
                lat_res = (bounds.top - bounds.bottom) / height
                start_row = int((bounds.top - 60) / lat_res)
                end_row = int((bounds.top + 60) / lat_res)
                
                # Read the data within latitude bounds
                data = src.read(1)[start_row:end_row, :]
                if data.dtype == np.int16:
                    data = data.astype(np.float32)
                    data[data == -32768] = np.nan
                else:
                    data[np.abs(data) > 1e6] = np.nan
                    
                # Normalize the data
                u, s = np.nanmean(data), np.nanstd(data)
                print(tif_file, u, s)
                data[np.isnan(data)] = u
                data = (data - u) / s
                
                # Get filename without extension and path for dictionary key
                data_arrays.append(data)
            
        # Get data arrays in requested order and stack them
        self.data = np.stack(data_arrays, axis=0)  # Shape: (n_datasets, height, width)
        
        # Store valid dimensions for random cropping
        self.height, self.width = self.data.shape[1:]
        
    def __len__(self):
        """Return a fixed length (can be adjusted based on needs)"""
        return 10000  # Arbitrary number of samples per epoch
        
    def __getitem__(self, idx):
        """
        Get a random crop from the stacked datasets.
        
        Args:
            idx (int): Index (not used since we're returning random crops)
            
        Returns:
            torch.Tensor: Tensor of shape (n_datasets, crop_height, crop_width)
        """
        # Calculate valid ranges for the top-left corner of the crop
        max_h = self.height - self.crop_size[0]
        max_w = self.width - self.crop_size[1]
        
        # Choose target side of threshold if requested
        if self.filter_threshold is not None:
            want_above = (np.random.random() < self.filter_pct)
            h_start = w_start = 0
            found = False
            for _ in range(self.max_filter_tries):
                h_start = np.random.randint(0, max_h + 1)
                w_start = np.random.randint(0, max_w + 1)
                first_mean = self.data[0, 
                                       h_start:h_start + self.crop_size[0],
                                       w_start:w_start + self.crop_size[1]].mean()
                if (first_mean > self.filter_threshold) == want_above:
                    found = True
                    break
            if not found:
                # fallback to a random crop if condition not found
                h_start = np.random.randint(0, max_h + 1)
                w_start = np.random.randint(0, max_w + 1)
        else:
            # Generate random crop coordinates
            h_start = np.random.randint(0, max_h + 1)
            w_start = np.random.randint(0, max_w + 1)
        
        # Extract the crop
        crop = self.data[:, 
                        h_start:h_start + self.crop_size[0],
                        w_start:w_start + self.crop_size[1]]
        
        crop = torch.from_numpy(crop).float()
        # Randomly flip horizontally and vertically
        if np.random.random() < 0.5:
            crop = torch.flip(crop, dims=[-1])
            
        # Randomly rotate by multiples of 90 degrees
        k = np.random.randint(4)  # Number of 90 degree rotations
        if k > 0:
            crop = torch.rot90(crop, k=k, dims=(-2, -1))
        
        
        if self.resize_size is not None:
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=self.resize_size,
                mode='area',
                antialias=False
            ).squeeze(0)
            return {'image': crop}
        
        return {'image': torch.from_numpy(crop).float()}