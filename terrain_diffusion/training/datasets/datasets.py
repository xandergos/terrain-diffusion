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
        LOWFREQ_MEAN = -2128
        LOWFREQ_STD = 2353
        
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
            
            # Calculate scaled indices for 1/8 resolution data
            i_low = i // 8
            j_low = j // 8
            h_low = h // 8
            w_low = w // 8
            
            # Load residual data (always available, full res)
            residual_data = torch.from_numpy(dset[i:i+h, j:j+w])[None]
            residual_data = (residual_data - res_group.attrs['residual_mean']) / res_group.attrs['residual_std']
            
            # Load lowres data (always available, 1/8 res)
            lowres_data = torch.from_numpy(f[f"{group_path}/lowfreq"][i_low:i_low+h_low, j_low:j_low+w_low])[None]
            lowres_data = F.interpolate(lowres_data[None], size=(h, w), mode='nearest')[0]
            lowres_data = (lowres_data - LOWFREQ_MEAN) / LOWFREQ_STD
            
            # Load and upsample climate data with backup
            try:
                climate_data = torch.from_numpy(f[f"{group_path}/climate"][:, i_low:i_low+h_low, j_low:j_low+w_low])
                climate_data = climate_data[self.climate_channels]
                climate_data = (climate_data - res_group.attrs['climate_mean'][self.climate_channels, None, None]) /\
                    res_group.attrs['climate_std'][self.climate_channels, None, None]
                climate_data = F.interpolate(climate_data[None], size=(h, w), mode='nearest')[0]
                # Create climate mask and handle NaNs
                climate_mask = ~torch.isnan(climate_data[0:1])
                climate_data = torch.nan_to_num(climate_data, 0.0)
            except KeyError:
                climate_data = torch.zeros((4, h, w))
                climate_mask = torch.zeros((1, h, w))
            
            # Load and one-hot encode landcover data with backup
            landcover_classes = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126, 200]
            
            try:
                landcover_data = torch.from_numpy(f[f"{group_path}/landcover"][i:i+h, j:j+w]).long()
            except KeyError:
                landcover_data = torch.full((h, w), 200, dtype=torch.long)
                
            # Create lookup table for class indices
            max_class = landcover_classes[-1]
            lookup = torch.full((max_class + 1,), len(landcover_classes) - 1, dtype=torch.long)
            for idx, class_val in enumerate(landcover_classes):
                lookup[class_val] = idx
            landcover_indices = lookup[landcover_data]
            
            landcover_onehot = F.one_hot(landcover_indices, num_classes=len(landcover_classes))
            landcover_onehot = landcover_onehot.permute(2, 0, 1)  # [C, H, W] format
            
            # Load watercover data with backup
            try:
                water_data = torch.from_numpy(f[f"{group_path}/watercover"][i:i+h, j:j+w])[None] / 100
            except KeyError:
                water_data = torch.zeros((1, h, w))
            
            # Concatenate all channels
            data = torch.cat([
                residual_data,      # 1 channel
                lowres_data,        # 1 channel
                climate_data,       # 4 channels
                climate_mask,       # 1 channel
                landcover_onehot,   # 23 channels
                water_data,         # 1 channel
            ], dim=0).float()
            
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
        LOWFREQ_MEAN = -2128
        LOWFREQ_STD = 2353
        
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
                edge_margin = 1 if self.clip_edges else 0
                i = random.randint(0 + edge_margin, max_i - edge_margin) * 8
                j = random.randint(0 + edge_margin, max_j - edge_margin) * 8
            else:
                # Center crop, ensuring it's aligned to 8-pixel boundaries
                i = ((data_shape[-2] - self.crop_size) // 16) * 8
                j = ((data_shape[-1] - self.crop_size) // 16) * 8
            h, w = self.crop_size, self.crop_size
            
            # Calculate scaled indices for 1/8 resolution data
            i_low = i // 8
            j_low = j // 8
            h_low = h // 8
            w_low = w // 8
            
            # Load residual data (always available, full res)
            residual_data = torch.from_numpy(dset[i:i+h, j:j+w])[None]
            residual_data = (residual_data - res_group.attrs['residual_mean']) / res_group.attrs['residual_std']
            
            # Load lowres data (always available, 1/8 res)
            lowres_data = torch.from_numpy(f[f"{group_path}/lowfreq"][i_low:i_low+h_low, j_low:j_low+w_low])[None]
            lowres_data = F.interpolate(lowres_data[None], size=(h, w), mode='nearest')[0]
            lowres_data = (lowres_data - LOWFREQ_MEAN) / LOWFREQ_STD
            
            # Load and upsample climate data with backup
            try:
                climate_data = torch.from_numpy(f[f"{group_path}/climate"][:, i_low:i_low+h_low, j_low:j_low+w_low])
                climate_data = climate_data[self.climate_channels]
                climate_data = (climate_data - res_group.attrs['climate_mean'][self.climate_channels, None, None]) /\
                    res_group.attrs['climate_std'][self.climate_channels, None, None]
                climate_data = F.interpolate(climate_data[None], size=(h, w), mode='nearest')[0]
                # Create climate mask and handle NaNs
                climate_mask = ~torch.isnan(climate_data[0:1])
                climate_data = torch.nan_to_num(climate_data, 0.0)
            except KeyError:
                climate_data = torch.zeros((4, h, w))
                climate_mask = torch.zeros((1, h, w))
            
            # Load and one-hot encode landcover data with backup
            landcover_classes = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126, 200]
            
            try:
                landcover_data = torch.from_numpy(f[f"{group_path}/landcover"][i:i+h, j:j+w]).long()
            except KeyError:
                landcover_data = torch.full((h, w), 200, dtype=torch.long)
                
            # Create lookup table for class indices
            max_class = landcover_classes[-1]
            lookup = torch.full((max_class + 1,), len(landcover_classes) - 1, dtype=torch.long)
            for idx, class_val in enumerate(landcover_classes):
                lookup[class_val] = idx
            landcover_indices = lookup[landcover_data]
            
            landcover_onehot = F.one_hot(landcover_indices, num_classes=len(landcover_classes))
            landcover_onehot = landcover_onehot.permute(2, 0, 1)  # [C, H, W] format
            landcover_onehot = landcover_onehot * np.sqrt(len(landcover_classes))
            
            # Load watercover data with backup
            try:
                water_data = torch.from_numpy(f[f"{group_path}/watercover"][i:i+h, j:j+w])[None] / 100
            except KeyError:
                water_data = torch.zeros((1, h, w))
            
            # Concatenate all channels
            data = torch.cat([
                residual_data,      # 1 channel
                lowres_data,        # 1 channel
                climate_data,       # 4 channels
                climate_mask,       # 1 channel
                landcover_onehot,   # 23 channels
                water_data,         # 1 channel
            ], dim=0).float()
            
        # Apply transforms
        transform_idx = random.randrange(8) if not self.eval_dataset else 0
        flip = (transform_idx // 4) == 1
        rotate_k = transform_idx % 4
            
        if flip:
            data = torch.flip(data, dims=[-1])
        if rotate_k != 0:
            data = torch.rot90(data, k=rotate_k, dims=[-2, -1])
            
        data = data * self.sigma_data
        if class_label is not None:
            return {'image': data, 'cond_inputs': [torch.tensor(class_label)]}
        else:
            return {'image': data}

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
                 include_climate=True):
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
            include_climate (bool, optional): Whether to include climate data
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
        self.include_climate = include_climate
        
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
        LOWFREQ_MEAN = -2128
        LOWFREQ_STD = 2353
        
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            res_group = f[str(res)]
            data_latent = f[f"{group_path}/latent"]
            data_lowfreq = f[f"{group_path}/lowfreq"]
            
            shape = data_latent.shape
            assert data_lowfreq.shape == shape[2:]
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
                
            data_latent = torch.from_numpy(data_latent[transform_idx, :, i:i+h, j:j+w])
            data_lowfreq = torch.from_numpy(data_lowfreq[li:li+lh, lj:lj+lw])[None]
            
            lowfreq_mean = torch.mean(data_lowfreq)
            
            # Retrieve climate data for channels 0, 3, 11, 14 (temp, temp seasonality, precip, precip seasonality)
            if not self.include_climate and f"{group_path}/climate" in f:
                climate_data = f[f"{group_path}/climate"][[0, 3, 11, 14]][:, li:li+lh, lj:lj+lw]
                climate_data = (climate_data - res_group.attrs['climate_mean'][[0, 3, 11, 14], None, None]) / res_group.attrs['climate_std'][[0, 3, 11, 14], None, None] * self.sigma_data
                climate_data = torch.from_numpy(climate_data).float()
                any_nan_climate = torch.isnan(climate_data).all(dim=(-2, -1)).any().item()
                if any_nan_climate:
                    climate_nanmean = torch.zeros(4)
                else:
                    climate_nanmean = torch.nanmean(climate_data, dim=(-2, -1))
            else:
                climate_data = torch.zeros(4, lh, lw)
                climate_nanmean = torch.zeros(4)
                any_nan_climate = True
            
        data_lowfreq = data_lowfreq.float()
        data_lowfreq = (data_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD * self.sigma_data
            
        assert data_latent.shape[-1] > 0, f"Latent crop is empty. i: {i}, j: {j}, h: {h}, w: {w}"
            
        # Apply transforms to lowfreq to match latent
        if flip:
            data_lowfreq = torch.flip(data_lowfreq, dims=[-1])
        if rotate_k != 0:
            data_lowfreq = torch.rot90(data_lowfreq, k=rotate_k, dims=[-2, -1])
            
        latent_channels = data_latent.shape[0]
        means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
        sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
        sampled_latent = (sampled_latent - self.latents_mean) / self.latents_std * self.sigma_data
        
        # Only include sampled_latent in img tensor instead of concatenating with other data
        cond_inputs = [lowfreq_mean.reshape([]).float()]
        if class_label is not None:
            cond_inputs += [torch.tensor(class_label)]
        if self.include_climate:
            cond_inputs += [
                       climate_nanmean[0].float().reshape([]),
                       climate_nanmean[1].float().reshape([]), 
                       climate_nanmean[2].float().reshape([]),
                       climate_nanmean[3].float().reshape([]),
                       torch.tensor(1 if any_nan_climate else 0, dtype=torch.int64).reshape([])
                       ]
            
        return {'image': sampled_latent.float(), 'cond_inputs': cond_inputs, 'path': group_path}

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
        LOWFREQ_MEAN = -2128
        LOWFREQ_STD = 2353
        
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

class ETOPODataset(Dataset):
    def __init__(self, folder, size, mean, std, crop_size=None, eval_dataset=False, blur_sigma=None):
        self.folder = folder
        self.size = size
        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        self.eval_dataset = eval_dataset
        self.blur_sigma = blur_sigma

        files = os.listdir(folder)
        self.files = [os.path.join(folder, file) for file in files if file.endswith('.tif') or file.endswith('.tiff')]

    def __len__(self):
        return len(self.files)

    @lru_cache(maxsize=300)
    def read_image(self, file):
        img = Image.open(file)
        img = img.resize((self.size, self.size), resample=Image.Resampling.NEAREST)
        img = np.array(img)
        img = (img - self.mean) / self.std
        img = img.astype(np.float32)
        img = torch.from_numpy(img)[None]
        if self.blur_sigma is not None:
            img = TF.gaussian_blur(img, kernel_size=(1+2*self.blur_sigma, 1+2*self.blur_sigma), sigma=(self.blur_sigma, self.blur_sigma))
        
        return img
    
    def __getitem__(self, index):
        img = self.read_image(self.files[index])
        
        if self.crop_size is not None:
            i = random.randint(0, img.shape[1] - self.crop_size)
            j = random.randint(0, img.shape[2] - self.crop_size)
            img = img[:, i:i+self.crop_size, j:j+self.crop_size]
            
        transform_idx = random.randrange(8) if not self.eval_dataset else 0
        flip = (transform_idx // 4) == 1
        rotate_k = transform_idx % 4
        if flip:
            img = torch.flip(img, dims=[-1])
        if rotate_k != 0:
            img = torch.rot90(img, k=rotate_k, dims=[-2, -1])
            
        return {'image': img}
        
