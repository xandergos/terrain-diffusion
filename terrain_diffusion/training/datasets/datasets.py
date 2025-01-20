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
                 sigma_data=0.5,
                 use_watercover=False,
                 require_watercover=False):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing the dataset.
            crop_size (int): Size of the random crop to extract from each image.
            pct_land_ranges (list): Ranges of acceptable pct_land values [min, max], for each subset. Elements can be None to include everything.
            subset_resolutions (list): Resolutions to filter subsets by. Each subset will only include elevation data with the corresponding resolution. Elements be None to include all resolutions.
            subset_weights (list): Weights for each subset, determining the relative probability of sampling from that subset. Default is None, which results in uniform sampling.
            subset_class_labels (list): Class labels for each subset, determining the class of each subset. Defaults to None, which results in no class labels being returned.
            eval_dataset (bool, optional): Whether the dataset should be transformed deterministically. Defaults to False.
            split (str, optional): Split to use. Defaults to None (all splits).
            sigma_data (float, optional): Data standard deviation. Defaults to 0.5.
            use_watercover (bool, optional): Whether to use watercover data. Defaults to False.
            require_watercover (bool, optional): Whether to ignore samples without watercover data. Defaults to False, in which case missing watercover is assumed to be all 0.
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
        self.sigma_data = sigma_data
        self.use_watercover = use_watercover
        self.require_watercover = require_watercover
        
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
                            
                        if self.require_watercover and 'watercover' not in subchunk_group:
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
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r') as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            dset = f[f"{group_path}/residual"]
            data_shape = dset.shape
            dset_mean = dset.attrs['residual_mean']
            dset_std = dset.attrs['residual_std']
            
            if not self.eval_dataset:
                i = random.randint(0, data_shape[-2] - self.crop_size)
                j = random.randint(0, data_shape[-1] - self.crop_size)
                h, w = self.crop_size, self.crop_size
            else:
                i, j = (data_shape[-2] - self.crop_size) // 2, (data_shape[-1] - self.crop_size) // 2
                h, w = self.crop_size, self.crop_size
                
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Load elevation data
            elev_data = torch.from_numpy(dset[i:i+h, j:j+w])[None]
            elev_data = (elev_data - dset_mean) / dset_std
            
            # Load watercover if requested
            if self.use_watercover:
                try:
                    water_data = torch.from_numpy(f[f"{group_path}/watercover"][i:i+h, j:j+w])[None] / 100 * self.sigma_data
                except KeyError:
                    water_data = torch.zeros_like(elev_data)
                data = torch.cat([elev_data, water_data], dim=0)
            else:
                data = elev_data
                
        assert data.shape[-1] > 0, f"Crop is empty. i: {i}, j: {j}, h: {h}, w: {w}, image shape: {data_shape}"
            
        # Apply transforms
        if flip:
            data = torch.flip(data, dims=[-1])
        if rotate_k != 0:
            data = torch.rot90(data, k=rotate_k, dims=[-2, -1])
            
        if class_label is not None:
            return {'image': data, 'cond_inputs': [torch.tensor(class_label)]}
        else:
            return {'image': data}
    
class H5SuperresTerrainDataset(Dataset):
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
                 split=None,
                 use_watercover=False,
                 require_watercover=False):
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
            use_watercover (bool, optional): Whether to use watercover data. Defaults to False.
            require_watercover (bool, optional): Whether to ignore samples without watercover data. Defaults to False, in which case missing watercover is assumed to be all 0.
        """
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
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        chunk_id, res, subchunk_id = self.keys[subset_idx][index]
        with h5py.File(self.h5_file, 'r', rdcc_nbytes=16*1024**2) as f:
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            latent_dset = f[f"{group_path}/latent"]
            residual_dset = f[f"{group_path}/residual"]
            
            latent_shape = latent_dset.shape
            residual_shape = residual_dset.shape
            
            upscale_factor = residual_shape[-2] // latent_shape[2]
            latent_crop_size = self.crop_size // upscale_factor
            
            if not self.eval_dataset:
                if self.clip_edges:
                    i = random.randint(1, latent_shape[2] - latent_crop_size - 1)
                    j = random.randint(1, latent_shape[3] - latent_crop_size - 1)
                else:
                    i = random.randint(0, latent_shape[2] - latent_crop_size)
                    j = random.randint(0, latent_shape[3] - latent_crop_size)
            else:
                i = (latent_shape[2] - latent_crop_size) // 2
                j = (latent_shape[3] - latent_crop_size) // 2
                
            h = w = latent_crop_size
            li, lj, lh, lw = i * upscale_factor, j * upscale_factor, h * upscale_factor, w * upscale_factor
                
            transform_idx = random.randrange(8) if not self.eval_dataset else 0
            flip = (transform_idx // 4) == 1
            rotate_k = transform_idx % 4
            
            # Adjust residual crop for flips and rotations. Note that we are inverting the transformation so we do it in reverse.
            for _ in range(rotate_k):
                li, lj = lj, residual_shape[-1] - li - lh
            if flip:
                lj = residual_shape[-1] - lj - lw
                
            data_latent = torch.from_numpy(latent_dset[transform_idx, :, i:i+h, j:j+w])
            data_residual = torch.from_numpy(residual_dset[li:li+lh, lj:lj+lw])[None]
            residual_std = f[str(res)].attrs['residual_std']
            residual_mean = f[str(res)].attrs['residual_mean']
            
            # Load watercover if requested
            if self.use_watercover:
                try:
                    water_data = torch.from_numpy(f[f"{group_path}/watercover"][li:li+lh, lj:lj+lw])[None] / 100 * self.sigma_data
                except KeyError:
                    water_data = torch.zeros_like(data_residual)
                data_residual = torch.cat([data_residual, water_data], dim=0)
            
        data_residual = (data_residual - residual_mean) / residual_std * self.sigma_data
            
        assert data_latent.shape[-1] > 0, f"Latent crop is empty. i: {i}, j: {j}, h: {h}, w: {w}"
            
        # Apply transforms to residual to match latent
        if flip:
            data_residual = torch.flip(data_residual, dims=[-1])
        if rotate_k != 0:
            data_residual = torch.rot90(data_residual, k=rotate_k, dims=[-2, -1])
            
        latent_channels = data_latent.shape[0]
        means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
        sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
        upsampled_latent = torch.nn.functional.interpolate(sampled_latent[None], (self.crop_size, self.crop_size), mode='nearest')[0]
        
        img = data_residual
        cond_img = upsampled_latent
        
        if class_label is not None:
            return {'image': img, 'cond_img': cond_img, 'cond_inputs': [torch.tensor(class_label)]}
        else:
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
                 cond_p_mean=1,
                 cond_p_std=2,
                 use_landcover=True,
                 use_climate=True,
                 landcover_dropout_pct=0.1):
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
            use_landcover (bool, optional): Whether to use landcover data. Defaults to True.
            use_climate (bool, optional): Whether to use climate data. Defaults to True.
            landcover_dropout_pct (float, optional): Percentage of landcover data to drop out. Defaults to 0.1.
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
        self.cond_p_mean = cond_p_mean
        self.cond_p_std = cond_p_std
        self.use_landcover = use_landcover
        self.use_climate = use_climate
        self.landcover_dropout_pct = landcover_dropout_pct
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
            print(chunk_id, subchunk_id)
            group_path = f"{res}/{chunk_id}/{subchunk_id}"
            data_latent = f[f"{group_path}/latent"]
            data_lowfreq = f[f"{group_path}/lowfreq"]
            data_landcover = f[f"{group_path}/landcover"] if self.use_landcover and 'landcover' in f[group_path] else None
            data_climate = f[f"{group_path}/climate"] if self.use_climate and 'climate' in f[group_path] else None
            
            shape = data_latent.shape
            assert data_lowfreq.shape == shape[2:]
        
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
            
            if self.use_landcover:
                try:
                    r = data_landcover.shape[-2] // shape[-2]
                    assert shape[-2] * r == data_landcover.shape[-2], "r is not an integer"
                    data_landcover = torch.from_numpy(data_landcover[li*r:(li+lh)*r:r, lj*r:(lj+lw)*r:r])[None]
                    if random.random() < 0.9:
                        landcover_aug = 'default'
                    else:
                        landcover_aug = 'water/land'
                    del r
                except AttributeError:
                    data_landcover = torch.full((1, self.crop_size, self.crop_size), 200, dtype=torch.int32)
                    landcover_aug = 'none'
                
                data_landcover = data_landcover.float()
                data_landcover = torch.nn.functional.interpolate(data_landcover[None], (self.crop_size, self.crop_size), mode='nearest')[0]
                data_landcover = data_landcover.int()
                
                # One-hot encode data_landcover
                landcover_classes = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                                    111, 112, 113, 114, 115, 116, 
                                    121, 122, 123, 124, 125, 126, 
                                    200]
                
                # Create one-hot encoding
                data_landcover_onehot = torch.zeros((len(landcover_classes), self.crop_size, self.crop_size), dtype=torch.float32)
                for i, cls in enumerate(landcover_classes):
                    data_landcover_onehot[i] = (data_landcover[0] == cls).float()
                
                if landcover_aug == 'default':
                    self.augment_onehot(data_landcover_onehot)
                elif landcover_aug == 'water/land':
                    # Create mask for water (class 200)
                    water_mask = (data_landcover == 200).float()
                    
                    # Set all non-water classes to 1 where there is no water
                    data_landcover_onehot[:-1] = (~water_mask.bool()).float()
                    data_landcover_onehot[-1] = (water_mask.bool()).float()
                
                # Landcover dropout
                landcover_dropout_mask = torch.rand(data_landcover_onehot.shape[-2], data_landcover_onehot.shape[-1]) < self.landcover_dropout_pct
                data_landcover_onehot[:, landcover_dropout_mask] = 1
                    
                # Perform pixelwise normalization of landcover encodings
                data_landcover_onehot_normalized = data_landcover_onehot / torch.sqrt(data_landcover_onehot.mean(dim=0, keepdim=True) + 1e-8)
                data_landcover_onehot = data_landcover_onehot_normalized.float()
            
            climate_means = []
            if self.use_climate:
                data_mean_temp = data_climate[0, li:li+lh, lj:lj+lw]
                data_temp_seasonality = data_climate[3, li:li+lh, lj:lj+lw]
                data_annual_precip = data_climate[11, li:li+lh, lj:lj+lw]
                data_precip_seasonality = data_climate[14, li:li+lh, lj:lj+lw]
                
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
                
                climate_means = [torch.nan_to_num(mean, nan=0.0) for mean in climate_means]
            
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
        
        img = torch.cat([sampled_latent, data_lowfreq], dim=0)
        lowfreq_sigma = torch.exp(torch.randn([1, 1, 1], dtype=torch.float32) * self.cond_p_std + self.cond_p_mean)
        lowfreq_t = torch.atan(lowfreq_sigma / self.sigma_data)
        lowfreq_img = torch.cos(lowfreq_t) * data_lowfreq + torch.sin(lowfreq_t) * torch.randn_like(data_lowfreq) * self.sigma_data
        lowfreq_img = lowfreq_img / self.sigma_data
        
        # Start with lowfreq data
        cond_channels = [lowfreq_img]
        
        # Add landcover data if available
        if self.use_landcover:
            cond_channels.append(data_landcover_onehot)
            
        # Stack all conditional channels
        cond_img = torch.cat(cond_channels, dim=0)
        
        cond_inputs = []
        if class_label is not None:
            cond_inputs.append(torch.tensor(class_label))
        cond_inputs.append(lowfreq_t.squeeze())
        if self.use_climate:
            cond_inputs += climate_means
            
        return {'image': img, 'cond_img': cond_img, 'cond_inputs': cond_inputs}


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
        
