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
from terrain_diffusion.data.laplacian_encoder import LaplacianPyramidEncoder
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
                 split=None):
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
                for key in sorted(list(f.keys())):
                    dset = f[key]
                    pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                    resolution_valid = res is None or str(dset.attrs['resolution']) == str(res)
                    split_valid = split is None or dset.attrs['split'] == split
    
                    if pct_land_valid and resolution_valid and split_valid:
                        self.keys[i].add((dset.attrs['filename'], dset.attrs['resolution'], dset.attrs['chunk_id']))
        self.keys = [list(keys) for keys in self.keys]

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, index):
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        file, res, chunk_id = self.keys[subset_idx][index]
        key = '$'.join([file, f'{res}m', chunk_id, 'highfreq'])
        
                
        with h5py.File(self.h5_file, 'r') as f:
            data_shape = f[key].shape
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
            
            data = torch.from_numpy(f[key][:, i:i+h, j:j+w])
            
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
                 split=None):
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
                for key in sorted(list(f.keys())):
                    dset = f[key]
                    pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                    resolution_valid = res is None or str(dset.attrs['resolution']) == str(res)
                    split_valid = split is None or dset.attrs['split'] == split

                    if pct_land_valid and resolution_valid and split_valid:
                        self.keys[i].add((dset.attrs['filename'], dset.attrs['resolution'], dset.attrs['chunk_id']))
        self.keys = [list(keys) for keys in self.keys]

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, index):
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        file, res, chunk_id = self.keys[subset_idx][index]
        base_key = '$'.join([file, f'{res}m', chunk_id])
        key_latent = base_key + '$latent'
        key_highfreq = base_key + '$highfreq'
        
        with h5py.File(self.h5_file, 'r') as f:
            latent_shape = f[key_latent].shape
            highfreq_shape = f[key_highfreq].shape
            
        upscale_factor = highfreq_shape[1] // latent_shape[2]
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
        
        # Adjust highfreq crop for flips and rotations. Note that we are inverting the transformation so we do it in reverse.
        for _ in range(rotate_k):
            li, lj = lj, highfreq_shape[2] - li - lh
        if flip:
            lj = highfreq_shape[2] - lj - lw
            
        with h5py.File(self.h5_file, 'r') as f:
            data_latent = torch.from_numpy(f[key_latent][transform_idx, :, i:i+h, j:j+w])
            data_highfreq = torch.from_numpy(f[key_highfreq][:, li:li+lh, lj:lj+lw])
            
        assert data_latent.shape[-1] > 0, f"Latent crop is empty. i: {i}, j: {j}, h: {h}, w: {w}"
            
        # Apply transforms to highfreq to match latent
        if flip:
            data_highfreq = torch.flip(data_highfreq, dims=[-1])
        if rotate_k != 0:
            data_highfreq = torch.rot90(data_highfreq, k=rotate_k, dims=[-2, -1])
            
        latent_channels = data_latent.shape[0]
        means, logvars = data_latent[:latent_channels//2], data_latent[latent_channels//2:]
        sampled_latent = torch.randn_like(means) * (logvars * 0.5).exp() + means
        upsampled_latent = torch.nn.functional.interpolate(sampled_latent[None], (self.crop_size, self.crop_size), mode='nearest')[0]
        
        img = data_highfreq
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
                 split=None):
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
                for key in sorted(list(f.keys())):
                    dset = f[key]
                    pct_land_valid = pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1]
                    resolution_valid = res is None or str(dset.attrs['resolution']) == str(res)
                    split_valid = split is None or dset.attrs['split'] == split

                    if pct_land_valid and resolution_valid and split_valid:
                        self.keys[i].add((dset.attrs['filename'], dset.attrs['resolution'], dset.attrs['chunk_id']))
        self.keys = [list(keys) for keys in self.keys]

    def __len__(self):
        return max(len(keys) for keys in self.keys)

    def __getitem__(self, idx):
        # Draw a random subset based on subset weights
        subset_idx = random.choices(range(len(self.subset_weights)), weights=self.subset_weights, k=1)[0]
        index = random.randrange(len(self.keys[subset_idx]))
        class_label = self.subset_class_labels[subset_idx] if self.subset_class_labels is not None else None
        
        file, res, chunk_id = self.keys[subset_idx][index]
        base_key = '$'.join([file, f'{res}m', chunk_id])
        key_latent = base_key + '$latent'
        key_lowfreq = base_key + '$lowfreq'
        
        with h5py.File(self.h5_file, 'r') as f:
            shape = f[key_latent].shape
        
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
            
        with h5py.File(self.h5_file, 'r') as f:
            data_latent = torch.from_numpy(f[key_latent][transform_idx, :, i:i+h, j:j+w])
            data_lowfreq = torch.from_numpy(f[key_lowfreq][:, li:li+lh, lj:lj+lw])
            
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
        cond_img_sigma = torch.exp(torch.randn([1, 1, 1], dtype=torch.float32) * 2.0 + 1)
        cond_t = torch.atan(cond_img_sigma / self.sigma_data)
        cond_img = torch.cos(cond_t) * data_lowfreq + torch.sin(cond_t) * torch.randn_like(data_lowfreq) * self.sigma_data
        cond_img = cond_img / self.sigma_data
        
        if class_label is not None:
            return {'image': img, 'cond_img': cond_img, 'cond_inputs': [torch.tensor(class_label), cond_t.squeeze()]}
        else:
            return {'image': img, 'cond_img': cond_img, 'cond_inputs': [cond_t.squeeze()]}


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
        
