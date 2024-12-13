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
    """Dataset for reading terrain data from an HDF5 file."""

    def __init__(self, h5_file, crop_size, pct_land_range, dataset_label, eval_dataset=False):
        """
        Args:
            h5_file (str): Path to the HDF5 file.
            crop_size (int): Size of the square crop in the original image.
            pct_land_range (list): Range of acceptable pct_land values [min, max].
        """
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_range = pct_land_range
        with h5py.File(self.h5_file, 'r') as f:
            self.keys = []
            for key in f.keys():
                if pct_land_range[0] <= f[key].attrs['pct_land'] <= pct_land_range[1] and (dataset_label is None or str(f[key].attrs['label']) == str(dataset_label)):
                    self.keys.append(key)

        if not eval_dataset:
            self.transform = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomChoice([
                    T.Identity(),
                    T.RandomRotation((90, 90)),
                    T.RandomRotation((180, 180)),
                    T.RandomRotation((270, 270))
                ])
            ])
        else:
            self.transform = T.Identity()
            
        self.eval_dataset = eval_dataset
        self.dummy_data = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.dummy_data is None:
            with h5py.File(self.h5_file, 'r') as f:
                self.dummy_data = torch.from_numpy(f[self.keys[index]][:1, :, :])
        if not self.eval_dataset:
            i, j, h, w = T.RandomCrop.get_params(self.dummy_data, output_size=(self.crop_size, self.crop_size))
        else:
            i, j, h, w = self.crop_size // 2, self.crop_size // 2, self.crop_size, self.crop_size
        with h5py.File(self.h5_file, 'r') as f:
            data = torch.from_numpy(f[self.keys[index]][:1, i:i+h, j:j+w])
        data = self.transform(data)
        return {'image': data}
    
class H5SuperresTerrainDataset(H5AutoencoderDataset):
    """Dataset for reading terrain data from an HDF5 file."""
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
        if subset_weights is None:
            subset_weights = [1] * len(pct_land_ranges)
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_ranges = pct_land_ranges
        self.subset_resolutions = subset_resolutions
        self.subset_weights = subset_weights
        self.subset_class_labels = subset_class_labels
        self.latents_mean = torch.tensor(latents_mean).view(-1, 1, 1) if isinstance(latents_mean, list) else torch.clone(latents_mean).view(-1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(-1, 1, 1) if isinstance(latents_std, list) else torch.clone(latents_std).view(-1, 1, 1)
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
            assert f"split:{split}" in f.attrs, f"Split '{split}' not found in dataset."
            for i, (pct_land_range, res) in enumerate(zip(pct_land_ranges, subset_resolutions)):
                if pct_land_range is None:
                    pct_land_range = [0, 1]
                for key in sorted(list(f.keys())):
                    dset = f[key]
                    if pct_land_range[0] <= dset.attrs['pct_land'] <= pct_land_range[1] and \
                    (res is None or str(dset.attrs['resolution']) == str(res)) and \
                    (split is None or dset.attrs['filename'] in f.attrs[f"split:{split}"]):
                        dset = f[key]
                        self.keys[i].add((dset.attrs['filename'], dset.attrs['resolution'], dset.attrs['chunk_id']))
        self.keys = [list(keys) for keys in self.keys]
        
        self.dummy_data_latent = None

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
        
        if self.dummy_data_latent is None:
            with h5py.File(self.h5_file, 'r') as f:
                self.dummy_data_latent = torch.from_numpy(f[key_latent][0, :1, :, :])
                self.dummy_data_highfreq = torch.from_numpy(f[key_highfreq][:1, :, :])
                
        upscale_factor = self.dummy_data_highfreq.shape[1] // self.dummy_data_latent.shape[1]
        latent_crop_size = self.crop_size // upscale_factor
        if not self.eval_dataset:
            if self.clip_edges:
                i, j, h, w = T.RandomCrop.get_params(self.dummy_data_latent[:, 1:-1, 1:-1], output_size=(latent_crop_size, latent_crop_size))
                i, j = i + 1, j + 1
            else:
                i, j, h, w = T.RandomCrop.get_params(self.dummy_data_latent, output_size=(latent_crop_size, latent_crop_size))
        else:
            i, j = (self.dummy_data_latent.shape[-2] - latent_crop_size) // 2, (self.dummy_data_latent.shape[-1] - latent_crop_size) // 2
            h, w = latent_crop_size, latent_crop_size
            
        li, lj, lh, lw = i * upscale_factor, j * upscale_factor, h * upscale_factor, w * upscale_factor
            
        transform_idx = random.randrange(8) if not self.eval_dataset else 0
        flip = (transform_idx // 4) == 1
        rotate_k = transform_idx % 4
        
        # Adjust highfreq crop for flips and rotations. Note that we are inverting the transformation so we do it in reverse.
        for _ in range(rotate_k):
            li, lj = lj, self.dummy_data_highfreq.shape[-1] - li - lh
        if flip:
            lj = self.dummy_data_highfreq.shape[-1] - lj - lw
            
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
        sampled_latent = (sampled_latent - self.latents_mean) / self.latents_std
        upsampled_latent = torch.nn.functional.interpolate(sampled_latent[None], (self.crop_size, self.crop_size), mode='nearest')[0]
        
        img = data_highfreq
        cond_img = upsampled_latent
        
        if class_label is not None:
            return {'image': img, 'cond_img': cond_img, 'cond_inputs': [torch.tensor(class_label)]}
        else:
            return {'image': img, 'cond_img': cond_img}


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