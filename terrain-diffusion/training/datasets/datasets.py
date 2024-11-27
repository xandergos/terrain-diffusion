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
from data.laplacian_encoder import LaplacianPyramidEncoder
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



class CachedTiffDataset(Dataset):
    """Simple dataset that reads tiff images from a folder and applies a transformation.
    The images are cached when read for the first time if caching is enabled. The cache has no limit."""

    def __init__(
            self,
            paths,
            pretransform,
            posttransform,
            read_image_fn=None,
            cache: dict=None
    ):
        """
        Args:
            paths:
                The paths where the images are located
            pretransform:
                The transform to apply to the images after reading and before caching them.
            posttransform:
                The transform to apply to the images after reading and after caching them.
            read_image_fn:
                (Optional) The function to use to read the images. `PIL.Image.open` by default.
            use_cache:
                (Optional) Whether to use caching. Default is True.
        """
        super().__init__()
        self.read_image_fn = read_image_fn or Image.open
        self.paths = paths
        self.pretransform = pretransform
        self.posttransform = posttransform
        self.cache = cache

    def __len__(self):
        return len(self.paths)

    def shuffle(self):
        random.shuffle(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if self.cache is not None:
            if path not in self.cache:
                img = self.read_image_fn(path)
                out = self.pretransform(img)
                self.cache[path] = out
            else:
                out = self.cache[path]
        else:
            img = self.read_image_fn(path)
            out = self.pretransform(img)
        out = self.posttransform(out)
        return out
    
class BaseTerrainDataset(CachedTiffDataset):
    """
    A simple base terrain dataset that applies a Laplacian pyramid encoder to the images.
    """
    def __init__(self, paths, image_size, crop_size,
                 pyramid_scales, pyramid_sigma, pyramid_raw_mean, pyramid_raw_std,
                 read_image_fn=None, cache: dict = None, eval_dataset=False, root_dir=None):
        """
        Args:
            paths (list | str): The paths where the images are located. 
                If a string is provided, behavior depends on whether it is a folder or file. If it is a file, it is treated as a json file containing a list of paths.
                If it is a folder, all files in the folder are used.
            image_size (int): The size to which images will be resized before applying the Laplacian pyramid.
            crop_size (int): The size of the crop to extract from the image.
            pyramid_scales (list): Amount to downsample each layer in the Laplacian pyramid.
            pyramid_sigma (float): Sigma used for gaussian blur in the Laplacian pyramid.
            pyramid_raw_mean (list): Expected mean of each channel in the Laplacian pyramid.
            pyramid_raw_std (list): Expected standard deviation of each channel in the Laplacian pyramid.
            read_image_fn (callable, optional): The function to use to read the images. Defaults to PIL.Image.open.
            cache (dict, optional): A dictionary to use for caching. Defaults to None (no caching).
            eval_dataset (bool, optional): Whether to use evaluation mode (center crop, no augmentations). Defaults to False.
            root_dir (str, optional): The root directory to prepend to the paths. Defaults to None (paths are used as is).
        """
        if isinstance(paths, str):
            if os.path.isdir(paths):
                paths = [os.path.join(paths, p) for p in os.listdir(paths)]
            else:
                with open(paths) as f:
                    paths = json.load(f)
        if root_dir is not None:
            paths = [os.path.join(root_dir, p) for p in paths]
        
        self.encoder = LaplacianPyramidEncoder(pyramid_scales, pyramid_sigma, pyramid_raw_mean, pyramid_raw_std)
        pretransform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=False),
            AreaResize((image_size, image_size)),
            self.encoder,
        ])
        if not eval_dataset:
            posttransform = T.Compose([
                T.RandomCrop((crop_size, crop_size)),
                T.RandomVerticalFlip(),
                T.RandomChoice([T.Identity(), T.RandomRotation((90, 90)), T.RandomRotation((180, 180)), T.RandomRotation((270, 270))])
            ])
        else:
            posttransform = T.Compose([
                T.CenterCrop((crop_size, crop_size)),
            ])
        super().__init__(paths, pretransform, posttransform, read_image_fn, cache)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return {'image': item}

class SuperresTerrainDataset(CachedTiffDataset):
    """
    A terrain dataset that applies a Laplacian pyramid encoder to the images, 
    using lower level for conditioning.
    """
    def __init__(self, paths, image_size, crop_size,
                 pyramid_scales, pyramid_sigma, pyramid_raw_mean, pyramid_raw_std,
                 upsample_factor, noise_scale=0.0,
                 read_image_fn=None, cache: dict = None, eval_dataset=False, 
                 root_dir=None):
        """
        Args:
            paths (list | str): The paths where the images are located. 
                If a string is provided, behavior depends on whether it is a folder or file. If it is a file, it is treated as a json file containing a list of paths.
                If it is a folder, all files in the folder are used.
            image_size (int): The size to which images will be resized before applying the Laplacian pyramid.
            crop_size (int): The size of the crop to extract from the image.
            pyramid_scales (list): Amount to downsample each layer in the Laplacian pyramid.
            pyramid_sigma (float): Sigma used for gaussian blur in the Laplacian pyramid.
            pyramid_raw_mean (list): Expected mean of each channel in the Laplacian pyramid.
            pyramid_raw_std (list): Expected standard deviation of each channel in the Laplacian pyramid.
            upsample_factor (int): The upsampling factor to apply to the conditional image.
            noise_scale (float): The scale of the gaussian noise to add to the conditional image.
            read_image_fn (callable, optional): The function to use to read the images. Defaults to PIL.Image.open.
            cache (dict, optional): A dictionary to use for caching. Defaults to None (no caching).
            eval_dataset (bool, optional): Whether to use evaluation mode (center crop, no augmentations). Defaults to False.
            root_dir (str, optional): The root directory to prepend to the paths. Defaults to None (paths are used as is).
        """
        if isinstance(paths, str):
            if os.path.isdir(paths):
                paths = [os.path.join(paths, p) for p in os.listdir(paths)]
            else:
                with open(paths) as f:
                    paths = json.load(f)
        if root_dir is not None:
            paths = [os.path.join(root_dir, p) for p in paths]
        
        self.noise_scale = noise_scale
        self.encoder = LaplacianPyramidEncoder(pyramid_scales, pyramid_sigma, pyramid_raw_mean, pyramid_raw_std)
        pretransform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=False),
            T.Resize((image_size, image_size)),
            TupleTransform(
                # Target image
                self.encoder,
                # Conditional image
                T.Compose([
                    T.Resize((image_size // upsample_factor, image_size // upsample_factor)), 
                    T.Lambda(lambda x: torch.nn.functional.interpolate(x[None], (image_size, image_size), mode='bicubic', align_corners=False)[0]),
                    self.encoder,
                ]),
            ),
        ])
        super().__init__(paths, pretransform, T.Identity(), read_image_fn, cache)
        
        self.upsample_factor = upsample_factor
        self.crop_size = crop_size
        self.image_size = image_size
        self.eval_dataset = eval_dataset
        
        # Apply this after the parent
        if not eval_dataset:
            self._posttransform = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomChoice([T.Identity(), T.RandomRotation((90, 90)), T.RandomRotation((180, 180)), T.RandomRotation((270, 270))])
            ])
        else:
            self._posttransform = T.Identity()

    def __getitem__(self, index):
        img, cond_img = super().__getitem__(index)
        
        # Assume only the first level needs to be predicted; other levels are low frequency and can just be upsampled.
        x = img[:1]
        
        if self.eval_dataset:
            i, j, h, w = x.shape[1] // 2 - self.crop_size // 2, x.shape[1] // 2 - self.crop_size // 2, self.crop_size, self.crop_size
        else:
            i, j, h, w = T.RandomCrop.get_params(cond_img, output_size=(self.crop_size, self.crop_size))

        x = x[:, i:i+h, j:j+w]
        cond_img = cond_img[:, i:i+h, j:j+w]
        
        # temporarily merge along channel dim to apply post transform
        z = torch.cat([x, cond_img], dim=0)
        z = self._posttransform(z)
        x, cond_img = z[:x.shape[0]], z[x.shape[0]:]
        cond_img[:1] += TF.gaussian_noise(cond_img[:1], clip=False, sigma=self.noise_scale)
        cond_img[:1] /= np.sqrt(1 + self.noise_scale**2)
        cond_img *= 2
        
        #center_y, center_x = i + h // 2, j + w // 2
        #translate_x = center_x - cond_img.shape[2] // 2
        #translate_y = center_y - cond_img.shape[1] // 2
        #context = []
        #masked_cond = torch.concat([cond_img, torch.ones_like(cond_img[:1])], dim=0)
        #for scale in self.context_scales:
        #    c = T.functional.affine(masked_cond, angle=0, translate=(-translate_x * scale, -translate_y * scale), scale=scale, shear=0, fill=0)
        #    c = T.functional.resize(c, (self.crop_size, self.crop_size))
        #    context.append(c)
            
        return {'image': x, 'cond_img': cond_img}
    
class H5BaseTerrainDataset(Dataset):
    """Dataset for reading terrain data from an HDF5 file."""

    def __init__(self, h5_file, crop_size, pct_land_range, dataset_label, model_label, eval_dataset=False,
                 latent_channels=8, latents_mean=None, latents_std=None, sigma_data=0.5):
        """
        Args:
            h5_file (str): Path to the HDF5 file.
            crop_size (int): Size of the square crop.
            pct_land_range (list): Range of acceptable pct_land values [min, max].
        """
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_range = pct_land_range
        self.model_label = torch.tensor(model_label)
        self.latent_channels = latent_channels
        self.latents_mean = torch.tensor(latents_mean).view(-1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(-1, 1, 1)
        self.sigma_data = sigma_data
        with h5py.File(self.h5_file, 'r') as f:
            self.keys = []
            for key in f.keys():
                if pct_land_range[0] <= f[key].attrs['pct_land'] <= pct_land_range[1] and (dataset_label is None or str(f[key].attrs['label']) == str(dataset_label)):
                    self.keys.append(key)

        if not eval_dataset:
            self.transform = T.Compose([
                T.RandomCrop((self.crop_size, self.crop_size)),
                T.RandomVerticalFlip(),
                T.RandomChoice([
                    T.Identity(),
                    T.RandomRotation((90, 90)),
                    T.RandomRotation((180, 180)),
                    T.RandomRotation((270, 270))
                ])
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop((self.crop_size, self.crop_size)),
            ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as f:
            data = torch.from_numpy(f[self.keys[index]][:])
        data = self.transform(data)
        
        latents = data[:self.latent_channels]
        latent_means, latent_vars = latents[:self.latent_channels//2], latents[self.latent_channels//2:]
        sampled_latents = torch.randn_like(latent_means) * latent_vars.exp() + latent_means
        sampled_latents = (sampled_latents - self.latents_mean) / self.latents_std * self.sigma_data
        data = torch.cat([sampled_latents, data[self.latent_channels:]], dim=0)
        
        # Conditional input using last channel
        cond_img_noise = (torch.rand([]) * 12 - 4).exp()
        noise_label = 0.25 * torch.log(cond_img_noise)
        cond_img = data[-1:] if cond_img_noise.item() < 2900 else torch.zeros_like(data[-1:])  # After 2900 just make 0 to ensure no leakage
        cond_img = (cond_img + torch.randn_like(cond_img) * cond_img_noise) / np.sqrt(1 + cond_img_noise**2)
        
        return {'image': data, 'cond_img': cond_img, 'cond_inputs': [self.model_label, noise_label]}


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
    def __init__(self, h5_file, crop_size, pct_land_range, dataset_label, eval_dataset=False,
                 latents_mean=None, latents_std=None, sigma_data=0.5, clip_edges=True):
        """
        Args:
            h5_file (str): Path to the HDF5 file.
            crop_size (int): Size of the square crop.
            pct_land_range (list): Range of acceptable pct_land values [min, max].
        """
        self.h5_file = h5_file
        self.crop_size = crop_size
        self.pct_land_range = pct_land_range
        self.latents_mean = torch.tensor(latents_mean).view(-1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(-1, 1, 1)
        self.sigma_data = sigma_data
        self.clip_edges = clip_edges
        self.eval_dataset = eval_dataset
        
        with h5py.File(self.h5_file, 'r') as f:
            self.keys = []
            search_label = dataset_label + '_latent' if dataset_label is not None else None
            for key in f.keys():
                if pct_land_range[0] <= f[key].attrs['pct_land'] <= pct_land_range[1] and (search_label is None or str(f[key].attrs['label']) == str(search_label)):
                    self.keys.append('_'.join(key.split('_')[:-1]))
        
        self.dummy_data_latent = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key_latent = self.keys[index] + '_latent'
        key_highfreq = self.keys[index] + '_highfreq'
        
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
        
        return {'image': img, 'cond_img': cond_img}

class MultiDataset(Dataset):
    def __init__(self, *sub_datasets, weights=None):
        """
        Args:
            sub_datasets (list): The list of sub datasets.
            labels (list, optional): The list of labels to use for each sub dataset. Defaults to a range.
            weights (list, optional): The list of weights to use for each sub dataset. Defaults to a uniform distribution.
        """
        self.weights = weights or [1] * len(sub_datasets)
        self.cum_weights = [sum(self.weights[:i+1]) for i in range(len(self.weights))]
        self.sub_datasets = sub_datasets
        self.ordering = []
        for ds in self.sub_datasets:
            self.ordering.append(list(range(len(ds))))

    def __len__(self):
        return max(len(ds) for ds in self.sub_datasets)

    def __getitem__(self, index):
        ds_idx = random.choices(list(range(len(self.sub_datasets))), cum_weights=self.cum_weights)[0]
        ds = self.sub_datasets[ds_idx]
        ordering = self.ordering[ds_idx]
        item_idx = ordering[index % len(ordering)]
        x = ds[item_idx]
        return x

    def shuffle(self):
        for ordering in self.ordering:
            random.shuffle(ordering)
    
    def split(self, val_pct, generator=None):
        train_datasets = []
        val_datasets = []
        for dataset in self.sub_datasets:
            train_size = int(len(dataset) * (1 - val_pct))
            train_dset, val_dset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
            train_datasets.append(train_dset)
            val_datasets.append(val_dset)
        return MultiDataset(*train_datasets, weights=self.weights), MultiDataset(*val_datasets, weights=self.weights)


class LongDataset(Dataset):
    def __init__(self, base_dataset, length=10 ** 12, shuffle=True):
        self.base_dataset = base_dataset
        self.length = length
        self.shuffle = shuffle

    def __len__(self):
        return self.length

    def base_length(self, batch_size):
        return math.ceil(len(self.base_dataset) / batch_size)

    def __getitem__(self, index):
        if index % len(self.base_dataset) == 0 and self.shuffle:
            self.base_dataset.shuffle()
        return self.base_dataset[index % len(self.base_dataset)]


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