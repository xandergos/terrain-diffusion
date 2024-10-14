import json
import math
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from diffusion.encoder import LaplacianPyramidEncoder
from .transforms import TupleTransform

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
            T.Resize((image_size, image_size)),
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
                 upsample_factor,
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
            self.posttransform = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomChoice([T.Identity(), T.RandomRotation((90, 90)), T.RandomRotation((180, 180)), T.RandomRotation((270, 270))])
            ])
        else:
            self.posttransform = T.Identity()

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
        z = self.posttransform(z)
        x, cond_img = z[:x.shape[0]], z[x.shape[0]:]
        
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

class MultiDataset(Dataset):
    def __init__(self, *sub_datasets, labels=None, weights=None):
        """
        Args:
            sub_datasets (list): The list of sub datasets.
            labels (list, optional): The list of labels to use for each sub dataset. Defaults to a range.
            weights (list, optional): The list of weights to use for each sub dataset. Defaults to a uniform distribution.
        """
        self.weights = weights or [1] * len(sub_datasets)
        self.cum_weights = [sum(self.weights[:i+1]) for i in range(len(self.weights))]
        self.sub_datasets = sub_datasets
        self.labels = labels or [list(range(len(ds))) for ds in sub_datasets]
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
        if isinstance(x, dict):
            return {**x, 'label': self.labels[ds_idx]}
        if isinstance(x, tuple) or isinstance(x, list):
            return *x, self.labels[ds_idx]
        else:
            return x, self.labels[ds_idx]

    def shuffle(self):
        for ordering in self.ordering:
            random.shuffle(ordering)


class LongDataset(Dataset):
    def __init__(self, base_dataset, length=10 ** 12):
        self.base_dataset = base_dataset
        self.length = length

    def __len__(self):
        return self.length

    def base_length(self, batch_size):
        return math.ceil(len(self.base_dataset) / batch_size)

    def __getitem__(self, index):
        if index % len(self.base_dataset) == 0:
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