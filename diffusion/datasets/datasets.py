import json
import math
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as T

from diffusion.encoder import LaplacianPyramidEncoder

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

class UpsamplingTerrainDataset(CachedTiffDataset):
    """
    Upsampling terrain dataset. Returns a tuple of (image, cond_image, context_images), 
    where image is the image to predict (the next level in the pyramid),
    cond_images are the lower levels of the pyramid (cropped),
    and context_images are the lower levels of the pyramid but zoomed out around the cond_images. 
    The amount of zoom is determined by the context_scales.

    It is crucial that the transforms applied result in an image with multiple channels, where the first channel is the output
    and the rest are condition images. Usually, this means that eventually LaplacianPyramidEncoder is applied.
    """
    def __init__(self, paths, pretransform, posttransform, read_image_fn=None, cache: dict = None,
                 crop_size=64, context_scales=None, eval_dataset=False, root_dir=None):
        """
        Args:
            paths (list | str): The paths where the images are located. 
                If a string is provided, behavior depends on whether it is a folder or file. If it is a file, it is treated as a json file containing a list of paths.
                If it is a folder, all files in the folder are used.
            pretransform (callable): The transform to apply to the images after reading and before caching them. This is where deterministic transforms should be applied.
            posttransform (callable): The transform to apply to the images after reading and after caching them. This is where random transforms should be applied.
            read_image_fn (callable, optional): The function to use to read the images. Defaults to `PIL.Image.open`.
            cache (dict, optional): A dictionary to use for caching. Defaults to None. You can pass in a dictionary subset to provide custom caching logic.
            crop_size (int): The size of the crop to extract from the image.
            context_scales (list, optional): List of scales for context images. Defaults to None. 1 means use full resolution, 4 means zoom in 4x, etc. 
                A scale of image_size / crop_size will give cond_image.
            eval_dataset (bool, optional): Whether to make crops deterministically (center crop). Defaults to False.
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
        super().__init__(paths, pretransform, posttransform, read_image_fn, cache)
        self.crop_size = crop_size
        self.context_scales = context_scales or []
        self.eval_dataset = eval_dataset

    def __getitem__(self, index):
        out = super().__getitem__(index)
        x = out[:1]
        cond_img = out[1:]
        if self.eval_dataset:
            i, j, h, w = x.shape[1] // 2 - self.crop_size // 2, x.shape[1] // 2 - self.crop_size // 2, self.crop_size, self.crop_size
        else:
            i, j, h, w = T.RandomCrop.get_params(cond_img, output_size=(self.crop_size, self.crop_size))

        x = x[:, i:i+h, j:j+w]

        center_y, center_x = i + h // 2, j + w // 2
        translate_x = center_x - cond_img.shape[2] // 2
        translate_y = center_y - cond_img.shape[1] // 2
        context = []
        masked_cond = torch.concat([cond_img, torch.ones_like(cond_img[:1])], dim=0)
        for scale in self.context_scales:
            c = T.functional.affine(masked_cond, angle=0, translate=(-translate_x * scale, -translate_y * scale), scale=scale, shear=0, fill=0)
            c = T.functional.resize(c, (self.crop_size, self.crop_size))
            context.append(c)
            
        cond_img = cond_img[:, i:i+h, j:j+w]
        return {'image': x, 'cond_image': cond_img, 'context': context}


class MultiDataset(Dataset):
    def __init__(self, *sub_datasets, weights=None):
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
        if isinstance(x, dict):
            return {**x, 'label': ds_idx}
        if isinstance(x, tuple) or isinstance(x, list):
            return *x, ds_idx
        else:
            return x, ds_idx

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