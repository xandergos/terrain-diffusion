from functools import lru_cache

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusion.samplers.tiled import TiledSampler
import torchvision.transforms.v2.functional as TF


def superresolution_sampler(samplers, upscaling_factors, encoders, postprocessor, noise_scales=[0.0]):
    """
    Inject the contextual network inputs into the samplers.
    Args:
        samplers (list[TiledSampler]): The samplers to inject the contextual network inputs into.
        upscaling_factors (list[int]): The upscaling factors of the samplers. 
            The first sampler does not have a parent so the list should be one element shorter than the number of samplers.
    """
    # To prevent modifying the list after the function is called
    samplers = list(samplers)
    for i, up in enumerate(upscaling_factors):
        i += 1
        
        @lru_cache(maxsize=300)
        def get_cond_img(tile_y, tile_x):
            top, left, bottom, right = samplers[i].get_tile_bounds(tile_y, tile_x)
            img_size = right - left
            
            # small_img is NOT encoded
            img = samplers[i-1].get_region(top // up, left // up, bottom // up, right // up)
            # upsample
            img = torch.nn.functional.interpolate(img, (img_size, img_size), mode='bilinear', align_corners=False)
            img = encoders[i].encode(img)
            img[:1] += TF.gaussian_noise(img[:1], clip=False, sigma=noise_scales[i-1])
            img[:1] /= np.sqrt(1 + noise_scales[i-1]**2)
            img *= 2
            
            return img
        
        def network_inputs(tiles_y, tiles_x, batch_idx):
            return {'x': torch.stack([get_cond_img(tile_y, tile_x)[batch_idx] for tile_y, tile_x, batch_idx in zip(tiles_y, tiles_x, batch_idx)], dim=0)}
        
        def intermediate_postprocessor(tile_y, tile_x, img):
            cond_img = get_cond_img(tile_y, tile_x)
            return postprocessor(tile_y, tile_x, torch.cat([img[:, :1], cond_img[:, 1:]], dim=1))
                
        samplers[i].network_inputs = network_inputs
        samplers[i].postprocessor = intermediate_postprocessor
    return samplers[-1]
