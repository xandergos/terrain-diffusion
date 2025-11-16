import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import numpy as np

def laplacian_encode(x, downsample_size, sigma, interp_mode=TF.InterpolationMode.BILINEAR):
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(x)
    
    # Unsqueeze to 4 dimensions if needed
    squeeze_count = 0
    while x.ndim < 4:
        x = x.unsqueeze(0)
        squeeze_count += 1
        
    lowres = TF.resize(x, downsample_size, interpolation=interp_mode)
    lowres = TF.gaussian_blur(lowres, kernel_size=int(sigma*2)//2*2 + 1, sigma=sigma)
    lowres_up = TF.resize(lowres, x.shape[-2:], interpolation=interp_mode)
    residual = x - lowres_up
    
    # Squeeze back to original dimensions
    while squeeze_count > 0:
        residual = residual.squeeze(0)
        lowres = lowres.squeeze(0)
        squeeze_count -= 1
    
    if is_numpy:
        residual = residual.numpy()
        lowres = lowres.numpy()
    return residual, lowres

def laplacian_decode(residual, lowres, interp_mode=TF.InterpolationMode.BILINEAR):
    is_numpy = isinstance(residual, np.ndarray)
    
    # Convert to torch first if numpy
    if is_numpy:
        residual = torch.from_numpy(residual)
        lowres = torch.from_numpy(lowres)
    
    # Unsqueeze to 4 dimensions if needed
    squeeze_count = 0
    while residual.ndim < 4:
        residual = residual.unsqueeze(0)
        lowres = lowres.unsqueeze(0)
        squeeze_count += 1
    lowres_up = TF.resize(lowres, residual.shape[-2:], interpolation=interp_mode, antialias=False)
    
    # Squeeze back to original dimensions
    while squeeze_count > 0:
        residual = residual.squeeze(0)
        lowres = lowres.squeeze(0)
        lowres_up = lowres_up.squeeze(0)
        squeeze_count -= 1
        
    if is_numpy:
        residual = residual.numpy()
        lowres_up = lowres_up.numpy()
    return residual + lowres_up

def laplacian_denoise(residual, lowres, sigma, interp_mode=TF.InterpolationMode.BILINEAR):
    decoded = laplacian_decode(residual, lowres, interp_mode)
    _, new_lowres = laplacian_encode(decoded, lowres.shape[-1], sigma, interp_mode)
    return residual, new_lowres
