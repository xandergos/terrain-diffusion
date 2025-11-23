import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import numpy as np

def pad_linear_extrapolation(x):
    # x: (..., H, W)
    h, w = x.shape[-2:]
    
    # Pad H
    if h > 1:
        top = x[..., 0:1, :]
        second = x[..., 1:2, :]
        top_pad = 2 * top - second
        
        bot = x[..., -1:, :]
        second_last = x[..., -2:-1, :]
        bot_pad = 2 * bot - second_last
    else:
        top_pad = x[..., 0:1, :]
        bot_pad = x[..., -1:, :]
    
    x = torch.cat([top_pad, x, bot_pad], dim=-2)
    
    # Pad W
    if w > 1:
        left = x[..., :, 0:1]
        second_w = x[..., :, 1:2]
        left_pad = 2 * left - second_w
        
        right = x[..., :, -1:]
        second_last_w = x[..., :, -2:-1]
        right_pad = 2 * right - second_last_w
    else:
        left_pad = x[..., :, 0:1]
        right_pad = x[..., :, -1:]
        
    x = torch.cat([left_pad, x, right_pad], dim=-1)
    return x

def resize_extrapolated(x, size, interpolation=TF.InterpolationMode.BILINEAR, **kwargs):
    if not isinstance(size, (tuple, list)):
        return TF.resize(x, size, interpolation=interpolation, **kwargs)
        
    target_h, target_w = size
    h, w = x.shape[-2:]
    
    scale_h = target_h / h
    scale_w = target_w / w
    
    x_padded = pad_linear_extrapolation(x)
    
    new_h = int(round(target_h + 2 * scale_h))
    new_w = int(round(target_w + 2 * scale_w))
    
    out = TF.resize(x_padded, (new_h, new_w), interpolation=interpolation, **kwargs)
    
    pad_h = int(round(scale_h))
    pad_w = int(round(scale_w))
    
    return out[..., pad_h:pad_h+target_h, pad_w:pad_w+target_w]

def laplacian_encode(x, downsample_size, sigma, interp_mode=TF.InterpolationMode.BILINEAR, extrapolate=False):
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
    if not extrapolate:
        lowres_up = TF.resize(lowres, x.shape[-2:], interpolation=interp_mode)
    else:
        lowres_up = resize_extrapolated(lowres, x.shape[-2:], interpolation=interp_mode)
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

def laplacian_decode(residual, lowres, interp_mode=TF.InterpolationMode.BILINEAR, extrapolate=False, pre_padded=False):
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
        
    resize_shape = residual.shape[-2:]
    if pre_padded:
        pad_pixels = residual.shape[-1] // (lowres.shape[-1] - 2)
        resize_shape = (resize_shape[-2] + 2 * pad_pixels, resize_shape[-1] + 2 * pad_pixels)
    else:
        resize_shape = residual.shape[-2:]
    if not extrapolate:
        lowres_up = TF.resize(lowres, resize_shape, interpolation=interp_mode)
    else:
        lowres_up = resize_extrapolated(lowres, resize_shape, interpolation=interp_mode)
    
    if pre_padded:
        lowres_up = lowres_up[..., pad_pixels:-pad_pixels, pad_pixels:-pad_pixels]
    
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
    decoded = laplacian_decode(residual, lowres, interp_mode, extrapolate=True)
    _, new_lowres = laplacian_encode(decoded, lowres.shape[-1], sigma, interp_mode)
    return residual, new_lowres
