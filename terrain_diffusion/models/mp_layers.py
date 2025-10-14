"""Magnitude-preserving layers and helper functions."""
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn


def normalize(x, dim=None, eps=1e-4):
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm


def resample(x, mode='keep', factor=2):
    """Resample the input tensor x.
    If mode is 'keep', the input tensor is returned as is.
    If the mode is 'down', the input tensor is downsampled by a factor of 2 by a 1x1 convolution with stride 2.
    If the mode is 'up', the input tensor is upsampled by a factor of 2 by a 2x2 convolution with stride 2 and uniform weight 1.
    If the mode is 'up_bilinear', the input tensor is upsampled using bilinear interpolation.
    """
    if mode == 'keep':
        return x
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, torch.ones([c, 1, 1, 1], device=x.device, dtype=x.dtype), groups=c, stride=factor)
    if mode == 'up_bilinear':
        return torch.nn.functional.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, torch.ones([c, 1, factor, factor], device=x.device, dtype=x.dtype), groups=c, stride=factor)


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

def mp_hardsilu(x):
    return torch.nn.functional.hardswish(x) / 0.576

def mp_sigmoid(x):
    return torch.sigmoid(x) / 0.208

def mp_leaky_relu(x, alpha):
    factor = np.sqrt((1 + alpha**2) / 2)
    return torch.nn.functional.leaky_relu(x, alpha) / factor


def mp_sum(args, w=None):
    """
    Magnitude preserving sum of tensors.
    parameters:
        args: list of tensors to sum.
        w: list of weights for each tensor. If None, all tensors are weighted equally. Should sum to 1 to preserve magnitude.
            If a float, the weights are [1-w, w] (a linear interpolation).
    """
    if w is None:
        w = torch.full((len(args),), 1 / len(args), dtype=args[0].dtype, device=args[0].device)
    elif isinstance(w, float):
        w = torch.tensor([1-w, w], dtype=args[0].dtype, device=args[0].device)
    else:
        w = torch.tensor(w, dtype=args[0].dtype, device=args[0].device)
    
    return torch.sum(torch.stack([args * w for args, w in zip(args, w)]), dim=0) / torch.linalg.vector_norm(w)


def mp_concat(args, dim=1, w=None):
    """
    Magnitude preserving concatenation of tensors.
    It should be noted that the concatenated tensors are already magnitude preserving, however the
    contribution of each tensor in subsequent layers is proportional to the number of channels it has.
    This function corrects for this by scaling the tensors to have the same overall magnitude, but
    the contributions of each tensor is the same.
    parameters:
        args: list of tensors to concatenate.
        w: list of weights for each tensor. If None, all tensors are weighted equally. Should sum to 1 to preserve magnitude.
            If a float, the weights are [1-w, w] (a linear interpolation).
    """
    if w is None:
        w = torch.full((len(args),), 1 / len(args), dtype=args[0].dtype, device=args[0].device)
    elif isinstance(w, float):
        w = torch.tensor([1-w, w], dtype=args[0].dtype, device=args[0].device)
    else:
        w = torch.tensor(w, dtype=args[0].dtype, device=args[0].device)
    N = [x.shape[dim] for x in args]
    sum_N = torch.tensor(sum(N), dtype=args[0].dtype, device=args[0].device)
    C = torch.sqrt(sum_N / torch.sum(torch.square(w)))
    return torch.concat([args[i] * (C / np.sqrt(args[i].shape[dim]) * w[i]) for i in range(len(args))], dim=dim)

class MPPositionalEmbedding(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        half_dim = num_channels // 2
        emb = math.log(10) / (half_dim - 1)
        self.register_buffer('freqs', torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, x):
        # Convert input to float32 for higher precision calculations
        y = x.to(torch.float32)
        
        # Compute outer product of input with frequencies
        y = y.outer(self.freqs.to(torch.float32))
        
        # Apply sin and cos, concatenate, and normalize by sqrt(2) to maintain unit variance
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=1) * np.sqrt(2)
        
        # Convert back to original dtype and return
        return y.to(x.dtype)

class MPFourier(nn.Module):
    def __init__(self, num_channels, s=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * s)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        # Convert input to float32 for higher precision calculations
        y = x.to(torch.float32)

        # Compute outer product of input with frequencies
        # This creates a 2D tensor where each row is the input multiplied by a frequency
        y = y.outer(self.freqs.to(torch.float32))

        # Add phase shifts to each element
        y = y + self.phases.to(torch.float32)

        # Apply cosine function to get periodic features
        # Multiply by sqrt(2) to maintain unit variance
        y = y.cos() * np.sqrt(2)

        # Convert back to original dtype and return
        return y.to(x.dtype)

class MPConvResample(nn.Module):
    def __init__(self, resample_mode, kernel, in_channels, out_channels, skip_weight=0.0):
        """Resamples a tensor with MP convolution or transposed convolution.

        Args:
            resample_mode (str): Either 'up', 'up_bilinear', or 'down'.
            kernel (list): Kernel size for the convolution.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            skip_weight (float): Weight for the skip connection.
        """
        super().__init__()
        self.resample_mode = resample_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_weight = skip_weight
        self.stride = kernel[0]
        if self.resample_mode == 'down':
            self.weight = nn.Parameter(torch.ones(out_channels, in_channels, *kernel))
        elif self.resample_mode == 'up' or self.resample_mode == 'up_bilinear':
            self.weight = nn.Parameter(torch.ones(in_channels, out_channels, *kernel))
        else:
            raise ValueError("resample_mode must be either 'up' or 'down'")

    def forward(self, x, gain=1):
        # Keep weight in float32 during normalization
        w = self.weight.to(torch.float32)

        # For numerical stability, we normalize the weights to internally have a norm of 1.
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))

        # Weights are already normalized, but this is critical so that gradients are propogated through the normalization.
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)

        upsampled = resample(x, mode=self.resample_mode, factor=self.stride)
        if self.resample_mode == 'down':
            y = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=0)
        else:
            y = torch.nn.functional.conv_transpose2d(x, w, stride=self.stride, padding=0)
        return mp_sum([y, upsampled], w=self.skip_weight)

    def norm_weights(self):
        with torch.no_grad():
            self.weight.copy_(normalize(self.weight.to(torch.float32)))

class MPConv(nn.Module):
    """
    Magnitude preserving convolution. Conveniently, a kernel of [] is the same as a linear layer.

    This class is a wrapper around the standard Conv2d layer, but with the following modifications:
    - During training, the weight is forced to be normalized to have a magnitude of 1.
    - The weights are then normalized to have a norm of 1 and then scaled to preserve the magnitude of the outputs.

    `gain` is used to scale the output of the layer to potentially provide more control. The default value of 1 keeps output magnitudes similar to input magnitudes.
    """
    def __init__(self, in_channels, out_channels, kernel, groups=1, no_padding=False):
        super().__init__()
        self.out_channels = out_channels
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert groups == 1 or len(kernel) == 2, "Groups other than 1 require a 2D kernel"
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))
        self.groups = groups
        self.no_padding = no_padding
    def forward(self, x, gain=1):
        # Keep weight in float32 during normalization
        w = self.weight.to(torch.float32)

        # For numerical stability, we normalize the weights to internally have a norm of 1.
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))

        # Weights are already normalized, but this is critical so that gradients are propogated through the normalization.
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)

        # If the kernel is 0D, just do a linear layer
        if w.ndim == 2:
            return nn.functional.linear(x, w)
        
        # Otherwise do a 2D convolution
        assert w.ndim == 4
        return nn.functional.conv2d(x, w, padding=(0 if self.no_padding else w.shape[-1]//2,), groups=self.groups)
    
    def norm_weights(self):
        with torch.no_grad():
            self.weight.copy_(normalize(self.weight.to(torch.float32)))

class MPEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x):
        w = self.weight.to(torch.float32)

        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
                
        w = normalize(w)
        w = w.to(x.dtype)

        assert torch.max(x) < self.weight.shape[0], f"Embedding index out of bounds: {torch.max(x).item()}"
        return nn.functional.embedding(x, self.weight)
    
    def norm_weights(self):
        with torch.no_grad():
            self.weight.copy_(normalize(self.weight.to(torch.float32)))

