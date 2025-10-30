from __future__ import annotations

from terrain_diffusion.models.mp_layers import mp_concat
from terrain_diffusion.training.evaluation import _linear_weight_window, _tile_starts
from typing import Optional, Callable
import numpy as np

import torch
import torch.nn.functional as F

def _process_cond_img(
    cond_img: torch.Tensor, 
    histogram_raw: torch.Tensor, 
    cond_means: torch.Tensor, 
    cond_stds: torch.Tensor,
    noise_level: torch.Tensor = torch.tensor(0.0)
) -> torch.Tensor:
    """
    Process the conditioning image and compute the final conditioning tensor.

    Args:
        cond_img (torch.Tensor): Conditioning image tensor of shape (B, C, 4, 4).
            Channels are: means (signed-sqrt), p5 (signed-sqrt), temp mean (C), temp std (C), precip mean (mm/yr), precip std (coeff of var)
        histogram_raw (torch.Tensor): Raw histogram (pre-softmax values) features to include in the conditioning vector. Length equal to the number of subsets trained on.
        cond_means (torch.Tensor): Array or tensor with means for normalization.
        cond_stds (torch.Tensor): Array or tensor with stds for normalization.
        noise_level (float): Noise level (0-1) to apply to the conditioning tensor.

    Returns:
        torch.Tensor: Processed conditioning tensor to be passed into the model.
    """
    cond_img = (cond_img - torch.tensor(cond_means, device=cond_img.device).view(1, -1, 1, 1)) / torch.tensor(cond_stds, device=cond_img.device).view(1, -1, 1, 1)
    
    cond_img[0:1] = cond_img[0:1].nan_to_num(cond_means[0])
    cond_img[1:2] = cond_img[1:2].nan_to_num(cond_means[1])
    
    noise_level = (noise_level - 0.5) * np.sqrt(12)
    
    means_crop = cond_img[:, 0:1]
    p5_crop = cond_img[:, 1:2]
    climate_means_crop = cond_img[:, 2:6, 1:3, 1:3].mean(dim=(2, 3))
    mask_crop = cond_img[:, 6:7]
    
    climate_means_crop[torch.isnan(climate_means_crop)] = torch.randn_like(climate_means_crop[torch.isnan(climate_means_crop)])
    
    return mp_concat([means_crop.flatten(1), p5_crop.flatten(1), climate_means_crop.flatten(1), mask_crop.flatten(1), histogram_raw, noise_level.view(-1, 1)], dim=1).float()

def sample_base_diffusion(
    model,
    scheduler,
    shape: tuple[int, ...],
    cond_inputs,
    *,
    cond_means: torch.Tensor|np.ndarray,
    cond_stds: torch.Tensor|np.ndarray,
    noise_level: float = 0.0,
    histogram_raw: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    steps: int = 15,
    guide_model = None,
    guidance_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    tile_size: Optional[int] = None,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    """Sample a base image using diffusion, with optional tiled blending.

    Args:
        model: base diffusion model (EDM parameterization)
        scheduler: diffusion scheduler providing sigmas/timesteps and preconditioning
        shape: output shape (B, C, H, W)
        cond_means: Array or tensor with means for means (signed-sqrt), p5 (signed-sqrt), temp mean (C), temp std (C), precip mean (mm/yr), precip std (coeff of var), mask.
        cond_stds: Array or tensor with stds for means (signed-sqrt), p5 (signed-sqrt), temp mean (C), temp std (C), precip mean (mm/yr), precip std (coeff of var), mask.
        noise_level: Noise level (0-1) to apply to the conditioning tensor. Shape (B, 1).
        histogram_raw: Raw histogram (pre-softmax values) features to include in the conditioning vector. Length equal to the number of subsets trained on.
        dtype: dtype of generated tensor
        steps: number of scheduler steps
        cond_inputs: Either a tensor image (required for tiling) or a 1D pre-processed tensor
        guide_model: optional guide model for two-model guidance
        guidance_scale: guidance interpolation factor
        generator: optional torch.Generator
        tile_size: if set, enables tiled sampling with square tiles of this size
        weight_window_fn: optional callable(size, device, dtype) -> [1,1,S,S] weights. Defaults to _linear_weight_window

    Returns:
        torch.Tensor: Generated tensor with shape 'shape'
    """
    device = next(model.parameters()).device
    sigma0 = scheduler.sigmas[0].to(device)

    if tile_size is None:
        samples = torch.randn(shape, generator=generator, device=device, dtype=dtype) * sigma0
        scheduler.set_timesteps(steps)
        with torch.no_grad():
            for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                t = t.to(device)
                sigma = sigma.to(device)
                scaled_input = scheduler.precondition_inputs(samples, sigma)
                cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))

                x = scaled_input
                if guide_model is None or guidance_scale == 1.0:
                    model_output = model(x, noise_labels=cnoise, conditional_inputs=cond_inputs or [])
                else:
                    model_output_m = model(x, noise_labels=cnoise, conditional_inputs=cond_inputs or [])
                    model_output_g = guide_model(x, noise_labels=cnoise, conditional_inputs=cond_inputs or [])
                    model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)

                samples = scheduler.step(model_output, t, samples, generator=generator).prev_sample
        return samples

    # Tiled sampling
    B, C, H, W = shape
    stride = tile_size // 2
    output = torch.zeros(shape, device=device, dtype=dtype)
    output_weights = torch.zeros(shape, device=device, dtype=dtype)
    if weight_window_fn is None:
        weight_window_fn = _linear_weight_window
    weights = weight_window_fn(tile_size, device, dtype)

    initial_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype) * sigma0

    h_starts = _tile_starts(H, tile_size, stride)
    w_starts = _tile_starts(W, tile_size, stride)
    
    if cond_inputs.ndim == 1 and len(h_starts) * len(w_starts) > 1:
        raise ValueError(f"cond_inputs must be a tensor image for tiled sampling. Cond inputs must have width {len(w_starts)+3} and height {len(h_starts)+3}.")
    elif cond_inputs.ndim == 4:
        assert cond_inputs.shape[-1] == len(w_starts)+3
        assert cond_inputs.shape[-2] == len(h_starts)+3

    with torch.no_grad():
        for ic, i0 in enumerate(h_starts):
            for jc, j0 in enumerate(w_starts):
                if cond_inputs.ndim == 4:
                    tile_cond = cond_inputs[..., ic:ic+4, jc:jc+4]
                    tile_cond = [_process_cond_img(tile_cond, histogram_raw, cond_means, cond_stds, noise_level)]
                else:
                    tile_cond = [cond_inputs]
                
                i1, j1 = i0 + tile_size, j0 + tile_size
                tile_samples = initial_noise[..., i0:i1, j0:j1]

                scheduler.set_timesteps(steps)
                for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                    t = t.to(device)
                    sigma = sigma.to(device)
                    scaled_input = scheduler.precondition_inputs(tile_samples, sigma)
                    cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(tile_samples.shape[0]))

                    x = scaled_input
                    if guide_model is None or guidance_scale == 1.0:
                        model_output = model(x, noise_labels=cnoise, conditional_inputs=tile_cond)
                    else:
                        model_output_m = model(x, noise_labels=cnoise, conditional_inputs=tile_cond)
                        model_output_g = guide_model(x, noise_labels=cnoise, conditional_inputs=tile_cond)
                        model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)

                    tile_samples = scheduler.step(model_output, t, tile_samples, generator=generator).prev_sample

                output[..., i0:i1, j0:j1] += tile_samples * weights
                output_weights[..., i0:i1, j0:j1] += weights


    return output / output_weights / scheduler.config.sigma_data


@torch.no_grad()
def sample_base_consistency_tiled(
    model,
    scheduler,
    *,
    cond_img: Optional[torch.Tensor],
    noise: torch.Tensor,
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    intermediate_t: Optional[torch.Tensor | float | list | tuple] = None,
    conditional_inputs=None,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    """Run n-step consistency sampling for base models with tiling.

    Args:
        model: consistency base model (single-step update network)
        scheduler: provides sigmas and sigma_data for trig-flow conversion
        cond_img: optional conditioning image to concat to model input
        noise: standard initial noise tensor used for constructing x_t at each step. If len(shape) == 5, the first channel is for the noise level.
        tile_size: optional square tile size in pixels; defaults to min(H, W)
        tile_stride: optional stride between tile starts; defaults to tile_size
        intermediate_t: list/tensor/scalar of additional timesteps after init
        conditional_inputs: optional list passed through to model forward
        weight_window_fn: callable(size, device, dtype) -> [1,1,S,S] weights

    Returns:
        Generated sample with the same shape/dtype/device as noise
    """
    b, c, h, w = noise.shape[-4:]
    device, dtype = noise.device, noise.dtype
    sigma_data = scheduler.config.sigma_data

    if cond_img is not None:
        cond_img = cond_img.to(device=device, dtype=dtype)
        if cond_img.shape[-2:] != (h, w):
            cond_img = F.interpolate(cond_img, size=(h, w), mode="nearest")

    if weight_window_fn is None:
        weight_window_fn = _linear_weight_window
    if tile_size is None:
        tile_size = min(h, w)
    if tile_stride is None:
        tile_stride = tile_size
    weights = weight_window_fn(tile_size, device, dtype)

    out = torch.zeros(noise.shape[-4:], device=device, dtype=dtype)
    out_w = torch.zeros(noise.shape[-4:], device=device, dtype=dtype)

    # Build timesteps: init_t from largest sigma, then optional intermediates
    s0 = scheduler.sigmas[0]
    init_t = torch.tensor(torch.atan(torch.as_tensor(s0 / sigma_data)), device=device, dtype=dtype)

    if intermediate_t is None:
        t_scalars = (init_t,)
    else:
        if torch.is_tensor(intermediate_t):
            flat_t = intermediate_t.flatten()
            t_scalars = (init_t,) if flat_t.numel() == 0 else (init_t, *[tt.to(device=device, dtype=dtype) for tt in flat_t])
        elif isinstance(intermediate_t, (list, tuple)):
            t_scalars = (init_t, *[torch.tensor(t, device=device, dtype=dtype) for t in intermediate_t])
        else:
            t_scalars = (init_t, torch.tensor(intermediate_t, device=device, dtype=dtype))

    h_starts = _tile_starts(h, tile_size, tile_stride)
    w_starts = _tile_starts(w, tile_size, tile_stride)

    for i0 in h_starts:
        for j0 in w_starts:
            i1, j1 = i0 + tile_size, j0 + tile_size
            samples = torch.zeros((b, c, tile_size, tile_size), device=device, dtype=dtype)
            tile_noise = noise[..., i0:i1, j0:j1]
            tile_cond = cond_img[..., i0:i1, j0:j1] if cond_img is not None else None

            for step, t_scalar in enumerate(t_scalars):
                t = t_scalar.view(1, 1, 1, 1).expand(b, 1, 1, 1)
                z = tile_noise * sigma_data if len(noise.shape) == 4 else tile_noise[step] * sigma_data
                x_t = torch.cos(t) * samples + torch.sin(t) * z
                model_in = x_t / sigma_data
                if tile_cond is not None:
                    model_in = torch.cat([model_in, tile_cond], dim=1)
                pred = -model(model_in, noise_labels=t.flatten(), conditional_inputs=conditional_inputs or [])
                samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred

            out[..., i0:i1, j0:j1] += samples * weights
            out_w[..., i0:i1, j0:j1] += weights

    return out / out_w / sigma_data
