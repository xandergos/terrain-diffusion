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
    cond_means_tensor = torch.as_tensor(cond_means, device=cond_img.device)
    cond_stds_tensor = torch.as_tensor(cond_stds, device=cond_img.device)
    cond_img = (cond_img - cond_means_tensor.view(1, -1, 1, 1)) / cond_stds_tensor.view(1, -1, 1, 1)
    
    cond_img[0:1] = cond_img[0:1].nan_to_num(cond_means[0])
    cond_img[1:2] = cond_img[1:2].nan_to_num(cond_means[1])
    
    noise_level = (noise_level - 0.5) * np.sqrt(12)
    
    means_crop = cond_img[:, 0:1]
    p5_crop = cond_img[:, 1:2]
    climate_means_crop = cond_img[:, 2:6, 1:3, 1:3].mean(dim=(2, 3))
    mask_crop = cond_img[:, 6:7]
    
    climate_means_crop[torch.isnan(climate_means_crop)] = torch.randn_like(climate_means_crop[torch.isnan(climate_means_crop)])
    
    return mp_concat([means_crop.flatten(1), p5_crop.flatten(1), climate_means_crop.flatten(1), mask_crop.flatten(1), histogram_raw, noise_level.view(-1, 1)], dim=1).float()

@torch.no_grad()
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
                    model_output = model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                else:
                    model_output_m = model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                    model_output_g = guide_model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
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


def sample_base_consistency(
    model,
    scheduler,
    shape: tuple[int, ...],
    cond_inputs,
    *,
    cond_means: torch.Tensor|np.ndarray,
    cond_stds: torch.Tensor|np.ndarray,
    noise_level: float = 0.0,
    histogram_raw: torch.Tensor,
    intermediate_t: float = 0.0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    tile_size: Optional[int] = None,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
    noise: Optional[torch.Tensor] = None,
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
    sigma_data = scheduler.config.sigma_data
    
    init_t = torch.atan(sigma0 / sigma_data).to(device=device, dtype=dtype)
    if intermediate_t > 0:
        t_scalars = (init_t, torch.tensor(intermediate_t, device=device, dtype=dtype))
    else:
        t_scalars = (init_t,)

    # Tiled sampling
    B, C, H, W = shape
    stride = tile_size // 2
    if weight_window_fn is None:
        weight_window_fn = _linear_weight_window
    weights = weight_window_fn(tile_size, device, dtype)

    h_starts = _tile_starts(H, tile_size, stride)
    w_starts = _tile_starts(W, tile_size, stride)
    
    if cond_inputs.ndim == 1 and len(h_starts) * len(w_starts) > 1:
        raise ValueError(f"cond_inputs must be a tensor image for tiled sampling. Cond inputs must have width {len(w_starts)+3} and height {len(h_starts)+3}.")
    elif cond_inputs.ndim == 4:
        assert cond_inputs.shape[-1] == len(w_starts)+3
        assert cond_inputs.shape[-2] == len(h_starts)+3

    sample = torch.zeros(shape, device=device, dtype=dtype)
    for step, t_scalar in enumerate(t_scalars):
        if noise is None:
            step_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            step_noise = noise[step]
        output = torch.zeros(shape, device=device, dtype=dtype)
        output_weights = torch.zeros(shape, device=device, dtype=dtype)
        for ic, i0 in enumerate(h_starts):
            for jc, j0 in enumerate(w_starts):
                if cond_inputs.ndim == 4:
                    tile_cond = cond_inputs[..., ic:ic+4, jc:jc+4]
                    tile_cond = [_process_cond_img(tile_cond, histogram_raw, cond_means, cond_stds, noise_level)]
                else:
                    tile_cond = [cond_inputs]
                
                i1, j1 = i0 + tile_size, j0 + tile_size
                z = step_noise[..., i0:i1, j0:j1] * sigma_data
                tile_sample = sample[..., i0:i1, j0:j1]

                t = t_scalar.view(1, 1, 1, 1).expand(B, 1, 1, 1)
                x_t = torch.cos(t) * tile_sample + torch.sin(t) * z
                
                model_in = x_t / sigma_data
                pred = -model(model_in, noise_labels=t.flatten(), conditional_inputs=tile_cond)
                tile_samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred

                output[..., i0:i1, j0:j1] += tile_samples * weights
                output_weights[..., i0:i1, j0:j1] += weights

        sample = output / output_weights

    return sample / scheduler.config.sigma_data
