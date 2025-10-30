import torch
import torch.nn.functional as F
from typing import Optional, Callable
from terrain_diffusion.training.evaluation import _linear_weight_window, _tile_starts


def _scale_score(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Optionally scale the EDM score for sharper updates.

    Parameters
    - model_output: model's raw output (EDM score parameterization)
    - sample: current noisy sample x_t
    - sigma: current noise level (broadcastable to sample)
    - sigma_data: data standard deviation used by the scheduler
    - alpha: scaling factor (>1.0 increases sharpness); 1.0 disables scaling

    Returns
    - Adjusted score with the same shape as model_output
    """
    if alpha == 1.0:
        return model_output
    v_t = -sigma_data * model_output
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=sample.dtype, device=sample.device)
    while sigma.ndim < sample.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (sample.ndim - sigma.ndim)))
    sdata = torch.as_tensor(sigma_data, dtype=sample.dtype, device=sample.device)
    t = torch.atan(sigma / sdata)
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    x0_pred = sample * cos_t - v_t * sin_t
    noise_pred = sample * sin_t + v_t * cos_t
    x0_alpha = sample + alpha * (x0_pred - sample)
    v_t_alpha = noise_pred * cos_t - x0_alpha * sin_t
    return v_t_alpha / -sdata


@torch.no_grad()
def sample_decoder_diffusion_tiled(
    model,
    scheduler,
    cond_img: torch.Tensor,
    noise: torch.Tensor,
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    *,
    num_steps: Optional[int] = None,
    guidance_model=None,
    guidance_scale: float = 1.0,
    score_scaling: float = 1.0,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    """Run tiled conditional diffusion sampling for decoder models.

    Args:
        model: decoder diffusion model (EDM parameterization)
        scheduler: diffusion scheduler providing sigmas/timesteps and preconditioning
        cond_img: conditioning image concatenated to model input per tile
        noise: initial standard noise tensor (shape of the output)
        tile_size: optional square tile size in pixels; defaults to min(H, W)
        tile_stride: optional stride between tile starts; defaults to tile_size
        num_steps: number of scheduler steps (if None, use current scheduler state)
        guidance_model: optional guide model for two-model guidance
        guidance_scale: guidance interpolation factor
        score_scaling: EDM score scaling alpha (1.0 disables)
        weight_window_fn: optional callable(size, device, dtype) -> [1,1,S,S] weights

    Returns:
        Generated sample with the same shape/dtype/device as noise
    """
    if num_steps is not None:
        scheduler.set_timesteps(num_steps)

    b, c, h, w = noise.shape
    device, dtype = noise.device, noise.dtype
    # Ensure conditioning image matches spatial size and dtype/device of noise
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
    out = torch.zeros_like(noise)
    out_w = torch.zeros_like(noise)
    sigma_data = float(scheduler.config.sigma_data)

    h_starts = _tile_starts(h, tile_size, tile_stride)
    w_starts = _tile_starts(w, tile_size, tile_stride)

    for i0 in h_starts:
        for j0 in w_starts:
            i1, j1 = i0 + tile_size, j0 + tile_size
            samples = noise[..., i0:i1, j0:j1]
            tile_cond = cond_img[..., i0:i1, j0:j1]

            for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                t = t.to(device)
                sigma = sigma.to(device)
                scaled_input = scheduler.precondition_inputs(samples, sigma)
                cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(b))
                model_in = torch.cat([scaled_input, tile_cond], dim=1)

                if guidance_model is None or guidance_scale == 1.0:
                    mo = model(model_in, noise_labels=cnoise, conditional_inputs=[])
                else:
                    mo_m = model(model_in, noise_labels=cnoise, conditional_inputs=[])
                    mo_g = guidance_model(model_in, noise_labels=cnoise, conditional_inputs=[])
                    mo = mo_g + guidance_scale * (mo_m - mo_g)

                mo = _scale_score(mo, samples, sigma, sigma_data, alpha=score_scaling)
                samples = scheduler.step(mo, t, samples).prev_sample

            out[..., i0:i1, j0:j1] += samples * weights
            out_w[..., i0:i1, j0:j1] += weights

    return out / out_w


@torch.no_grad()
def sample_decoder_consistency_tiled(
    model,
    scheduler,
    cond_img: torch.Tensor,
    noise: torch.Tensor,
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    *,
    intermediate_t: Optional[torch.Tensor | float | list | tuple] = None,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    """Run n-step consistency sampling for decoder models with tiling.

    Args:
        model: consistency decoder model (single-step update network)
        scheduler: provides sigmas and sigma_data for trig-flow conversion
        cond_img: conditioning image concatenated to model input per tile
        noise: standard initial noise tensor used for constructing x_t at each step
        tile_size: optional square tile size in pixels; defaults to min(H, W)
        tile_stride: optional stride between tile starts; defaults to tile_size
        intermediate_t: list of timesteps for additional steps after init.
        weight_window_fn: optional callable(size, device, dtype) -> [1,1,S,S] weights

    Returns:
        Generated sample with the same shape/dtype/device as noise
    """
    b, c, h, w = noise.shape
    device, dtype = noise.device, noise.dtype
    # Ensure conditioning image matches spatial size and dtype/device of noise
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
    out = torch.zeros_like(noise)
    out_w = torch.zeros_like(noise)

    sigma_data = float(scheduler.config.sigma_data)
    s0 = float(scheduler.sigmas[0])
    init_t = torch.tensor(torch.atan(torch.as_tensor(s0 / sigma_data)), device=device, dtype=dtype)

    if intermediate_t is None:
        t_scalars = (init_t,)
    else:
        if torch.is_tensor(intermediate_t):
            flat_t = intermediate_t.flatten()
            if flat_t.numel() == 0:
                t_scalars = (init_t,)
            else:
                t_scalars = (init_t, *[tt.to(device=device, dtype=dtype) for tt in flat_t])
        elif isinstance(intermediate_t, (list, tuple)):
            t_scalars = (init_t, *[torch.tensor(t, device=device, dtype=dtype) for t in intermediate_t])
        else:
            t_scalars = (init_t, torch.tensor(float(intermediate_t), device=device, dtype=dtype))

    h_starts = _tile_starts(h, tile_size, tile_stride)
    w_starts = _tile_starts(w, tile_size, tile_stride)

    for i0 in h_starts:
        for j0 in w_starts:
            i1, j1 = i0 + tile_size, j0 + tile_size
            samples = torch.zeros((b, c, tile_size, tile_size), device=device, dtype=dtype)
            tile_cond = cond_img[..., i0:i1, j0:j1]
            tile_noise = noise[..., i0:i1, j0:j1]

            for t_scalar in t_scalars:
                t = t_scalar.view(1, 1, 1, 1).expand(b, 1, 1, 1).to(device=device, dtype=dtype)
                z = tile_noise * sigma_data
                x_t = torch.cos(t) * samples + torch.sin(t) * z
                model_in = x_t / sigma_data
                model_in = torch.cat([model_in, tile_cond], dim=1)
                pred = -model(model_in, noise_labels=t.flatten(), conditional_inputs=[])
                samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred

            out[..., i0:i1, j0:j1] += samples * weights
            out_w[..., i0:i1, j0:j1] += weights

    return out / out_w / sigma_data


