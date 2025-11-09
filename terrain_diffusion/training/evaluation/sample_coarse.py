import torch
from typing import Optional, Callable, Union

from terrain_diffusion.training.evaluation import _linear_weight_window, _tile_starts


def _cond_inputs_from_snr(
    cond_snr: Union[float, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Build conditional input scalars expected by the coarse model from SNR.

    Uses the same parametrization as in the visualization script:
      t = atan(snr); cond_input_i = log(tan(t_i) / 8)
    Returned as a list of 0-d tensors on the correct device/dtype.
    """
    if not isinstance(cond_snr, torch.Tensor):
        snr = torch.as_tensor(cond_snr, device=device, dtype=dtype)
    else:
        snr = cond_snr.to(device=device, dtype=dtype)

    t = torch.atan(snr)
    vals = torch.log(torch.tan(t) / 8.0)
    return [v.detach().to(device=device, dtype=dtype).view(-1) for v in vals.transpose(0, 1)]


@torch.no_grad()
def sample_coarse_tiled(
    model,
    scheduler,
    cond_img: torch.Tensor,
    cond_snr: Union[float, torch.Tensor],
    *,
    steps: int = 15,
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample from the coarse diffusion model with optional tiling and blending.

    Args:
        model: Diffusion model. Called as model(x, noise_labels, conditional_inputs).
        scheduler: Scheduler providing sigmas, set_timesteps, timesteps, precondition_inputs,
            trigflow_precondition_noise, step, and config.sigma_data.
        cond_img: Conditional image tensor [B, C_cond, H, W]. Normalized.
        cond_snr: Scalar or per-channel SNR for the conditional image.
        steps: Number of scheduler steps.
        tile_size: Spatial tile size. If None, processes the full image without tiling.
        tile_stride: Stride between tiles. Defaults to tile_size if not provided.
        weight_window_fn: Function to build spatial blending weights for tiles.
        generator: Optional torch.Generator for deterministic sampling.
        dtype: Compute dtype for autocast during model forward.

    Returns:
        Tensor of samples with shape [B, out_channels, H, W].
    """
    assert cond_img.ndim == 4, "cond_img must be [B, C, H, W]"

    b, c_cond, h, w = cond_img.shape
    device = cond_img.device

    if tile_size is None:
        tile_size = w  # match sample_autoencoder: default to full image width
    if tile_stride is None:
        tile_stride = tile_size

    if weight_window_fn is None:
        weight_window_fn = _linear_weight_window
    weights = weight_window_fn(tile_size, device, dtype)

    # Determine output channels robustly
    out_channels = int(getattr(model.config, 'out_channels', None) or getattr(model.config, 'in_channels', 1))

    samples_out = torch.zeros((b, out_channels, h, w), device=device, dtype=dtype)
    samples_w = torch.zeros_like(samples_out)

    # Prepare conditioning inputs from SNR (list of scalars)
    cond_inputs = _cond_inputs_from_snr(cond_snr, device, dtype)
    t_cond = torch.atan(cond_snr).view(1, -1, 1, 1).to(cond_img.device)
    cond_img = torch.cos(t_cond) * cond_img + torch.sin(t_cond) * torch.randn_like(cond_img)

    # Prepare scheduler
    scheduler.set_timesteps(int(steps))
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    h_starts = _tile_starts(h, tile_size, tile_stride)
    w_starts = _tile_starts(w, tile_size, tile_stride)

    for i0 in h_starts:
        for j0 in w_starts:
            i1, j1 = i0 + tile_size, j0 + tile_size

            # Extract conditional tile (pad/truncate handled by _tile_starts boundaries)
            cond_tile = cond_img[..., i0:i1, j0:j1]

            # Initialize noisy sample for this tile
            init_sigma = scheduler.sigmas[0].to(device)
            tile_shape = (b, out_channels, cond_tile.shape[-2], cond_tile.shape[-1])
            tile_sample = torch.randn(tile_shape, device=device, generator=generator) * init_sigma

            # Diffusion steps
            for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                t = t.to(device)
                sigma = sigma.to(device)

                scaled_in = scheduler.precondition_inputs(tile_sample, sigma)
                cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(b))

                x_in = torch.cat([scaled_in, cond_tile], dim=1)
                with torch.autocast(device_type=autocast_device, dtype=dtype):
                    model_out = model(x_in, noise_labels=cnoise, conditional_inputs=cond_inputs)

                step_out = scheduler.step(model_out, t, tile_sample, generator=generator)
                tile_sample = step_out.prev_sample

            tile_sample = tile_sample / scheduler.config.sigma_data

            # Blend into output canvas
            samples_out[..., i0:i1, j0:j1] += tile_sample * weights
            samples_w[..., i0:i1, j0:j1] += weights

    return samples_out / samples_w
