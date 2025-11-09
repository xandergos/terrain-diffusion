import torch
import math
from typing import Optional, Callable

from terrain_diffusion.training.evaluation import _linear_weight_window, _tile_starts


@torch.no_grad()
def sample_autoencoder_tiled(
    model,
    images: torch.Tensor,
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    *,
    cond_img: Optional[torch.Tensor] = None,
    conditional_inputs=None,
    use_mode: bool = False,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    """Reconstruct inputs with the autoencoder using spatial tiling and blending.

    If tile_size is None, uses non-tiled full-image sampling (tile_size = image size).
    The returned tensor has shape [B, model.config.out_channels, H, W].
    """
    conditional_inputs = conditional_inputs or []

    b, _, h, w = images.shape
    if tile_size is None:
        tile_size = images.shape[-1]
    if tile_stride is None:
        tile_stride = tile_size
    device, dtype = images.device, images.dtype
    if weight_window_fn is None:
        weight_window_fn = _linear_weight_window
    weights = weight_window_fn(tile_size, device, dtype)

    out_channels = int(getattr(model.config, 'out_channels', images.shape[1]))
    output = torch.zeros((b, out_channels, h, w), device=device, dtype=dtype)
    output_w = torch.zeros_like(output)

    encoder_input = images if cond_img is None else torch.cat([images, cond_img], dim=1)

    h_starts = _tile_starts(h, tile_size, tile_stride)
    w_starts = _tile_starts(w, tile_size, tile_stride)

    for i0 in h_starts:
        for j0 in w_starts:
            i1, j1 = i0 + tile_size, j0 + tile_size

            tile_in = encoder_input[..., i0:i1, j0:j1]
            means, logvars = model.preencode(tile_in, conditional_inputs)
            latent = model.postencode(means, logvars, use_mode=use_mode)
            tile_out = model.decode(latent)

            output[..., i0:i1, j0:j1] += tile_out * weights
            output_w[..., i0:i1, j0:j1] += weights

    return output / output_w


@torch.no_grad()
def decode_autoencoder_latents_tiled(
    model,
    latents: torch.Tensor,
    tile_size: Optional[int] = None,
    tile_stride: Optional[int] = None,
    *,
    weight_window_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode pre-computed latents with optional spatial tiling and blending.

    If tile_size is None, decodes the full latent tensor in one pass.
    """
    b, _, lh, lw = latents.shape
    device, dtype = latents.device, latents.dtype

    if tile_size is None:
        return model.decode(latents)

    if tile_stride is None:
        tile_stride = tile_size

    if weight_window_fn is None:
        weight_window_fn = _linear_weight_window
    weights = weight_window_fn(tile_size, device, dtype)

    scale_up = 8
    out_h = lh * scale_up
    out_w = lw * scale_up

    out_channels = int(getattr(model.config, 'out_channels', None) or getattr(model.config, 'in_channels', 1))
    output = torch.zeros((b, out_channels, out_h, out_w), device=device, dtype=dtype)
    output_w = torch.zeros_like(output)

    h_starts = _tile_starts(out_h, tile_size, tile_stride)
    w_starts = _tile_starts(out_w, tile_size, tile_stride)

    for i0 in h_starts:
        for j0 in w_starts:
            i1, j1 = i0 + tile_size, j0 + tile_size

            li0 = i0 // scale_up
            lj0 = j0 // scale_up
            li_len = math.ceil(tile_size / scale_up)
            lj_len = math.ceil(tile_size / scale_up)
            li1 = min(lh, li0 + li_len)
            lj1 = min(lw, lj0 + lj_len)

            tile_lat = latents[..., li0:li1, lj0:lj1]
            tile_out = model.decode(tile_lat)

            i_off = i0 - li0 * scale_up
            j_off = j0 - lj0 * scale_up
            tile_slice = tile_out[..., i_off:i_off + (i1 - i0), j_off:j_off + (j1 - j0)]

            output[..., i0:i1, j0:j1] += tile_slice * weights
            output_w[..., i0:i1, j0:j1] += weights

    return output / output_w


