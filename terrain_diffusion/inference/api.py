import json
import os
from typing import Optional, Tuple

import click

import numpy as np
import torch
from flask import Flask, Response, jsonify, request

from terrain_diffusion.inference.world_pipeline import WorldPipeline

app = Flask(__name__)

_PIPELINE: Optional[WorldPipeline] = None
_PIPELINE_CONFIG: dict = {}


def _select_device() -> str:
    env_device = os.getenv("TERRAIN_DEVICE")
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_pipeline() -> WorldPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    cfg = _PIPELINE_CONFIG
    _PIPELINE = WorldPipeline(
        cfg.get('hdf5_file', 'world.h5'),
        device=cfg.get('device') or _select_device(),
        seed=cfg.get('seed'),
        log_mode=cfg.get('log_mode', 'verbose'),
        drop_water_pct=cfg.get('drop_water_pct', 0.5),
        frequency_mult=cfg.get('frequency_mult', [1.0, 1.0, 1.0, 1.0, 1.0]),
        cond_snr=cfg.get('cond_snr', [0.5, 0.5, 0.5, 0.5, 0.5]),
        histogram_raw=cfg.get('histogram_raw', [0.0, 0.0, 0.0, 1.0, 1.5]),
        latents_batch_size=cfg.get('latents_batch_size', 4),
    )
    return _PIPELINE


def _parse_quad() -> Tuple[int, int, int, int]:
    def _get_int(name: str) -> int:
        val = request.args.get(name, type=int)
        if val is None:
            raise ValueError(f"Missing required query param '{name}'")
        return val

    i1 = _get_int("i1")
    j1 = _get_int("j1")
    i2 = _get_int("i2")
    j2 = _get_int("j2")
    if i2 <= i1 or j2 <= j1:
        raise ValueError("Expected i2>i1 and j2>j1")
    return i1, j1, i2, j2


def _elev_to_int16(elev: torch.Tensor) -> np.ndarray:
    """Convert signed-sqrt elevation to int16 (squared back to meters, clamped)."""
    arr = elev.detach().cpu().numpy().astype(np.float32, copy=False)
    trans = np.sign(arr) * (arr ** 2)
    trans = np.floor(trans)
    return np.clip(trans, -32768, 32767).astype('<i2', copy=False)


def _binary_response(elev: torch.Tensor, climate: Optional[torch.Tensor]) -> Response:
    """
    Binary response format:
      - elevation: int16 little-endian (H*W*2 bytes)
      - climate: 4 channels of float32 little-endian (H*W*4*4 bytes)
        [temp, t_season, precip, p_cv]
    """
    elev_i16 = _elev_to_int16(elev)
    h, w = elev_i16.shape
    payload = elev_i16.tobytes()

    if climate is not None and climate.shape[0] >= 4:
        # climate shape: (4, H, W) -> transpose to (H, W, 4) for interleaved layout
        climate_np = climate[:4].detach().cpu().numpy().astype('<f4', copy=False)
        climate_np = np.transpose(climate_np, (1, 2, 0))  # (H, W, 4)
        payload += climate_np.tobytes()

    resp = Response(payload, mimetype="application/octet-stream")
    resp.headers["X-Height"] = str(h)
    resp.headers["X-Width"] = str(w)
    resp.headers["X-Elev-Dtype"] = "int16-le"
    resp.headers["X-Elev-Transform"] = "signed_square_floor"
    resp.headers["X-Climate-Dtype"] = "float32-le"
    resp.headers["X-Climate-Channels"] = "temp,t_season,precip,p_cv"
    return resp


def _get_terrain(world: WorldPipeline, i1: int, j1: int, i2: int, j2: int, scale: int) -> dict:
    """
    Get terrain data at arbitrary scale.
    
    Args:
        world: WorldPipeline instance
        i1, j1, i2, j2: Coordinates in target (scaled) resolution
        scale: Scale factor relative to 90m (1.0 = 90m, 2.0 = 45m, etc.)
    
    Returns dict with 'elev' (H, W) and 'climate' (4, H, W) tensors.
    """
    if scale == 1:
        # Native 90m - just fetch directly
        out = world.get_90(i1, j1, i2, j2, with_climate=True)
        return {"elev": out["elev"], "climate": out.get("climate")}

    # Convert target coordinates to 90m space
    i1_90 = i1 // scale
    j1_90 = j1 // scale
    i2_90 = -(-i2 // scale)  # ceil division
    j2_90 = -(-j2 // scale)

    # Add 1 pixel padding for bilinear interpolation edge handling
    i1_90_pad = i1_90 - 1
    j1_90_pad = j1_90 - 1
    i2_90_pad = i2_90 + 1
    j2_90_pad = j2_90 + 1

    out_90 = world.get_90(i1_90_pad, j1_90_pad, i2_90_pad, j2_90_pad, with_climate=True)
    elev_90 = out_90["elev"]
    climate_90 = out_90.get("climate")

    # Compute output size
    out_h = i2 - i1
    out_w = j2 - j1

    # Upsample elevation using bilinear interpolation
    elev_up = torch.nn.functional.interpolate(
        elev_90.unsqueeze(0).unsqueeze(0),
        scale_factor=scale,
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Calculate crop indices
    pad_up = scale  # 1 pixel padding in 90m = scale pixels in upsampled space
    offset_i = i1 - i1_90 * scale
    offset_j = j1 - j1_90 * scale
    crop_i1 = pad_up + offset_i
    crop_j1 = pad_up + offset_j

    elev = elev_up[crop_i1:crop_i1 + out_h, crop_j1:crop_j1 + out_w]

    climate = None
    if climate_90 is not None:
        climate_up = torch.nn.functional.interpolate(
            climate_90.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        climate = climate_up[:, crop_i1:crop_i1 + out_h, crop_j1:crop_j1 + out_w]

    return {"elev": elev, "climate": climate}


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/terrain")
def terrain():
    """
    Get terrain data at arbitrary scale.
    
    Query params:
        i1, j1, i2, j2: Bounding box in target resolution coordinates
        scale: Integer scale factor relative to 90m (default: 1)
               1 = 90m, 2 = 45m, 4 = 22.5m, 8 = 11.25m, etc.
    
    Returns binary data:
        - elevation: int16-le (H*W*2 bytes), squared back to meters
        - climate: float32-le interleaved (H*W*4*4 bytes)
                   channels: temp, t_season, precip, p_cv
    """
    try:
        i1, j1, i2, j2 = _parse_quad()
        scale = request.args.get("scale", default=1, type=int)
        if scale < 1:
            raise ValueError("scale must be >= 1")

        world = _get_pipeline()
        out = _get_terrain(world, i1, j1, i2, j2, scale)
        return _binary_response(out["elev"], out.get("climate"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@click.command()
@click.option("--hdf5-file", default="world.h5", help="HDF5 file path")
@click.option("--seed", type=int, default=None, help="Random seed (default: from file or random)")
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--drop-water-pct", type=float, default=0.5, help="Drop water percentage")
@click.option("--frequency-mult", default="[1.0, 1.0, 1.0, 1.0, 1.0]", help="Frequency multipliers (JSON)")
@click.option("--cond-snr", default="[0.5, 0.5, 0.5, 0.5, 0.5]", help="Conditioning SNR (JSON)")
@click.option("--histogram-raw", default="[0.0, 0.0, 0.0, 1.0, 1.5]", help="Histogram raw values (JSON)")
@click.option("--latents-batch-size", type=int, default=4, help="Batch size for latent generation")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose", help="Logging mode")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Server port")
def main(hdf5_file, seed, device, drop_water_pct, frequency_mult, cond_snr, histogram_raw, latents_batch_size, log_mode, host, port):
    """Terrain API server"""
    global _PIPELINE_CONFIG
    _PIPELINE_CONFIG = {
        'hdf5_file': hdf5_file,
        'seed': seed,
        'device': device,
        'drop_water_pct': drop_water_pct,
        'frequency_mult': json.loads(frequency_mult),
        'cond_snr': json.loads(cond_snr),
        'histogram_raw': json.loads(histogram_raw),
        'latents_batch_size': latents_batch_size,
        'log_mode': log_mode,
    }
    app.run(host=host, port=port, debug=False, threaded=False)


if __name__ == "__main__":
    main()

