import os
from typing import Optional, Tuple

import numpy as np
import torch
from flask import Flask, Response, jsonify, request

from terrain_diffusion.inference.world_pipeline import WorldPipeline

app = Flask(__name__)

_PIPELINE: Optional[WorldPipeline] = None


def _select_device() -> str:
    env_device = os.getenv("TERRAIN_DEVICE")
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_pipeline() -> WorldPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    _PIPELINE = WorldPipeline(
        'world_mc.h5', device='cuda', seed=1, log_mode='debug',
        drop_water_pct=0.5,
        frequency_mult=[2.0, 2.0, 2.0, 2.0, 2.0],
        cond_snr=[0.5, 0.5, 0.5, 0.5, 0.5],
        histogram_raw=[0.0, 0.0, 0.0, 2.0, 2.0],
        mode="a",
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


def _tensor_to_json(t: torch.Tensor):
    arr = t.detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "dtype": "float32",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "elev": arr.tolist(),
    }


def _transform_to_int16_bytes(t: torch.Tensor) -> Tuple[bytes, Tuple[int, int]]:
    arr = t.detach().cpu().numpy().astype(np.float32, copy=False)
    trans = np.sign(arr) * (arr ** 2)
    trans = np.floor(trans)
    trans = np.clip(trans, -32768, 32767).astype('<i2', copy=False)
    return trans.tobytes(), (int(trans.shape[0]), int(trans.shape[1]))


def _binary_response(elev: torch.Tensor, biome: Optional[torch.Tensor] = None) -> Response:
    payload, (h, w) = _transform_to_int16_bytes(elev)
    if biome is not None:
        biome_np = biome.detach().cpu().numpy().astype('<i2', copy=False)
        payload = payload + biome_np.tobytes()
    resp = Response(payload, mimetype="application/octet-stream")
    resp.headers["X-Height"] = str(h)
    resp.headers["X-Width"] = str(w)
    resp.headers["X-Dtype"] = "int16-le"
    resp.headers["X-Transform"] = "signed_square_floor"
    return resp


# Minimal mapping needed by the classifier (generic overworld biomes)
_BIOME_ID = {
    "plains": 1,
    "snowy_plains": 3,
    "desert": 5,
    "swamp": 6,
    "forest": 8,
    "taiga": 15,
    "snowy_taiga": 16,
    "savanna": 17,
    "windswept_hills": 19,
    "jungle": 23,
    "badlands": 26,
    "meadow": 29,
    "grove": 31,
    "snowy_slopes": 32,
    "frozen_peaks": 33,
    "stony_peaks": 35,
}


def _classify_biome(elev: torch.Tensor, climate: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Rule-based classifier using:
      - elevation (meters approximated as elev**2 for positive elev)
      - BIO1: local temperature (deg C) adjusted by lapse rate
      - BIO4: temperature seasonality (std * 100) [approx]
      - BIO12: annual precipitation (mm/yr)
      - BIO15: precipitation coefficient of variation
    Returns tensor (H, W) with int16 biome ids.
    """
    device = elev.device
    h, w = int(elev.shape[0]), int(elev.shape[1])

    if climate is None or climate.shape[0] < 4:
        return torch.full((h, w), _BIOME_ID["plains"], dtype=torch.int16, device=device)

    # Convert signed-sqrt elevation back to meters (retain sign for below sea level)
    alt_m_signed = torch.sign(elev) * torch.square(torch.abs(elev))
    alt_m = torch.clamp(alt_m_signed, min=0.0)

    temp = climate[0]
    # Temperature seasonality: stored as std (deg C). Convert-like thresholds accordingly.
    t_season = climate[1]  # deg C std (not x100)
    precip = climate[2]
    p_cv = climate[3]

    out = torch.full((h, w), _BIOME_ID["plains"], dtype=torch.int16, device=device)

    # Elevation bands (m)
    high_peaks = alt_m > 3000.0
    mountains = alt_m > 1800.0
    subalpine = (alt_m > 1200.0) & (alt_m <= 1800.0)
    lowland = alt_m < 200.0

    # Temperature classes (deg C)
    cold = temp < -5.0
    cool = (temp >= -5.0) & (temp < 8.0)
    temperate = (temp >= 8.0) & (temp < 18.0)
    warm = (temp >= 18.0) & (temp < 24.0)
    hot = temp >= 24.0

    # Precip bands (mm/yr)
    arid = precip < 250.0
    semiarid = (precip >= 250.0) & (precip < 500.0)
    moderate_rain = (precip >= 500.0) & (precip < 1000.0)
    wet = (precip >= 1000.0) & (precip < 2000.0)
    very_wet = precip >= 2000.0

    # Seasonality
    seasonal_precip = p_cv >= 55.0
    # Reasonable thresholds for std in deg C (not x100):
    high_temp_seasonality = t_season >= 7.0
    low_temp_seasonality = t_season < 3.0

    # Oceans (below sea level). Pick ocean temperature class by local temp.
    sea = alt_m_signed < -0.5  # allow small numerical noise around 0
    if sea.any():
        warm_ocean = temp >= 24.0
        cold_ocean = temp <= 5.0
        frozen_ocean = temp <= -2.0
        generic_ocean = ~(warm_ocean | cold_ocean | frozen_ocean)
        out[sea & frozen_ocean] = 48  # frozen_ocean
        out[sea & warm_ocean] = 41    # warm_ocean
        out[sea & cold_ocean & ~frozen_ocean] = 46  # cold_ocean
        out[sea & generic_ocean] = 44  # ocean

    # Peaks and slopes
    mask = high_peaks & cold
    out[mask] = _BIOME_ID["frozen_peaks"]

    mask = high_peaks & ~cold
    out[mask] = _BIOME_ID["stony_peaks"]

    mask = mountains & cold
    out[mask] = _BIOME_ID["snowy_slopes"]

    # Subalpine: meadows and windswept hills
    mask = subalpine & (cool | temperate) & (wet | moderate_rain)
    out[mask] = _BIOME_ID["meadow"]

    mask = subalpine & ~(wet | moderate_rain)
    out[mask] = _BIOME_ID["windswept_hills"]

    # Hot-dry biomes
    mask = hot & (arid | (semiarid & high_temp_seasonality))
    out[mask] = _BIOME_ID["desert"]

    mask = (warm | hot) & semiarid & (high_temp_seasonality | seasonal_precip) & ~lowland
    out[mask] = _BIOME_ID["badlands"]

    # Tropical wet
    mask = hot & (wet | very_wet) & (t_season < 600.0)
    out[mask] = _BIOME_ID["jungle"]

    # Seasonal tropics
    mask = (warm | hot) & (semiarid | moderate_rain) & seasonal_precip & (t_season >= 400.0)
    out[mask] = _BIOME_ID["savanna"]

    # Warm wetlands
    mask = warm & (wet | very_wet) & seasonal_precip & lowland
    out[mask] = _BIOME_ID["swamp"]

    # Snowy/cold
    mask = cold & (moderate_rain | wet | very_wet) & ~mountains
    out[mask] = _BIOME_ID["snowy_taiga"]

    mask = cold & (arid | semiarid)
    out[mask] = _BIOME_ID["snowy_plains"]

    # Boreal/temperate
    mask = cool & (moderate_rain | wet | very_wet)
    out[mask] = _BIOME_ID["taiga"]

    mask = temperate & (moderate_rain | wet | very_wet)
    out[mask] = _BIOME_ID["forest"]

    # Windswept temperate highlands fallback
    mask = temperate & (subalpine | mountains) & (arid | semiarid | moderate_rain)
    out[mask] = _BIOME_ID["windswept_hills"]

    return out


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/90")
def elev_90():
    try:
        i1, j1, i2, j2 = _parse_quad()
        world = _get_pipeline()
        out = world.get_90(i1, j1, i2, j2, with_climate=True)
        elev = out["elev"]
        climate = out.get("climate")
        biome = _classify_biome(elev, climate)
        if request.args.get("format") == "json":
            return jsonify(_tensor_to_json(elev))
        return _binary_response(elev, biome=biome)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/45")
def elev_45():
    try:
        i1, j1, i2, j2 = _parse_quad()
        world = _get_pipeline()
        out = world.get_45(i1, j1, i2, j2, with_climate=True)
        elev = out["elev"]
        climate = out.get("climate")
        biome = _classify_biome(elev, climate)
        if request.args.get("format") == "json":
            return jsonify(_tensor_to_json(elev))
        return _binary_response(elev, biome=biome)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/22")
def elev_22():
    try:
        i1, j1, i2, j2 = _parse_quad()
        world = _get_pipeline()
        out = world.get_22(i1, j1, i2, j2, with_climate=True)
        elev = out["elev"]
        climate = out.get("climate")
        biome = _classify_biome(elev, climate)
        if request.args.get("format") == "json":
            return jsonify(_tensor_to_json(elev))
        return _binary_response(elev, biome=biome)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/11")
def elev_11():
    try:
        i1, j1, i2, j2 = _parse_quad()
        world = _get_pipeline()
        out = world.get_11(i1, j1, i2, j2, with_climate=True)
        elev = out["elev"]
        climate = out.get("climate")
        biome = _classify_biome(elev, climate)
        if request.args.get("format") == "json":
            return jsonify(_tensor_to_json(elev))
        return _binary_response(elev, biome=biome)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)

