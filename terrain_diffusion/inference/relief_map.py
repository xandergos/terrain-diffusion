from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def _to_numpy(x: np.ndarray) -> np.ndarray:
    """Convert input to contiguous float32 numpy array."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x

def _biome_palette() -> np.ndarray:
    """Return (31,3) float32 RGB palette for Koppen-Geiger classes with natural tones.

    Index 0 is Unknown.
    """
    # Colors chosen for realism (earth tones), normalized to 0..1
    lut = np.array(
        [
            [0, 0, 0],          # 0: Unknown
            [16, 86, 24],       # 1: Af  Tropical rainforest - deep green
            [38, 120, 40],      # 2: Am  Tropical monsoon - lush green
            [187, 212, 92],     # 3: Aw  Tropical savannah - yellow-green
            [227, 192, 122],    # 4: BWh Arid desert hot - sand
            [217, 200, 163],    # 5: BWk Arid desert cold - pale sand
            [210, 168, 90],     # 6: BSh Steppe hot - ochre
            [203, 182, 136],    # 7: BSk Steppe cold - tan
            [176, 156, 78],     # 8: Csa Med dry hot summer - olive
            [162, 148, 84],     # 9: Csb Med dry warm summer - olive-brown
            [148, 140, 104],    # 10: Csc Med dry cold summer - muted olive
            [132, 178, 96],     # 11: Cwa Temp dry winter hot summer - light green
            [112, 164, 96],     # 12: Cwb Temp dry winter warm summer - green
            [96, 148, 96],      # 13: Cwc Temp dry winter cold summer - darker green
            [124, 186, 84],     # 14: Cfa Temp no dry hot summer - bright green
            [96, 168, 84],      # 15: Cfb Temp no dry warm summer - temperate green
            [76, 140, 76],      # 16: Cfc Temp no dry cold summer - dark green
            [120, 140, 160],    # 17: Dsa Cold dry summer hot summer - cool grey-green
            [108, 130, 150],    # 18: Dsb Cold dry summer warm summer - cool grey-green
            [96, 120, 140],     # 19: Dsc Cold dry summer cold summer - cool slate
            [88, 112, 132],     # 20: Dsd Cold dry summer very cold winter - slate
            [136, 152, 176],    # 21: Dwa Cold dry winter hot summer - cool blue-grey
            [112, 136, 168],    # 22: Dwb Cold dry winter warm summer - blue-grey
            [100, 120, 160],    # 23: Dwc Cold dry winter cold summer - blue slate
            [84, 104, 140],     # 24: Dwd Cold dry winter very cold winter - deep blue slate
            [120, 170, 120],    # 25: Dfa Cold no dry hot summer - mixed forest
            [96, 150, 120],     # 26: Dfb Cold no dry warm summer - boreal edge
            [72, 120, 110],     # 27: Dfc Cold no dry cold summer - boreal
            [64, 96, 108],      # 28: Dfd Cold no dry very cold winter - dark boreal
            [173, 180, 180],    # 29: ET Polar tundra - grey-green tundra
            [230, 238, 244],    # 30: EF Polar frost - ice/snow
        ],
        dtype=np.float32,
    ) / 255.0
    return lut


def get_relief_map(
    elevation: np.ndarray,
    climate: np.ndarray,
    biome: np.ndarray,
    flow: np.ndarray,
    *,
    azimuths: Tuple[float, float, float, float] = (315.0, 45.0, 135.0, 225.0),
    flow_threshold: float = 7,
    sigma_large: float = 6.0,
    sigma_small: float = 1.2,
    resolution: float=90,
    rgb: np.ndarray | None = None,
    relief: float = 1.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a GDAL-style shaded relief map using Matplotlib, with optional river overlay.

    Args:
        elevation: (H, W) float meters.
        climate: unused.
        biome: unused.
        flow: (H, W) flow accumulation; rivers drawn where flow > flow_threshold.
        flow_threshold: threshold for river mask.

    Returns:
        (fig, ax): Matplotlib figure and GeoAxes containing the rendered map.
    """
    elev = _to_numpy(elevation)

    assert elev.ndim == 2, "elevation must be (H, W)"
    H, W = elev.shape
    flow_np = None
    if flow is not None:
        flow_np = _to_numpy(flow)
        assert flow_np.shape == elev.shape, "flow must be (H, W) matching elevation"

    # Hillshade (GDAL-style) parameters
    azimuth_deg = float(azimuths[0]) if isinstance(azimuths, (tuple, list)) and len(azimuths) > 0 else 315.0
    altitude_deg = 45.0  # sun elevation angle

    # Replace NaNs before any processing
    elev_f32 = elev.astype(np.float32, copy=False)
    if np.isnan(elev_f32).any():
        median_val = float(np.nanmedian(elev_f32)) if np.isfinite(np.nanmedian(elev_f32)) else 0.0
        elev_f32 = np.nan_to_num(elev_f32, nan=median_val)

    def compute_hillshade(src: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(src)
        dy, dx = dy/(15 * resolution/90), dx/(15 * resolution/90)
        slope_rad = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
        aspect_rad = np.arctan2(dy, -dx)
        az_rad = np.deg2rad(azimuth_deg)
        alt_rad = np.deg2rad(altitude_deg)
        hs = (
            np.sin(alt_rad) * np.sin(slope_rad)
            + np.cos(alt_rad) * np.cos(slope_rad) * np.cos(az_rad - aspect_rad)
        )
        return np.clip(hs, 0.0, 1.0).astype(np.float32)

    # Multi-scale hillshade: emphasize large landforms, suppress pixel-scale roughness
    elev_large = gaussian_filter(elev_f32, sigma=sigma_large)
    elev_small = gaussian_filter(elev_f32, sigma=sigma_small)

    hs_large = compute_hillshade(elev_large)
    hs_small = compute_hillshade(elev_small)
    hillshade = np.clip(0.75 * hs_large + 0.25 * hs_small, 0.0, 1.0)
    hillshade = np.power(hillshade, 0.85)  # gentle gamma to lift broad features

    # Colorize elevation; this will be used where biome is unknown
    if rgb is None:
        land_elev = np.maximum(0, elev)
        vmin, vmax = float(np.nanmin(land_elev)), float(np.nanmax(land_elev))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            vmin, vmax = 0.0, 1.0
        norm = (land_elev - vmin) / (vmax - vmin + 1e-8)
        cmap = plt.get_cmap("terrain")
        rgb = cmap(np.clip(norm**0.7, 0.0, 1.0))[..., :3].astype(np.float32)

    # Base RGB: prefer biome colors when available, otherwise elevation colormap
    base_rgb = rgb
    if biome is not None:
        b_idx = _to_numpy(biome).astype(np.int32, copy=False)
        if b_idx.shape == elev.shape:
            lut = _biome_palette()
            b_idx = np.clip(b_idx, 0, lut.shape[0] - 1)
            mask = b_idx > 0
            if np.any(mask):
                biome_rgb = lut[b_idx]
                mask3 = mask[..., None]
                base_rgb = np.where(mask3, biome_rgb, base_rgb)

    # GDAL-like intensity blend (ambient term + directional light)
    intensity = 0.35 + 0.65 * hillshade  # slightly higher ambient to reduce ragged contrast
    shaded_rgb = np.clip(base_rgb * (relief * intensity + (1 - relief))[..., None], 0.0, 1.0)
    shaded_rgb[np.isnan(elev)] = np.nan

    # Optional blue river overlay where flow exceeds threshold
    if flow_np is not None:
        river_mask = flow_np > float(flow_threshold)
        if np.any(river_mask):
            river_color = np.array([0.100, 0.450, 0.850], dtype=np.float32)
            river_alpha = 0.75
            shaded_rgb[river_mask] = (
                (1.0 - river_alpha) * shaded_rgb[river_mask]
                + river_alpha * river_color[None, :]
            )

    # Ocean coloring: fade from light blue (coast) to dark blue (deep ocean)
    ocean_mask = elev_f32 < 0.0
    if np.any(ocean_mask):
        depth = -elev_f32  # positive depth below sea level
        max_depth = 10_000.0
        if max_depth > 0.0:
            t = np.zeros_like(elev_f32, dtype=np.float32)
            t[ocean_mask] = np.clip(depth[ocean_mask] / max_depth, 0.0, 1.0)
            # Bias toward deeper blue sooner near the coast
            t = t ** 0.7
            t3 = t[..., None]
            # More saturated blues
            coast_color = np.array([0.68, 0.88, 1.00], dtype=np.float32)  # lighter, bluer coast
            deep_color = np.array([0.00, 0.10, 0.45], dtype=np.float32)   # deeper blue
            ocean_rgb = (1.0 - t3) * coast_color + t3 * deep_color
            shaded_rgb = np.where(ocean_mask[..., None], ocean_rgb, shaded_rgb)
            
    return shaded_rgb


__all__ = ["get_relief_map"]


