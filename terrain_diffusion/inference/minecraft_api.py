import os
from typing import Optional, Tuple

import click

import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType

from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path
from terrain_diffusion.common.cli_helpers import parse_kwargs, parse_cache_size

app = Flask(__name__)

_PIPELINE: Optional[WorldPipeline] = None
_PIPELINE_CONFIG: dict = {}


def _select_device() -> str:
    env_device = os.getenv("TERRAIN_DEVICE")
    if env_device:
        return env_device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("Warning: Using CPU (CUDA not available).")
    return dev


def _get_pipeline() -> WorldPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    cfg = _PIPELINE_CONFIG
    device = cfg.get('device') or _select_device()
    caching_strategy = cfg.get('caching_strategy', 'indirect')
    _PIPELINE = WorldPipeline.from_pretrained(
        cfg.get('model_path', 'xandergos/terrain-diffusion-90m'),
        seed=cfg.get('seed'),
        latents_batch_size=cfg.get('latents_batch_size', 4),
        log_mode=cfg.get('log_mode', 'verbose'),
        torch_compile=cfg.get('torch_compile', False),
        dtype=cfg.get('dtype'),
        caching_strategy=caching_strategy,
        **cfg.get('kwargs', {}),
    )
    _PIPELINE.to(device)
    hdf5_file = cfg.get('hdf5_file')
    if caching_strategy == 'direct':
        _PIPELINE.bind(hdf5_file=hdf5_file)
    else:
        _PIPELINE.bind(hdf5_file or 'TEMP')
    print(f"World seed: {_PIPELINE.seed}")
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
    trans = np.floor(arr)
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
    # Sparse forest variants (trees_forest density)
    "forest_sparse": 108,
    "taiga_sparse": 115,
    "snowy_taiga_sparse": 116,
}

# Perlin noise for small temperature perturbations used by biome classifier
_TEMP_NOISE = FastNoiseLite(seed=12345)
_TEMP_NOISE.noise_type = NoiseType.NoiseType_Perlin
_TEMP_NOISE.frequency = 1.0 / 500.0  # coarse ~500-block wavelength
_TEMP_NOISE.fractal_type = FractalType.FractalType_FBm
_TEMP_NOISE.fractal_octaves = 3
_TEMP_NOISE.fractal_lacunarity = 2.0
_TEMP_NOISE.fractal_gain = 0.5

_TEMP_NOISE_FINE = FastNoiseLite(seed=54321)
_TEMP_NOISE_FINE.noise_type = NoiseType.NoiseType_Perlin
_TEMP_NOISE_FINE.frequency = 1.0 / 128.0  # finer ~128-block wavelength
_TEMP_NOISE_FINE.fractal_type = FractalType.FractalType_FBm
_TEMP_NOISE_FINE.fractal_octaves = 2
_TEMP_NOISE_FINE.fractal_lacunarity = 2.0
_TEMP_NOISE_FINE.fractal_gain = 0.5

# Perlin noise for small precipitation perturbations used by biome classifier
_PRECIP_NOISE = FastNoiseLite(seed=12345)
_PRECIP_NOISE.noise_type = NoiseType.NoiseType_Perlin
_PRECIP_NOISE.frequency = 1.0 / 500.0  # coarse ~500-block wavelength
_PRECIP_NOISE.fractal_type = FractalType.FractalType_FBm
_PRECIP_NOISE.fractal_octaves = 5
_PRECIP_NOISE.fractal_lacunarity = 2.0
_PRECIP_NOISE.fractal_gain = 0.5

# Perlin noise for small temperature perturbations used by biome classifier
_SNOW_NOISE = FastNoiseLite(seed=12345)
_SNOW_NOISE.noise_type = NoiseType.NoiseType_Perlin
_SNOW_NOISE.frequency = 1.0 / 500.0  # coarse ~500-block wavelength
_SNOW_NOISE.fractal_type = FractalType.FractalType_FBm
_SNOW_NOISE.fractal_octaves = 3
_SNOW_NOISE.fractal_lacunarity = 2.0
_SNOW_NOISE.fractal_gain = 0.5

_SNOW_NOISE_FINE = FastNoiseLite(seed=54321)
_SNOW_NOISE_FINE.noise_type = NoiseType.NoiseType_Perlin
_SNOW_NOISE_FINE.frequency = 1.0 / 128.0  # finer ~128-block wavelength
_SNOW_NOISE_FINE.fractal_type = FractalType.FractalType_FBm
_SNOW_NOISE_FINE.fractal_octaves = 2
_SNOW_NOISE_FINE.fractal_lacunarity = 2.0
_SNOW_NOISE_FINE.fractal_gain = 0.5

# Perlin noise for elevation detail (adds texture to smooth upsampled terrain)
# Coarse layer for larger bumps/ridges
_ELEV_NOISE_COARSE = FastNoiseLite(seed=99999)
_ELEV_NOISE_COARSE.noise_type = NoiseType.NoiseType_Perlin
_ELEV_NOISE_COARSE.frequency = 1.0 / 24.0  # ~24-block wavelength
_ELEV_NOISE_COARSE.fractal_type = FractalType.FractalType_FBm
_ELEV_NOISE_COARSE.fractal_octaves = 3
_ELEV_NOISE_COARSE.fractal_lacunarity = 2.0
_ELEV_NOISE_COARSE.fractal_gain = 0.5

# Fine layer for small-scale roughness
_ELEV_NOISE_FINE = FastNoiseLite(seed=88888)
_ELEV_NOISE_FINE.noise_type = NoiseType.NoiseType_Perlin
_ELEV_NOISE_FINE.frequency = 1.0 / 6.0  # ~6-block wavelength for block-scale detail
_ELEV_NOISE_FINE.fractal_type = FractalType.FractalType_FBm
_ELEV_NOISE_FINE.fractal_octaves = 2
_ELEV_NOISE_FINE.fractal_lacunarity = 2.0
_ELEV_NOISE_FINE.fractal_gain = 0.6  # slightly more high-freq contribution


def _compute_climate_vars(temp: torch.Tensor, t_season: torch.Tensor, 
                          precip: torch.Tensor, p_cv: torch.Tensor) -> dict:
    """
    Derive ecologically meaningful variables from raw climate data.
    
    Args:
        temp: Mean annual temperature (°C)
        t_season: Temperature seasonality (std * 100, so 400 = 4°C std)
        precip: Annual precipitation (mm/yr)
        p_cv: Precipitation coefficient of variation (%)
    
    Returns dict with:
        - pet: Potential evapotranspiration estimate (mm/yr)
        - aridity_index: precip/PET ratio (>1 humid, <0.2 arid)
        - tree_moisture: Aridity index penalized by precip seasonality
        - growing_season: Days per year with temp > 5°C (sinusoidal model)
        - frost_free: Whether mean temp suggests minimal hard frost
        - tropical: Hot year-round with low temp seasonality
    """
    # Convert t_season from std*100 to actual std in °C
    t_std = t_season / 100.0
    
    # Thornthwaite-inspired quadratic fit to empirical PET data
    # Use effective temp to account for seasonal concentration of evaporation
    # In continental climates, evaporation is concentrated in warm summer months
    t_eff = (temp + 0.5 * t_std).clamp(min=0)
    pet = torch.clamp(250.0 + 25.0 * t_eff + 0.7 * t_eff.pow(2), min=250.0)
    
    # Aridity index: >1 humid, 0.5-1 subhumid, 0.2-0.5 semiarid, <0.2 arid
    aridity_index = precip / pet.clamp(min=1)
    
    # Penalize high precip seasonality (dry season stress reduces effective moisture)
    # Monsoon forests ~35% less productive than equatorial forests at same annual precip
    seasonality_penalty = 1.0 - 0.35 * (p_cv / 100.0).clamp(max=1.0)
    tree_moisture = aridity_index * seasonality_penalty
    
    # Growing season using sinusoidal temperature model
    # Assumes T(day) = T_mean + amplitude * sin(2π*day/365)
    # For sine wave: std = amplitude / sqrt(2), so amplitude ≈ std * 1.414
    amplitude = t_std * 1.414
    
    # Days with T > threshold: solve sin(θ) > (threshold - T_mean) / amplitude
    # Result: days = 365 * (0.5 - arcsin(x)/π) where x = (threshold - T_mean) / amplitude
    threshold = 5.0  # standard threshold for tree growth
    x = (threshold - temp) / amplitude.clamp(min=0.1)
    
    growing_season = torch.where(
        x <= -1.0,  # threshold below minimum temp - year-round growing
        torch.full_like(temp, 365.0),
        torch.where(
            x >= 1.0,  # threshold above maximum temp - no growing season
            torch.zeros_like(temp),
            365.0 * (0.5 - torch.asin(x.clamp(-1.0, 1.0)) / 3.14159)
        )
    )
    
    coldest_month = temp - 2.0 * t_std
    
    # Frost classification
    frost_free = temp >= 10.0  # minimal frost risk
    hard_frost = coldest_month < -10.0  # severe winter
    
    # Tropical: warm year-round, low temp seasonality (std < 5°C)
    tropical = (temp >= 18.0) & (t_std < 5.0)
    
    return {
        'pet': pet,
        'aridity_index': aridity_index,
        'tree_moisture': tree_moisture,
        'growing_season': growing_season,
        'frost_free': frost_free,
        'hard_frost': hard_frost,
        'tropical': tropical,
        'coldest_month': coldest_month,
    }

def _get_upsampled(world: WorldPipeline, i1: int, j1: int, i2: int, j2: int, scale: int, noise_scale: float = 1.0, pixel_size_m: float = 90.0) -> dict:
    """
    Get native resolution data and upsample by `scale` factor using bilinear interpolation.
    Coordinates are in the target (upsampled) resolution.
    Adds 2 pixel padding in native space to ensure correct edge interpolation and slope calculation.
    Optionally adds Perlin noise scaled by local relief to restore detail lost in upsampling.
    """
    # Convert to native resolution coordinates
    i1_native = i1 // scale
    j1_native = j1 // scale
    i2_native = -(-i2 // scale)  # ceil division
    j2_native = -(-j2 // scale)

    # Add 2 pixel padding: 1 for bilinear interpolation + 1 for slope calculation
    i1_native_pad = i1_native - 2
    j1_native_pad = j1_native - 2
    i2_native_pad = i2_native + 2
    j2_native_pad = j2_native + 2

    out_native = world.get(i1_native_pad, j1_native_pad, i2_native_pad, j2_native_pad, with_climate=True)
    elev_native = out_native["elev"]
    climate_native = out_native.get("climate")

    # Upsample elevation
    elev_up = torch.nn.functional.interpolate(
        elev_native.unsqueeze(0).unsqueeze(0),
        scale_factor=scale,
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Calculate crop indices: 2 pixel padding in native = 2*scale pixels in upsampled space
    pad_up = 2 * scale
    offset_i = i1 - i1_native * scale
    offset_j = j1 - j1_native * scale
    crop_i1 = pad_up + offset_i
    crop_j1 = pad_up + offset_j
    crop_i2 = crop_i1 + (i2 - i1)
    crop_j2 = crop_j1 + (j2 - j1)

    elev_smooth = elev_up[crop_i1:crop_i2, crop_j1:crop_j2]
    # Include 1 pixel padding for slope calculation
    elev_padded = elev_up[crop_i1-1:crop_i2+1, crop_j1-1:crop_j2+1]

    climate = None
    if climate_native is not None:
        climate_up = torch.nn.functional.interpolate(
            climate_native.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        climate = climate_up[:, crop_i1:crop_i2, crop_j1:crop_j2]

    # Add Perlin noise to restore detail lost in upsampling
    if noise_scale > 0:
        elev = elev_smooth.clone()
        h, w = elev.shape
        if h > 0 and w > 0:
            x = np.arange(j1, j1 + w, dtype=np.float32)
            y = np.arange(i1, i1 + h, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            coords = np.array([xx.ravel(), yy.ravel()], dtype=np.float32)
            
            # Generate coarse + fine noise layers
            noise_coarse = _ELEV_NOISE_COARSE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
            noise_fine = _ELEV_NOISE_FINE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
            noise_coarse_t = torch.from_numpy(noise_coarse).to(device=elev.device, dtype=elev.dtype)
            noise_fine_t = torch.from_numpy(noise_fine).to(device=elev.device, dtype=elev.dtype)
            
            # Compute local relief using a Sobel-like gradient on the smooth elevation
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=elev.dtype, device=elev.device).view(1, 1, 3, 3) / 8.0
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=elev.dtype, device=elev.device).view(1, 1, 3, 3) / 8.0
            dx = torch.nn.functional.conv2d(elev_padded.unsqueeze(0).unsqueeze(0), sobel_x, padding=0).squeeze()
            dy = torch.nn.functional.conv2d(elev_padded.unsqueeze(0).unsqueeze(0), sobel_y, padding=0).squeeze()
            gradient = torch.sqrt(dx**2 + dy**2)
            
            # Scale noise amplitude by local gradient (steeper = more noise)
            # Gamma > 1 concentrates noise on very steep slopes
            # Normalize gradient threshold by pixel size (40m/pixel at 90m resolution)
            slope_factor = (gradient / (40.0 * pixel_size_m / 90.0)).clamp(0.0, 1.0).pow(1.5)
            amp_coarse = noise_scale * 100.0 * slope_factor * pixel_size_m / world.native_resolution
            amp_fine = noise_scale * 70.0 * slope_factor * pixel_size_m / world.native_resolution
            
            # Apply noise only to land (elev >= 0)
            is_land = elev_smooth >= 0.0
            elev = elev + (noise_coarse_t * amp_coarse + noise_fine_t * amp_fine) * is_land.float()
    else:
        elev = elev_smooth

    return {"elev": elev, "elev_smooth": elev_smooth, "climate": climate, "elev_padded": elev_padded}


def _classify_biome(elev: torch.Tensor, climate: Optional[torch.Tensor], i0: int, j0: int,
                    elev_padded: torch.Tensor, pixel_size_m: float = 90.0) -> torch.Tensor:
    """
    Rule-based biome classifier using tree coverage, snow coverage, and climate.
    
    Steps:
      1. Compute derived climate variables (aridity index, tree moisture)
      2. Classify tree coverage: none / sparse / forest / dense / rainforest
      3. Classify snow coverage using coldest month estimate + noise
      4. Assign biomes using disjoint masks based on tree + snow + elevation
    
    Returns tensor (H, W) with int16 biome ids.
    """
    device = elev.device
    h, w = int(elev.shape[0]), int(elev.shape[1])

    if climate is None or climate.shape[0] < 4:
        return torch.full((h, w), _BIOME_ID["plains"], dtype=torch.int16, device=device)

    # === ELEVATION ===
    alt_m = torch.clamp(elev, min=0.0)

    # === RAW CLIMATE ===
    temp = climate[0]
    t_season = climate[1]
    precip = torch.clamp(climate[2], min=0.0)
    p_cv = climate[3]

    # === ADD NOISE TO CLIMATE VARIABLES ===
    if h > 0 and w > 0:
        x = np.arange(j0, j0 + w, dtype=np.float32)
        y = np.arange(i0, i0 + h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        coords = np.array([xx.ravel(), yy.ravel()], dtype=np.float32)
        
        # Temperature noise
        temp_noise_coarse = _TEMP_NOISE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
        temp_noise_fine = _TEMP_NOISE_FINE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
        temp_noise = 0.4 * temp_noise_coarse + 0.2 * temp_noise_fine
        temp = temp + torch.from_numpy(temp_noise).to(device=device, dtype=temp.dtype)
        
        # Precipitation noise (multiplicative for realistic relative variation ~±20%)
        precip_noise = _PRECIP_NOISE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
        precip_noise_factor = torch.from_numpy(1.0 + 0.2 * precip_noise).to(device=device, dtype=precip.dtype)
        precip = precip * precip_noise_factor
        
        # Snow noise (separate from temp noise for edge variation)
        snow_noise_coarse = _SNOW_NOISE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
        snow_noise_fine = _SNOW_NOISE_FINE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
        snow_noise = torch.from_numpy(3.0 * snow_noise_coarse + 2.0 * snow_noise_fine).to(device=device, dtype=temp.dtype)

    # === DERIVED CLIMATE VARIABLES ===
    cv = _compute_climate_vars(temp, t_season, precip, p_cv)
    tree_moisture = cv['tree_moisture']
    coldest_month = cv['coldest_month']
    tropical = cv['tropical']
    growing_season = cv['growing_season']

    # === SLOPE CALCULATION ===
    elev_m = elev_padded
    
    # Sobel kernels (normalized by 8 to get proper gradient estimate)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=elev.dtype, device=device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=elev.dtype, device=device).view(1, 1, 3, 3) / 8.0
    
    # Apply Sobel to get gradient in meters per pixel
    dx = torch.nn.functional.conv2d(elev_m.unsqueeze(0).unsqueeze(0), sobel_x, padding=0).squeeze()
    dy = torch.nn.functional.conv2d(elev_m.unsqueeze(0).unsqueeze(0), sobel_y, padding=0).squeeze()
    
    # Compute slope ratio (rise/run) - gradient in meters divided by pixel size
    gradient_m = torch.sqrt(dx**2 + dy**2)
    slope_ratio = gradient_m / pixel_size_m
    
    # Snow doesn't accumulate on slopes > ~38° (angle of repose for snow)
    # tan(38°) ≈ 0.78
    is_steep = slope_ratio > 0.78

    # === TREE SUITABILITY MODIFIERS ===
    # Growing season factor: trees need sufficient warm days to complete growth cycle
    # - <60 days: no trees (arctic/alpine tundra)
    # - 60-90 days: only hardiest shrubs/krummholz
    # - 90-120 days: sparse boreal conifers
    # - 120-150 days: full boreal forest
    # - 150+ days: temperate and tropical forests
    gs_min = 60.0   # below this, no trees
    gs_full = 150.0 # above this, full potential
    gs_factor = ((growing_season - gs_min) / (gs_full - gs_min)).clamp(0.0, 1.0)

    eff_tree_moisture = tree_moisture * gs_factor
    
    # === SLOPE BARE THRESHOLD (moisture-dependent) ===
    # Soil stability depends on vegetation root reinforcement:
    # - Arid zones (tree_moisture < 0.1): bare rock at ~35° (tan ≈ 0.70)
    #   Examples: Grand Canyon, Wadi Rum - no roots to stabilize soil
    # - Semi-arid (tree_moisture ~0.4): bare rock at ~42° (tan ≈ 0.90)
    #   Examples: Mediterranean gorges - shrubs provide some cohesion
    # - Humid zones (tree_moisture > 0.8): bare rock at ~50° (tan ≈ 1.19)
    #   Examples: Columbia River Gorge, Three Gorges - dense root networks
    bare_threshold_min = 0.7   # tan(35°) for arid regions
    bare_threshold_max = 1.19   # tan(50°) for humid regions
    moisture_factor = ((tree_moisture - 0.35) / 0.45).clamp(0.0, 1.0)
    bare_threshold = bare_threshold_min + (bare_threshold_max - bare_threshold_min) * moisture_factor

    # === TREE COVERAGE CLASSIFICATION ===
    # Based on effective moisture availability for tree growth
    # Thresholds extend UNEP aridity classification into humid zones:
    #   <0.2 arid (none), 0.2-0.5 semi-arid (sparse), 0.5-0.8 sub-humid (forest),
    #   0.8-1.3 humid (dense), >1.3 perhumid (rainforest)
    trees_none = eff_tree_moisture < 0.2
    # Distinguish treeless terrain by cause:
    #   - barren: too arid OR too cold for any vegetation → bare rock/stone
    #   - grass_capable: moderate conditions where grass survives
    # Barren if: hyper-arid (tree_moisture < 0.05) OR extreme cold (gs_factor < 0.15)
    # gs_factor < 0.15 means growing season < 73 days (high alpine/arctic)
    too_arid = tree_moisture < 0.05
    too_cold = growing_season < 60.0
    barren = too_arid | too_cold
    trees_sparse = ~trees_none & (eff_tree_moisture < 0.5)
    trees_forest = ~trees_none & (eff_tree_moisture >= 0.5) & (eff_tree_moisture < 0.8)
    trees_dense = ~trees_none & (eff_tree_moisture >= 0.8) & (eff_tree_moisture < 1.3)
    trees_rainforest = ~trees_none & (eff_tree_moisture >= 1.3)

    # Slope overrides: steep slopes prevent soil/root establishment
    # Medium slopes: sparse vegetation only (soil unstable for trees)
    # Bare slopes: no vegetation (cliff faces, no soil accumulation)
    # bare_threshold varies by moisture: 35° (arid) to 50° (humid)
    slope_medium = (slope_ratio >= 0.62) & (slope_ratio < bare_threshold)
    slope_bare = slope_ratio >= bare_threshold
    
    # Medium slopes: cap at sparse (demote forest/dense/rainforest to sparse)
    had_trees = trees_forest | trees_dense | trees_rainforest
    trees_sparse = trees_sparse | (slope_medium & had_trees)
    trees_forest = trees_forest & ~slope_medium
    trees_dense = trees_dense & ~slope_medium
    trees_rainforest = trees_rainforest & ~slope_medium
    
    # Bare slopes: no vegetation at all
    trees_none = trees_none | slope_bare
    trees_sparse = trees_sparse & ~slope_bare
    trees_forest = trees_forest & ~slope_bare
    trees_dense = trees_dense & ~slope_bare
    trees_rainforest = trees_rainforest & ~slope_bare

    # === SNOW COVERAGE CLASSIFICATION ===
    # Snow present based on mean annual temp (not coldest month) + noise
    # Threshold ~3°C: regions with mean temp below this typically have seasonal snow cover
    # Also requires minimum precipitation (~150 mm/yr) - very dry cold regions are polar deserts
    snow_temp = temp + snow_noise
    cold_enough = snow_temp < 0.0
    wet_enough = precip > 150.0
    would_have_snow = cold_enough & wet_enough
    has_snow = would_have_snow & ~is_steep

    # === ELEVATION BANDS ===
    is_ocean = elev < 0.0
    mountains = alt_m > 2500.0
    lowland = alt_m < 200.0

    # === TEMPERATURE BANDS ===
    frozen = temp < -5.0
    cold = (temp >= -5.0) & (temp < 5.0)
    cool = (temp >= 5.0) & (temp < 12.0)
    temperate = (temp >= 12.0) & (temp < 20.0)
    warm = (temp >= 20.0) & (temp < 26.0)
    hot = temp >= 26.0

    # === BIOME OUTPUT ===
    out = torch.full((h, w), _BIOME_ID["plains"], dtype=torch.int16, device=device)

    # Track unassigned pixels for disjoint assignment
    unassigned = torch.ones((h, w), dtype=torch.bool, device=device)

    # ============================================================
    # BRANCH 1: OCEAN (disjoint from all land)
    # ============================================================
    if is_ocean.any():
        ocean_frozen = is_ocean & frozen
        ocean_cold = is_ocean & cold & ~frozen
        ocean_warm = is_ocean & (warm | hot)
        ocean_normal = is_ocean & ~ocean_frozen & ~ocean_cold & ~ocean_warm
        
        out[ocean_frozen] = 48   # frozen_ocean
        out[ocean_cold] = 46     # cold_ocean
        out[ocean_warm] = 41     # warm_ocean
        out[ocean_normal] = 44   # ocean
        unassigned[is_ocean] = False

    # ============================================================
    # BRANCH 2: MOUNTAINS (>2500m)
    # Peaks emerge from bare slopes, other biomes from vegetation
    # ============================================================
    mtn_mask = mountains & unassigned
    if mtn_mask.any():
        # PEAKS: bare rock slopes (too steep for soil)
        mtn_frozen_peak = mtn_mask & slope_bare & has_snow
        out[mtn_frozen_peak] = _BIOME_ID["frozen_peaks"]
        
        mtn_stony_peak = mtn_mask & slope_bare & ~has_snow
        out[mtn_stony_peak] = _BIOME_ID["stony_peaks"]
        
        # NON-PEAK MOUNTAINS: has some soil/vegetation potential
        mtn_soil = mtn_mask & ~slope_bare
        
        # Snowy + no trees → snowy slopes
        mtn_snowy_barren = mtn_soil & has_snow & trees_none
        out[mtn_snowy_barren] = _BIOME_ID["snowy_slopes"]
        
        # Snowy + sparse/forest → snowy taiga sparse
        mtn_snowy_forest_sparse = mtn_soil & has_snow & (trees_sparse | trees_forest)
        out[mtn_snowy_forest_sparse] = _BIOME_ID["snowy_taiga_sparse"]
        
        # Snowy + dense/rainforest → snowy taiga
        mtn_snowy_forest_dense = mtn_soil & has_snow & (trees_dense | trees_rainforest)
        out[mtn_snowy_forest_dense] = _BIOME_ID["snowy_taiga"]
        
        # No snow + no trees: windswept (extreme arid) vs grove (cold steppe) vs plains
        mtn_windswept = mtn_soil & ~has_snow & trees_none & barren
        out[mtn_windswept] = _BIOME_ID["windswept_hills"]
        # Semi-arid mountains = grove (brown steppe)
        mtn_cold_steppe = mtn_soil & ~has_snow & trees_none & ~barren & ((tree_moisture < 0.35) | (precip < 350))
        out[mtn_cold_steppe] = _BIOME_ID["grove"]
        mtn_plains = mtn_soil & ~has_snow & trees_none & ~barren & ~mtn_cold_steppe
        out[mtn_plains] = _BIOME_ID["plains"]
        
        # No snow + sparse/forest → taiga sparse (mountain forest)
        mtn_taiga_sparse = mtn_soil & ~has_snow & (trees_sparse | trees_forest)
        out[mtn_taiga_sparse] = _BIOME_ID["taiga_sparse"]
        
        # No snow + dense/rainforest → taiga
        mtn_taiga_dense = mtn_soil & ~has_snow & (trees_dense | trees_rainforest)
        out[mtn_taiga_dense] = _BIOME_ID["taiga"]
        
        unassigned[mtn_mask] = False

    # ============================================================
    # BRANCH 3: LOWLAND + MIDLAND - primary biome classification
    # Uses tree coverage × snow × temperature
    # ============================================================
    land_mask = unassigned  # everything remaining

    # --- 3A: FROZEN/SNOWY + NO TREES ---
    snowy_barren = land_mask & has_snow & trees_none
    out[snowy_barren] = _BIOME_ID["snowy_plains"]
    land_mask = land_mask & ~snowy_barren

    # --- 3B: SNOWY + FOREST (snowy taiga) ---
    snowy_forest_sparse = land_mask & has_snow & (trees_sparse | trees_forest)
    out[snowy_forest_sparse] = _BIOME_ID["snowy_taiga_sparse"]
    snowy_forest_dense = land_mask & has_snow & (trees_dense | trees_rainforest)
    out[snowy_forest_dense] = _BIOME_ID["snowy_taiga"]
    land_mask = land_mask & ~(snowy_forest_sparse | snowy_forest_dense)

    # --- 3D: NO SNOW, NO TREES (desert/badlands/plains) ---
    dry_barren = land_mask & ~has_snow & trees_none
    # Hot + arid = desert
    desert_mask = dry_barren & (warm | hot)
    out[desert_mask] = _BIOME_ID["desert"]
    # Cool/cold + extreme arid = windswept (bare stone)
    windswept_mask = dry_barren & (cold | cool | temperate) & ~lowland & barren
    out[windswept_mask] = _BIOME_ID["windswept_hills"]
    # Semi-arid = grove (brown steppe) - aridity drives brown vs green grass
    # Use both tree_moisture AND precipitation floor - green grass needs ~350mm minimum
    cold_steppe = dry_barren & ((tree_moisture < 0.35) | (precip < 350)) & ~barren
    out[cold_steppe] = _BIOME_ID["grove"]
    # Moderate conditions = plains (green grassland)
    plains_barren = dry_barren & ~desert_mask & ~windswept_mask & ~cold_steppe
    out[plains_barren] = _BIOME_ID["plains"]
    land_mask = land_mask & ~dry_barren

    # --- 3E: NO SNOW, SPARSE/FOREST TREES → sparse variants ---
    sparse_forest_land = land_mask & ~has_snow & (trees_sparse | trees_forest)
    # Hot = jungle (no sparse variant)
    jungle_mask = sparse_forest_land & hot
    out[jungle_mask] = _BIOME_ID["jungle"]
    # Warm + trees_sparse (moisture-limited, not slope-demoted) = savanna
    savanna_mask = sparse_forest_land & warm & trees_sparse & ~slope_medium
    out[savanna_mask] = _BIOME_ID["savanna"]
    # Warm + trees_forest = forest sparse
    warm_forest = sparse_forest_land & warm & trees_forest
    out[warm_forest] = _BIOME_ID["forest_sparse"]
    # Temperate = forest sparse
    temp_sparse_forest = sparse_forest_land & temperate
    out[temp_sparse_forest] = _BIOME_ID["forest_sparse"]
    # Cool/cold = taiga sparse
    cool_sparse_forest = sparse_forest_land & (cool | cold)
    out[cool_sparse_forest] = _BIOME_ID["taiga_sparse"]
    land_mask = land_mask & ~sparse_forest_land

    # --- 3G: NO SNOW, DENSE FOREST ---
    dense_land = land_mask & ~has_snow & trees_dense
    # Hot dense = jungle
    jungle_dense = dense_land & hot
    out[jungle_dense] = _BIOME_ID["jungle"]
    # Warm dense lowland = swamp
    swamp_mask = dense_land & warm & lowland
    out[swamp_mask] = _BIOME_ID["swamp"]
    # Cool/cold dense = taiga
    taiga_dense = dense_land & (cool | cold) & ~jungle_dense & ~swamp_mask
    out[taiga_dense] = _BIOME_ID["taiga"]
    # Other warm/temperate dense = forest
    forest_dense = dense_land & ~jungle_dense & ~swamp_mask & ~taiga_dense
    out[forest_dense] = _BIOME_ID["forest"]
    land_mask = land_mask & ~dense_land

    # --- 3H: NO SNOW, RAINFOREST ---
    rain_land = land_mask & ~has_snow & trees_rainforest
    # Tropical rainforest = jungle
    jungle_rain = rain_land & (hot | (warm & tropical))
    out[jungle_rain] = _BIOME_ID["jungle"]
    # Lowland wet = swamp
    swamp_rain = rain_land & ~jungle_rain & lowland
    out[swamp_rain] = _BIOME_ID["swamp"]
    # Cool/cold rainforest = taiga
    taiga_rain = rain_land & (cool | cold) & ~jungle_rain & ~swamp_rain
    out[taiga_rain] = _BIOME_ID["taiga"]
    # Temperate rainforest = forest
    forest_rain = rain_land & ~jungle_rain & ~swamp_rain & ~taiga_rain
    out[forest_rain] = _BIOME_ID["forest"]
    land_mask = land_mask & ~rain_land

    # --- 3I: FALLBACK (any remaining unassigned land) ---
    out[land_mask] = _BIOME_ID["plains"]

    # --- BARE SLOPE OVERRIDE ---
    # Bare rock slopes at any elevation become peaks (already handled in mountains, 
    # but this catches lowland cliffs)
    lowland_bare = slope_bare & ~is_ocean & ~mountains
    out[lowland_bare & has_snow] = _BIOME_ID["frozen_peaks"]
    out[lowland_bare & ~has_snow] = _BIOME_ID["stony_peaks"]

    return out


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _parse_noise() -> float:
    """Parse noise query param. Defaults to 1.0. Accepts float values."""
    val = request.args.get("noise", "1.0")
    try:
        return float(val)
    except ValueError:
        return 1.0

def _get_base_pixel_size() -> float:
    return _get_pipeline().native_resolution


def _handle_1x():
    """Handler for 1x (base) resolution."""
    i1, j1, i2, j2 = _parse_quad()
    world = _get_pipeline()
    out_pad = world.get(i1 - 1, j1 - 1, i2 + 1, j2 + 1, with_climate=False)
    elev_padded = out_pad["elev"]
    out = world.get(i1, j1, i2, j2, with_climate=True)
    elev = out["elev"]
    climate = out.get("climate")
    biome = _classify_biome(elev, climate, i1, j1, elev_padded=elev_padded, pixel_size_m=_get_base_pixel_size())
    if request.args.get("format") == "json":
        return jsonify(_tensor_to_json(elev))
    return _binary_response(elev, biome=biome)


def _handle_upsampled(scale: int):
    """Handler for upsampled resolutions (2x, 4x, 8x)."""
    i1, j1, i2, j2 = _parse_quad()
    noise_scale = _parse_noise()
    world = _get_pipeline()
    pixel_size_m = _get_base_pixel_size() / scale
    out = _get_upsampled(world, i1, j1, i2, j2, scale=scale, noise_scale=noise_scale, pixel_size_m=pixel_size_m)
    elev = out["elev"]
    elev_smooth = out["elev_smooth"]
    climate = out.get("climate")
    elev_padded = out["elev_padded"]
    biome = _classify_biome(elev_smooth, climate, i1, j1, elev_padded=elev_padded, pixel_size_m=pixel_size_m)
    if request.args.get("format") == "json":
        return jsonify(_tensor_to_json(elev))
    return _binary_response(elev, biome=biome)


@app.get("/terrain")
def terrain():
    """
    Get terrain data at arbitrary scale.
    
    Query params:
        i1, j1, i2, j2: Bounding box in target resolution coordinates
        scale: Integer scale factor relative to native resolution (default: 1)
        noise: Noise scale factor (default: 1.0)
        format: 'json' for JSON output, otherwise binary
    """
    try:
        scale = request.args.get("scale", default=1, type=int)
        if scale < 1:
            raise ValueError("scale must be >= 1")
        if scale == 1:
            return _handle_1x()
        return _handle_upsampled(scale=scale)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Backwards compatibility endpoints
@app.get("/90")
def elev_1x():
    try:
        return _handle_1x()
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/45")
def elev_2x():
    try:
        return _handle_upsampled(scale=2)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/22")
def elev_4x():
    try:
        return _handle_upsampled(scale=4)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/11")
def elev_8x():
    try:
        return _handle_upsampled(scale=8)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@click.command()
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.option("--caching-strategy", type=click.Choice(["indirect", "direct"]), default="direct", help="Caching strategy: 'indirect' uses HDF5, 'direct' uses in-memory LRU cache")
@click.option("--hdf5-file", default=None, help="HDF5 file path (required for indirect caching, optional for direct)")
@click.option("--cache-size", default="100M", help="Cache size (e.g., 100M, 1G) for direct caching")
@click.option("--seed", type=int, default=None, help="Random seed (default: from file or random)")
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--batch-size", type=str, default="1,4", help="Batch size(s) for latent generation (e.g. '4' or '1,2,4,8')")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose", help="Logging mode")
@click.option("--compile/--no-compile", "torch_compile", default=True, help="Use torch.compile for faster inference")
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32", help="Model dtype")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Server port")
@click.option("--kwarg", "extra_kwargs", multiple=True, help="Additional key=value kwargs (e.g. --kwarg native_resolution=30)")
def main(model_path, hdf5_file, caching_strategy, cache_size, seed, device, batch_size, log_mode, torch_compile, dtype, host, port, extra_kwargs):
    """Minecraft terrain API server"""
    global _PIPELINE_CONFIG
    if caching_strategy == 'indirect' and hdf5_file is None:
        hdf5_file = 'TEMP'
    if hdf5_file is not None:
        hdf5_file = resolve_hdf5_path(hdf5_file)
    # Parse batch size(s)
    if ',' in batch_size:
        batch_sizes = [int(x.strip()) for x in batch_size.split(',')]
    else:
        batch_sizes = int(batch_size)
    # Normalize dtype
    if dtype == 'fp32':
        dtype = None
    _PIPELINE_CONFIG = {
        'model_path': model_path,
        'hdf5_file': hdf5_file,
        'caching_strategy': caching_strategy,
        'cache_limit': parse_cache_size(cache_size),
        'seed': seed,
        'device': device,
        'latents_batch_size': batch_sizes,
        'log_mode': log_mode,
        'torch_compile': torch_compile,
        'dtype': dtype,
        'kwargs': parse_kwargs(extra_kwargs),
    }
    _get_pipeline()  # Initialize pipeline upfront (triggers compilation)
    app.run(host=host, port=port, debug=False, threaded=False)


if __name__ == "__main__":
    main()

