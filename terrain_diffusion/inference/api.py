import os
from typing import Optional, Tuple

import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType

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
        frequency_mult=[1.0, 1.0, 1.0, 1.0, 1.0],
        cond_snr=[0.5, 0.5, 0.5, 0.5, 0.5],
        histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.5],
        mode="a",
        latents_batch_size=32,
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
        - growing_season: Approximate months with temp > 5°C
        - frost_free: Whether mean temp suggests minimal hard frost
        - tropical: Hot year-round with low temp seasonality
    """
    # Convert t_season from std*100 to actual std in °C
    t_std = t_season / 100.0
    
    # PET approximation (Hargreaves-inspired)
    # ~500mm at 10°C, ~1200mm at 25°C, minimal when frozen
    pet = torch.where(temp > 0,
                      50.0 * torch.pow(temp.clamp(min=1), 1.3),
                      torch.full_like(temp, 50.0))
    
    # Aridity index: >1 humid, 0.5-1 subhumid, 0.2-0.5 semiarid, <0.2 arid
    aridity_index = precip / pet.clamp(min=1)
    
    # Penalize high precip seasonality (dry season stress)
    seasonality_penalty = 1.0 - 0.25 * (p_cv / 100.0).clamp(max=1.0)
    tree_moisture = aridity_index * seasonality_penalty
    
    # Growing season length (crude: months where temp likely > 5°C)
    # Based on mean temp and seasonality
    # If mean=15°C and std=10°C, coldest month ~5°C -> ~12 months
    # If mean=5°C and std=15°C, only ~6 months above 5°C
    coldest_month = temp - 2.0 * t_std
    growing_season = torch.where(
        coldest_month >= 5.0, 
        torch.full_like(temp, 12.0),
        torch.where(
            temp >= 5.0,
            6.0 + 6.0 * (temp - 5.0) / (temp - coldest_month).clamp(min=1),
            torch.zeros_like(temp)
        )
    ).clamp(0, 12)
    
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

def _classify_biome(elev: torch.Tensor, climate: Optional[torch.Tensor], i0: int, j0: int) -> torch.Tensor:
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
    alt_m_signed = torch.sign(elev) * torch.square(torch.abs(elev))
    alt_m = torch.clamp(alt_m_signed, min=0.0)

    # === RAW CLIMATE ===
    temp = climate[0]
    t_season = climate[1]
    precip = climate[2]
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
        
        # Precipitation noise
        precip_noise = _PRECIP_NOISE.gen_from_coords(coords).astype(np.float32).reshape(h, w)
        precip = precip + torch.from_numpy(30.0 * precip_noise).to(device=device, dtype=precip.dtype)
        
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

    # Smooth modifiers for tree suitability
    # Temperature factor: no effect in normal forest temperatures, only penalize cold
    #  - 1.0 for temp >= 5°C
    #  - 0.0 at -8°C and below, linear ramp in between
    temp_factor = torch.where(
        temp >= 5.0,
        torch.ones_like(temp),
        ((temp - (-8.0)) / (5.0 - (-8.0))).clamp(0.0, 1.0),
    )
    # Altitude factor: mild penalty that only matters near very high elevations
    alt_mid = 2500.0
    alt_max = 4500.0
    alt_scale = ((alt_m - alt_mid) / (alt_max - alt_mid)).clamp(0.0, 1.0)
    alt_factor = 1.0 - 0.3 * alt_scale  # down to 70% effective moisture near 4500m

    eff_tree_moisture = tree_moisture# * temp_factor * alt_factor
    
    # === TREE COVERAGE CLASSIFICATION ===
    # Based on effective moisture availability for tree growth
    # Thresholds: <0.2 none, 0.2-0.5 sparse, 0.5-1.2 forest, 1.2-2.5 dense, >2.5 rainforest
    trees_none = eff_tree_moisture < 0.2
    trees_sparse = ~trees_none & (eff_tree_moisture < 0.5)
    trees_forest = ~trees_none & (eff_tree_moisture >= 0.5) & (eff_tree_moisture < 1.2)
    trees_dense = ~trees_none & (eff_tree_moisture >= 1.2) & (eff_tree_moisture < 2.5)
    trees_rainforest = ~trees_none & (eff_tree_moisture >= 2.5)

    # === SNOW COVERAGE CLASSIFICATION ===
    # Snow present based on mean annual temp (not coldest month) + noise
    # Threshold ~3°C: regions with mean temp below this typically have seasonal snow cover
    snow_temp = temp + snow_noise
    has_snow = snow_temp < 0.0

    # === ELEVATION BANDS ===
    is_ocean = alt_m_signed < 0.0
    high_peaks = alt_m > 4500.0
    mountains = (alt_m > 2500.0) & ~high_peaks
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
    # BRANCH 2: HIGH PEAKS (>4500m) - disjoint from lower elevations
    # ============================================================
    peak_mask = high_peaks & unassigned
    if peak_mask.any():
        # Frozen peaks: has snow
        peak_frozen = peak_mask & has_snow
        out[peak_frozen] = _BIOME_ID["frozen_peaks"]
        
        # Stony peaks: no snow (rare at this elevation but possible in tropics)
        peak_stony = peak_mask & ~has_snow
        out[peak_stony] = _BIOME_ID["stony_peaks"]
        
        unassigned[peak_mask] = False

    # ============================================================
    # BRANCH 3: MOUNTAINS (2500-4500m) - disjoint from peaks and lowlands
    # ============================================================
    mtn_mask = mountains & unassigned
    if mtn_mask.any():
        #
        # SNOW
        #
        
        # Snowy barren: frozen and dry
        mtn_snowy_barren = mtn_mask & has_snow & trees_none
        out[mtn_snowy_barren] = _BIOME_ID["snowy_slopes"]
        
        # Snowy sparse: mountains with snow, not barren but sparse
        mtn_snowy = mtn_mask & has_snow & trees_sparse
        out[mtn_snowy] = _BIOME_ID["snowy_plains"]
        
        # Snowy dense: mountains with snow, dense trees
        mtn_snowy = mtn_mask & has_snow & (trees_forest | trees_dense | trees_rainforest)
        out[mtn_snowy] = _BIOME_ID["snowy_taiga"]
        
        #
        # NO SNOW
        #
        
        # Windswept hills: dry mountains
        mtn_windswept = mtn_mask & ~has_snow & trees_none
        out[mtn_windswept] = _BIOME_ID["windswept_hills"]
        
        # Plains: no snow, cool/cold, sparse trees
        mtn_plains = mtn_mask & ~has_snow & trees_sparse
        out[mtn_plains] = _BIOME_ID["plains"]
        
        # Taiga: treeline zone (cold but not frozen, some moisture)
        mtn_taiga = mtn_mask & ~has_snow & (trees_forest | trees_dense | trees_rainforest)
        out[mtn_taiga] = _BIOME_ID["taiga"]
        
        unassigned[mtn_mask] = False

    # ============================================================
    # BRANCH 4: LOWLAND + MIDLAND - primary biome classification
    # Uses tree coverage × snow × temperature
    # ============================================================
    land_mask = unassigned  # everything remaining

    # --- 4A: FROZEN/SNOWY + NO/SPARSE TREES ---
    snowy_barren = land_mask & has_snow & (trees_none | trees_sparse)
    out[snowy_barren] = _BIOME_ID["snowy_plains"]
    land_mask = land_mask & ~snowy_barren

    # --- 4B: SNOWY + FOREST (snowy taiga) ---
    snowy_forest = land_mask & has_snow & (trees_forest | trees_dense | trees_rainforest)
    out[snowy_forest] = _BIOME_ID["snowy_taiga"]
    land_mask = land_mask & ~snowy_forest

    # --- 4D: NO SNOW, NO TREES (desert/badlands/plains) ---
    dry_barren = land_mask & ~has_snow & trees_none
    # Hot + arid = desert
    desert_mask = dry_barren & (warm | hot)
    out[desert_mask] = _BIOME_ID["desert"]
    # Cool/cold + arid = windswept
    windswept_mask = dry_barren & (cold | cool | temperate) & ~lowland
    out[windswept_mask] = _BIOME_ID["windswept_hills"]
    # Remaining dry barren = plains (cold steppe)
    plains_barren = dry_barren & ~desert_mask & ~windswept_mask
    out[plains_barren] = _BIOME_ID["plains"]
    land_mask = land_mask & ~dry_barren

    # --- 4E: NO SNOW, SPARSE TREES (savanna/steppe) ---
    sparse_land = land_mask & ~has_snow & trees_sparse
    # Warm/hot sparse = savanna
    savanna_mask = sparse_land & (warm | hot)
    out[savanna_mask] = _BIOME_ID["savanna"]
    # Cool/temperate sparse = meadow or plains
    meadow_mask = sparse_land & (cool | temperate)
    out[meadow_mask] = _BIOME_ID["plains"]
    # Cold sparse = taiga edge
    cold_sparse = sparse_land & cold
    out[cold_sparse] = _BIOME_ID["taiga"]
    land_mask = land_mask & ~sparse_land

    # --- 4F: NO SNOW, FOREST ---
    forest_land = land_mask & ~has_snow & trees_forest
    # Hot forest = jungle
    jungle_mask = forest_land & hot
    out[jungle_mask] = _BIOME_ID["jungle"]
    # Warm forest = forest (subtropical)
    warm_forest = forest_land & warm
    out[warm_forest] = _BIOME_ID["forest"]
    # Temperate forest
    temp_forest = forest_land & temperate
    out[temp_forest] = _BIOME_ID["forest"]
    # Cool forest = taiga
    cool_forest = forest_land & (cool | cold)
    out[cool_forest] = _BIOME_ID["taiga"]
    land_mask = land_mask & ~forest_land

    # --- 4G: NO SNOW, DENSE FOREST ---
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

    # --- 4H: NO SNOW, RAINFOREST ---
    rain_land = land_mask & ~has_snow & trees_rainforest
    # Tropical rainforest = jungle
    jungle_rain = rain_land & (hot | (warm & tropical))
    out[jungle_rain] = _BIOME_ID["jungle"]
    # Lowland wet = swamp
    swamp_rain = rain_land & ~jungle_rain & lowland
    out[swamp_rain] = _BIOME_ID["swamp"]
    # Temperate rainforest = forest
    forest_rain = rain_land & ~jungle_rain & ~swamp_rain
    out[forest_rain] = _BIOME_ID["forest"]
    land_mask = land_mask & ~rain_land

    # --- 4I: FALLBACK (any remaining unassigned land) ---
    out[land_mask] = _BIOME_ID["plains"]

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
        biome = _classify_biome(elev, climate, i1, j1)
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
        biome = _classify_biome(elev, climate, i1, j1)
        if request.args.get("format") == "json":
            return jsonify(_tensor_to_json(elev))
        return _binary_response(elev, biome=biome)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/22")
def elev_22():
    i1, j1, i2, j2 = _parse_quad()
    world = _get_pipeline()
    out = world.get_22(i1, j1, i2, j2, with_climate=True)
    elev = out["elev"]
    climate = out.get("climate")
    biome = _classify_biome(elev, climate, i1, j1)
    if request.args.get("format") == "json":
        return jsonify(_tensor_to_json(elev))
    return _binary_response(elev, biome=biome)


@app.get("/11")
def elev_11():
    try:
        i1, j1, i2, j2 = _parse_quad()
        world = _get_pipeline()
        out = world.get_11(i1, j1, i2, j2, with_climate=True)
        elev = out["elev"]
        climate = out.get("climate")
        biome = _classify_biome(elev, climate, i1, j1)
        if request.args.get("format") == "json":
            return jsonify(_tensor_to_json(elev))
        return _binary_response(elev, biome=biome)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False, threaded=False)

