# River / Water Mask Support Design

**Date:** 2026-05-10
**Target:** `xandergos/terrain-diffusion-30m` (30m resolution model)

## Goal

Add a water mask (rivers, lakes, coastline) as an additional output channel alongside elevation. At inference, the model generates both elevation and a 0-1 water probability map from the same seed-based conditioning — no new user inputs required.

## Non-Goals

- Explicit land cover generation or conditioning (future enhancement)
- Seasonal/transient water detection (permanent water only)
- Physics-based hydrology (flow accumulation, erosion models)
- Real-time conditioning on user-provided land cover at inference

## Design Overview

```
Seed → Perlin (elevation + climate, 5 channels, unchanged)
    → Coarse model:  6 → 7 channels (+ water probability, low-res)
    → Base/Latent model: 5 → 6 channels (+ water logits in latent space)
    → Decoder model:  1 → 2 channels (elevation residual + water mask, full-res)
    → Final output: elevation (m) + water mask (0-1)
```

## Architecture Changes

### 1. Training Data (HDF5)

Add two new datasets to the HDF5 file per subchunk:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `water` | `(H, W)` float32 | Binary water mask (1 = permanent water, 0 = land), followed by Gaussian blur to soften boundaries. Derived from Copernicus Land Cover "permanent water bodies" class at 100m, reprojected to model resolution (30m). |
| `landcover` | `(H, W, N)` float32, optional | Soft class probabilities per pixel (Gaussian-blurred one-hot). Reserved for future use; not wired into training initially. Derived from Copernicus Land Cover 100m classification. |

Water and landcover processing is done during `build_base_dataset.py`, stored alongside existing `residual`, `lowfreq`, `lowres_exact`, `climate`.

Per-channel normalization stats are computed for water (mean, std) during the stats calculation phase.

### 2. Autoencoder (`edm_autoencoder.py`)

Minimal config change — architecture already supports multi-channel:

- `in_channels`: 1 → 2
- `out_channels`: 1 → 2  
- `latent_channels`: 4 → 5 (one additional latent channel to encode water structure)
- `direct_skips`: unchanged (empty)

Water is encoded into the VAE latent space alongside elevation. The extra latent channel gives the VAE enough capacity to represent both elevation and water structure. The Gaussian blur applied to the water mask during preprocessing (see data pipeline section) makes it smooth enough for VAE compression.

The 5 VAE latent channels jointly encode elevation + water. The base model generates all 5.

### 3. Laplacian Pyramid (`laplacian_encoder.py`)

Water is NOT processed through the Laplacian pyramid. The pyramid is only meaningful for elevation (low-freq = regional shape, high-freq = detail). Applying Laplacian decomposition to a water mask would produce nonsensical results.

The water channel is kept as-is alongside the elevation residual throughout the Laplacian encode/decode pipeline as a pass-through — it is loaded/stored in HDF5 but not transformed.

### 4. Coarse Model (`diffusion_coarse`)

- **Output channels:** 6 → 7 (adds water probability channel alongside existing channels)
- **Input/conditioning:** Unchanged. The model learns to predict water from climate+elevation conditioning alone.
- **Training:** The coarse water probability target is derived from the downsampled (e.g., 1/64) water mask from the training data.

### 5. Base/Latent Model (`diffusion_base`)

- **Output channels:** 5 → 6 (5 VAE latents + 1 lowfreq). The extra VAE latent channel encodes water structure jointly with elevation.
- **Conditioning:** The 58-dim conditioning vector expands to include coarse water statistics (mean, presence) from the coarse model output, computed in the same spatial windows as existing conditioning stats.

### 6. Decoder Model (`diffusion_decoder`)

- **Output channels:** 1 → 2 (residual + water mask)
- **Conditioning:** Input channels expand from 5 to 6 (5 upsampled VAE latents). The 5 VAE latents are upsampled via nearest-neighbor from 64×64 to 512×512 and concatenated alongside the noisy residual input.
- **Loss:** Two-channel loss with per-channel logvar. Water and elevation have independent learned variances.

### 7. Inference Pipeline (`world_pipeline.py`)

- `_compute_elev()` returns both elevation and water mask
- `get()` returns `{'elev': elev, 'climate': climate, 'water': water}`
- `_decoder_inference()` generates 2 channels instead of 1
- Weight blending in the 2-channel tensor handles overlap regions
- `_compute_water()` reconstructs full-res water mask from decoder output (same structure as `_compute_elev()` or integrated into it)
- No new conditioning inputs required — same seed, same Perlin maps

### 8. CLI and API

- `world_generator.py`: add `--output-water` flag to write water mask GeoTIFF alongside elevation
- `api.py`: include `water` in query results when requested
- Visualization: `relief_map.py` already supports river overlay via flow accumulation; replace with direct water mask rendering (blue overlay where water > threshold)

## Data Pipeline Changes

### New Download Data Sources

| Source | Product | Resolution | Purpose |
|--------|---------|------------|---------|
| Copernicus Land Cover | Proba-V-C3 Global 2019 | 100m | Water mask labels (permanent water bodies) + future land cover |
| JRC Global Surface Water | Occurrence / Seasonality | 30m | Alternative/larger-area water labels (rivers, lakes, reservoirs) |

### Preprocessing (`build_base_dataset.py`)

For each elevation/climate tile processed:

1. Load Copernicus land cover TIFF for the same spatial extent
2. Extract "permanent water bodies" class (class 80) → binary mask
3. Reproject to match elevation grid resolution (30m) and CRS
4. Apply Gaussian blur (sigma ~1-2 pixels) to soften hard edges — diffusion models perform better with continuous targets
5. Store as `water` dataset in HDF5 subchunk

Optionally (for future):
1. Load Copernicus land cover classification
2. Convert to soft class probabilities (Gaussian-blurred one-hot)
3. Store as `landcover_probs` dataset in HDF5

### Stats Calculation

During the stats phase, compute mean and standard deviation for the water channel (in addition to existing residual and climate stats). These feed into per-channel normalization in all datasets.

## Training Changes

### Autoencoder Training

- Dataset (`H5AutoencoderDataset`): returns `(2, H, W)` tensor — [residual, water]
- Loss: reconstruction loss on both channels. Equal weight initially.
- The VAE KL-divergence loss applies to all 5 latent channels (no direct_skips — water is encoded jointly with elevation).

### Diffusion Training (Decoder)

- Trainer: generates 2-channel output instead of 1
- Loss: `loss = loss_elev + loss_water`, each with its own per-channel logvar weighting
- Multi-channel sampling: Karras/DPMSolver scheduler operates on both channels jointly

### Diffusion Training (Base)

- Latent target: 6 channels (5 VAE latents + 1 lowfreq). The 5 VAE latents jointly encode elevation and water.
- Conditioning: extended with water stats from coarse output

### Diffusion Training (Coarse)

- Output target: 7 channels (6 terrain/climate + 1 water probability)
- Water target: average-pooled water mask from training data at coarse resolution
- No changes to conditioning inputs

## Edge Cases and Considerations

- **Ocean vs inland water:** The water mask should only cover inland water (rivers, lakes) plus permanent coastline water. Open ocean should be handled by the existing `pct_land` filter and elevation sign (negative = ocean).
- **Water at tile boundaries:** Weighted blending in the decoder stage naturally handles edge continuity for the water channel.
- **Consistency distillation:** The consistency model training (single-step generation) must be updated to handle 2-channel output. AutoGuidance scales may differ per channel.
- **Water channel scaling:** Water is 0-1 probability. No signed-square-root transform applied (unlike elevation). Direct prediction in probability space.
- **Copernicus mismatch:** 100m resolution water labels → 30m model. Nearest-neighbor upsampling or bilinear resampling. The Gaussian blur step mitigates aliasing.
- **Model variant compatibility:** Initially target 30m model only. 90m requires separate training but same architecture changes.

## Success Criteria

1. Model generates plausible water masks that follow terrain (rivers in valleys, lakes in depressions) when sampling with realistic climate conditioning.
2. Water mask does not degrade elevation quality (elevation-only metrics remain within 5% of baseline).
3. Perlin-generated conditioning produces water masks on arbitrary seeds without external data.
4. CLI can export water mask as a separate GeoTIFF.
