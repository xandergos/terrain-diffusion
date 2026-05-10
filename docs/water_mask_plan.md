# Water Mask Implementation Plan

## Pre-trained models available (HuggingFace)

The full 30m pipeline lives at `xandergos/terrain-diffusion-30m` and contains three subfolders.
Individual models are also published separately:

| Model | HF repo | Role in water plan |
|---|---|---|
| Coarse | `xandergos/TerrainDiffusion-Diffusion-Coarse-128A` | **Reuse unchanged (MVP + Phase 2)** |
| Base (consistency) | `xandergos/TerrainDiffusion-Consistency-Base-192x3` | **Reuse unchanged (all phases)** |
| Base (diffusion) | `xandergos/TerrainDiffusion-Diffusion-Base-192x3` | Reuse unchanged |
| Decoder (consistency) | `xandergos/TerrainDiffusion-Consistency-Decoder-64x3` | **Warm-start body weights only** — first/last conv change shape |
| Autoencoder | (embedded in the pipeline, not published standalone) | **Reuse unchanged** |

The **encoded dataset** (`dataset.h5` with `latent`, `residual`, `lowfreq`, `climate`) is fully reused.
Only the `water` channel is appended to it.

---

## Phase 1 — MVP: Decoder-only water mask

**Goal:** the decoder generates elevation + water mask simultaneously.
Coarse and base models are untouched.
The decoder sees: `[noisy_2ch, 4_upsampled_latents]` → 6 input channels, 2 output channels.

### 1.1 Files to change

| File | Change |
|---|---|
| `terrain_diffusion/data/preprocessing/elevation_dataset.py` | `process_single_file_base`: add `water_folder` param, load JRC, 255 mask, threshold, Gaussian blur, return water per chunk |
| `terrain_diffusion/data/preprocessing/build_base_dataset.py` | Add `--water-folder` CLI option; add `water` to stored datasets and to stats loop; add `--append-water` flag to skip re-processing elevation |
| `terrain_diffusion/training/datasets/h5_decoder_terrain_dataset.py` | Load `water` alongside `residual`; stack as 2-channel `image`; fix `calculate_stats` to track per-channel; add `water_mean`/`water_std` constructor params |
| `configs/diffusion_decoder/diffusion_decoder_64-3_30m.cfg` | `in_channels=6`, `out_channels=2`; add `water_mean`, `water_std`, `water_sigma_data` |
| `terrain_diffusion/inference/world_pipeline.py` | `_decoder_inference`: 2-ch noise, split output; `_build_decoder_stage`: output window `(3, ...)` (2 signal + 1 weight); `_compute_elev`: use ch 0 only; add `_compute_water`; `get()`: include `water` in return dict |

### 1.2 Data pipeline

#### Step 1 — Download JRC tiles
Use the same tile grid as your existing DEM data (same `--output_size` and `--output_resolution`)
so geographic bounds align:

```bash
python -m terrain_diffusion data download \
    --image jrc \
    --output_dir data/jrc \
    --output_size 4096 \
    --output_resolution 92.15 \
    --num_workers 8
```

This exports JRC occurrence at native 30m resolution (hardcoded `native_scale=30`) into tiles
whose bounding boxes match your existing DEM tiles.

#### Step 2 — Append water to existing HDF5
The `--append-water` flag (new) iterates over existing chunk/subchunk groups,
skips everything that already has a `water` dataset,
and adds water using the same bounds recovered from `chunk_id` + `resolution`:

```bash
python -m terrain_diffusion.data.preprocessing.build_base_dataset \
    --highres-elevation-folder data/dem \
    --lowres-elevation-file data/global/ETOPO_2022_v1_30s_N90W180_bed.tif \
    --water-folder data/jrc \
    --output-file data/dataset.h5 \
    --append-water \
    --num-workers 8
```

`--append-water` sets the datasets_to_store list to `['water']` only,
reads the grid cell bounds from `create_equal_area_grid` using `chunk_id`,
and skips the Laplacian encode / climate steps entirely.

### 1.3 Training

**Dataset reuse:**
- `dataset.h5` — reuse as-is, only `water` is appended
- Coarse model — not loaded during decoder training
- Base model — not loaded during decoder training
- Autoencoder — not loaded (latents already encoded in HDF5)

**Warm-start from existing decoder:**
The `DiffusionTrainer.load_model_checkpoint` already skips shape-mismatched params.
Load `xandergos/TerrainDiffusion-Consistency-Decoder-64x3` as starting weights.
Only the first conv (5→6 input channels) and last conv (1→2 output channels) start random.
The entire U-Net body (which is most of the parameters) starts warm.

```bash
# 1. Train diffusion decoder (2-channel)
accelerate launch -m terrain_diffusion train \
    --config ./configs/diffusion_decoder/diffusion_decoder_64-3_30m.cfg

# 2. Save EMA model
python -m terrain_diffusion.training.save_model \
    -c checkpoints/diffusion_decoder_water-64x3/latest_checkpoint \
    -s 0.05

# 3. Distill to consistency model
#    Update consistency_decoder_64-3_30m.cfg with the saved model path, then:
accelerate launch -m terrain_diffusion distill \
    --config ./configs/diffusion_decoder/consistency_decoder_64-3_30m.cfg
```

**Expected training time (rough):** 200-400 epochs to see plausible water structure.
Monitor `val/loss` split by channel (add per-channel logging — see tests section).

### 1.4 Tests

Run all of these before touching a GPU:

```bash
python -m pytest tests/test_water_preprocessing.py  # unit
python -m pytest tests/test_water_dataset.py         # unit
python -m pytest tests/test_water_smoke.py           # model shape
```

#### Test 1 — JRC preprocessing (no GPU, no data)
File: `tests/test_water_preprocessing.py`

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def fuse_water(jrc, copernicus_class80=None, jrc_nodata=255, threshold=50, blur_sigma=2.0):
    """Reference implementation to test against."""
    jrc = jrc.copy().astype(np.float32)
    jrc[jrc == jrc_nodata] = 0          # 255 no-data → no water
    water = (jrc >= threshold).astype(np.float32)
    if copernicus_class80 is not None:
        cop_water = (copernicus_class80 == 80).astype(np.float32)
        water = np.clip(water + cop_water * (1 - water), 0, 1)
    return gaussian_filter(water, sigma=blur_sigma)

def test_nodata_255_not_classified_as_water():
    jrc = np.array([[255, 0, 60], [40, 100, 255]], dtype=np.float32)
    result = fuse_water(jrc)
    # pixels where jrc==255 must not be water
    assert result[0, 0] < 0.1
    assert result[1, 2] < 0.1

def test_threshold_at_50():
    jrc = np.full((10, 10), 49, dtype=np.float32)
    assert fuse_water(jrc).max() < 0.1
    jrc[:] = 50
    assert fuse_water(jrc, blur_sigma=0).min() > 0.9

def test_blur_output_range():
    jrc = np.zeros((32, 32), dtype=np.float32)
    jrc[10:20, 10:20] = 100
    result = fuse_water(jrc, blur_sigma=2)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_all_zero_tile():
    jrc = np.zeros((16, 16), dtype=np.float32)
    assert fuse_water(jrc).sum() == 0.0

def test_all_water_tile():
    jrc = np.full((16, 16), 100, dtype=np.float32)
    # after blur, interior should still be near 1
    result = fuse_water(jrc, blur_sigma=2)
    assert result[8, 8] > 0.9
```

#### Test 2 — Dataset shape (no GPU, needs synthetic HDF5)
File: `tests/test_water_dataset.py`

```python
import h5py, torch, tempfile, numpy as np
from terrain_diffusion.training.datasets.h5_decoder_terrain_dataset import H5DecoderTerrainDataset

def make_synthetic_h5(path, crop=64, latent_c=8):
    with h5py.File(path, 'w') as f:
        res = f.require_group('30')
        for cid in ['0', '1', '2']:
            sg = res.require_group(cid).require_group('chunk_0_0')
            H = crop * 8
            res_data = np.random.randn(H, H).astype(np.float32)
            dset = sg.create_dataset('residual', data=res_data)
            dset.attrs.update({'pct_land': 0.8, 'resolution': 30,
                               'data_type': 'residual', 'chunk_id': cid,
                               'subchunk_id': 'chunk_0_0', 'split': 'train'})
            water_data = np.random.uniform(0, 1, (H, H)).astype(np.float32)
            wdset = sg.create_dataset('water', data=water_data)
            wdset.attrs.update({'pct_land': 0.8, 'resolution': 30,
                                'data_type': 'water', 'chunk_id': cid,
                                'subchunk_id': 'chunk_0_0', 'split': 'train'})
            lat = np.random.randn(8, latent_c, crop // 8, crop // 8).astype(np.float32)
            sg.create_dataset('latent', data=lat)
            lf = np.random.randn(crop // 8, crop // 8).astype(np.float32)
            sg.create_dataset('lowfreq', data=lf)

def test_image_shape_is_2_channel():
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        make_synthetic_h5(tmp.name)
        ds = H5DecoderTerrainDataset(
            tmp.name, crop_size=64,
            pct_land_ranges=[[0, 1]], subset_resolutions=[30],
            residual_mean=0.0, residual_std=1.0,
            water_mean=0.1, water_std=0.2, split='train'
        )
        item = ds[0]
        assert item['image'].shape == (2, 64, 64), item['image'].shape

def test_augmentation_applies_consistently():
    """Both channels must receive identical spatial transforms."""
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        make_synthetic_h5(tmp.name)
        ds = H5DecoderTerrainDataset(
            tmp.name, crop_size=64,
            pct_land_ranges=[[0, 1]], subset_resolutions=[30],
            residual_mean=0.0, residual_std=1.0,
            water_mean=0.1, water_std=0.2, split='train'
        )
        # draw 20 samples and check both channels share spatial dims
        for i in range(20):
            item = ds[i]
            assert item['image'].shape[0] == 2

def test_cond_img_shape():
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        make_synthetic_h5(tmp.name)
        ds = H5DecoderTerrainDataset(
            tmp.name, crop_size=64,
            pct_land_ranges=[[0, 1]], subset_resolutions=[30],
            residual_mean=0.0, residual_std=1.0,
            water_mean=0.1, water_std=0.2, split='train'
        )
        item = ds[0]
        assert item['cond_img'].shape == (4, 64, 64)  # 4 latent channels
```

#### Test 3 — Model shape smoke test (no GPU)
File: `tests/test_water_smoke.py`

```python
import torch
from terrain_diffusion.models.edm_unet import EDMUnet2D

def test_decoder_forward_shape():
    model = EDMUnet2D(
        image_size=512, in_channels=6, out_channels=2,
        model_channels=64, model_channel_mults=[1, 2, 3, 4],
        layers_per_block=3, attn_resolutions=[], midblock_attention=False,
        concat_balance=0.5, conditional_inputs=[], fourier_scale='pos'
    )
    x = torch.randn(2, 6, 64, 64)
    noise = torch.randn(2)
    out = model(x, noise_labels=noise, conditional_inputs=[])
    assert out.shape == (2, 2, 64, 64)

def test_decoder_gradient_step():
    """Verify end-to-end: forward + loss + backward without error."""
    model = EDMUnet2D(
        image_size=512, in_channels=6, out_channels=2,
        model_channels=32, model_channel_mults=[1, 2, 3, 4],
        layers_per_block=2, attn_resolutions=[], midblock_attention=False,
        concat_balance=0.5, conditional_inputs=[], fourier_scale='pos'
    )
    sigma_data = 0.5
    images = torch.randn(2, 2, 64, 64)
    cond = torch.randn(2, 4, 64, 64)
    sigma = torch.tensor([0.5, 1.2]).reshape(-1, 1, 1, 1)
    t = torch.atan(sigma / sigma_data)
    noise = torch.randn_like(images) * sigma_data
    x_t = torch.cos(t) * images + torch.sin(t) * noise
    x = torch.cat([x_t / sigma_data, cond], dim=1)

    out, logvar = model(x, noise_labels=t.flatten(), conditional_inputs=[], return_logvar=True)
    pred_v_t = -sigma_data * out
    v_t = torch.cos(t) * noise - torch.sin(t) * images
    loss = (1 / (logvar.exp() * sigma_data**2) * (pred_v_t - v_t)**2 + logvar).mean()
    loss.backward()
    assert not torch.isnan(loss)
```

### 1.5 Visualization

#### A — During training: per-channel loss logging
Add to `DiffusionTrainer.train_step` (after computing loss):
```python
with torch.no_grad():
    per_ch = (1 / (logvar.exp() * sigma_data**2) * (pred_v_t - v_t)**2 + logvar)
    logs['loss_elev'] = per_ch[:, 0].mean().item()
    logs['loss_water'] = per_ch[:, 1].mean().item()
```
Watch `loss_water` decrease over training. If it plateaus while `loss_elev` improves,
increase water loss weight (see `logvar_linear` initialization or add explicit channel weight).

#### B — Dataset inspection (before training)
Run this one-off script to verify JRC data was fused correctly:

```python
import h5py, matplotlib.pyplot as plt
import numpy as np

with h5py.File('data/dataset.h5', 'r') as f:
    # grab a tile that has land
    sg = f['30']['42']['chunk_0_0']  # adjust chunk_id as needed
    residual = sg['residual'][:]
    water = sg['water'][:]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(residual, cmap='terrain')
axes[0].set_title('Residual (elevation)')
axes[1].imshow(water, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Water mask (0–1)')
plt.tight_layout()
plt.savefig('debug/water_check.png', dpi=150)
```

Look for: rivers in valleys, lakes in depressions. Red flag: water everywhere (255 nodata bug)
or water nowhere (threshold or path issue).

#### C — After training: inference visualization

```python
from terrain_diffusion.inference.world_pipeline import WorldPipeline
import matplotlib.pyplot as plt
import numpy as np

pipeline = WorldPipeline.from_pretrained('path/to/new/decoder/pipeline')
pipeline.to('cuda').bind()

result = pipeline.get(0, 0, 512, 512, with_climate=False)
elev = result['elev'].cpu().numpy()
water = result['water'].cpu().numpy()  # 0-1 probability

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(elev, cmap='terrain')
axes[0].set_title('Elevation (m)')
axes[1].imshow(water, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Water probability')
# overlay: water > 0.5 on shaded relief
from terrain_diffusion.inference.relief_map import make_relief
relief = make_relief(elev)
axes[2].imshow(relief)
axes[2].imshow(water > 0.5, alpha=0.4, cmap='Blues')
axes[2].set_title('Relief + water overlay')
plt.tight_layout()
plt.savefig('debug/water_inference.png', dpi=150)
```

**What to look for:**
- Water follows valleys (not random patches) → model learned terrain→water
- No water on mountain peaks or open ocean (ocean handled by negative elevation)
- Continuity at tile boundaries (blending handles this automatically)

**Red flags:**
- All-zero water output → check normalization or model initialization
- Water ignoring terrain structure → train longer or add coarse water (Phase 2)
- Artifacts at tile seams → `decoder_tile_stride` too aggressive

---

## Phase 2 — Coarse water channel

**Goal:** the coarse model generates a water probability channel alongside terrain.
This gives the decoder coarse spatial guidance for where rivers and lakes should appear,
improving long-range coherence (rivers continuing across tiles).

**Prerequisite:** Phase 1 complete and producing plausible but inconsistent water.

### 2.1 Files to change

| File | Change |
|---|---|
| `terrain_diffusion/training/datasets/coarse_dataset.py` | Add water channel 6 (coarse-pooled from HDF5 `water` data) to `CoarseDataset.__getitem__` → `image` shape `(7, H, W)` |
| `configs/diffusion_coarse/diffusion_coarse_water.cfg` | `out_channels=7`; `in_channels=11` (5 Perlin + 6 cond channels) |
| `terrain_diffusion/inference/world_pipeline.py` | `_coarse_inference`: change noise from 6 → 7 channels; output `(8, ...)` (7 signal + 1 weight); `_build_coarse_stage`: `shape=(8, ...)`, `output_window=(8, ...)`; `_latent_inference`/`_build_latent_stage`: coarse_window now reads 8 channels; `_decoder_inference`: concatenate coarse water (upsampled, ch 6) to decoder input → 7 input channels |
| `configs/diffusion_decoder/diffusion_decoder_64-3_30m_v2.cfg` | `in_channels=7` (noisy_2ch + 4_latents + 1_coarse_water) |

### 2.2 Datasets

- **Coarse model**: Retrain from scratch on `CoarseDataset` (uses ETOPO + WorldClim, not the encoded HDF5).
  ETOPO + WorldClim are already downloaded if you followed TRAINING.md.
  Add water by average-pooling the `water` channel from `dataset.h5` at coarse resolution.
- **Decoder model**: Retrain from Phase 1 weights (warm-start), adding coarse water as conditioning.

### 2.3 Training

```bash
# 1. Retrain coarse model with 7-channel output
accelerate launch -m terrain_diffusion train \
    --config ./configs/diffusion_coarse/diffusion_coarse_water.cfg

# 2. Save coarse model
python -m terrain_diffusion.training.save_model \
    -c checkpoints/diffusion_coarse_water/latest_checkpoint -s 0.05

# 3. Retrain diffusion decoder v2 (7-channel input, 2-channel output)
#    Initialize from Phase 1 decoder checkpoint
accelerate launch -m terrain_diffusion train \
    --config ./configs/diffusion_decoder/diffusion_decoder_64-3_30m_v2.cfg

# 4. Distill decoder v2 → consistency model
accelerate launch -m terrain_diffusion distill \
    --config ./configs/diffusion_decoder/consistency_decoder_64-3_30m_v2.cfg
```

### 2.4 Tests

```python
def test_coarse_model_7ch_output():
    model = EDMUnet2D(
        image_size=64, in_channels=11, out_channels=7, ...
    )
    x = torch.randn(1, 11, 64, 64)
    out = model(x, noise_labels=torch.randn(1), conditional_inputs=[...])
    assert out.shape == (1, 7, 64, 64)

def test_coarse_tile_store_channel_count():
    """After running _build_coarse_stage, coarse InfiniteTensor has shape (8, ...)."""
    pipeline = WorldPipeline(...)
    pipeline.bind()
    tile = pipeline.coarse[:, 0:4, 0:4]
    assert tile.shape[0] == 8  # 7 signal + 1 weight
```

### 2.5 Visualization

Compare Phase 1 and Phase 2 water masks side-by-side:

```python
result_v1 = pipeline_v1.get(0, 0, 1024, 1024)
result_v2 = pipeline_v2.get(0, 0, 1024, 1024)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes[0, 0].imshow(result_v1['elev'].numpy(), cmap='terrain')
axes[0, 0].set_title('Phase 1 — Elevation')
axes[0, 1].imshow(result_v1['water'].numpy(), cmap='Blues', vmin=0, vmax=1)
axes[0, 1].set_title('Phase 1 — Water (no coarse guidance)')
axes[1, 0].imshow(result_v2['elev'].numpy(), cmap='terrain')
axes[1, 0].set_title('Phase 2 — Elevation')
axes[1, 1].imshow(result_v2['water'].numpy(), cmap='Blues', vmin=0, vmax=1)
axes[1, 1].set_title('Phase 2 — Water (with coarse guidance)')
plt.savefig('debug/phase1_vs_phase2_water.png', dpi=150)
```

**What to look for:** rivers should align better across the ~7km coarse grid.
Phase 1 rivers can stop/start at tile boundaries; Phase 2 should be more continuous.

---

## Phase 3 — Base model conditioning (deferred)

Extend the base model's 58-dim conditioning vector with coarse water statistics.
This is the highest-complexity, lowest-marginal-gain change for the prototype.
Defer until Phase 2 output shows water coherence is limited by the latent representation.

**Prerequisite:** Phase 2 running and water masks are spatially coherent but
elevation in wet regions still looks wrong (e.g., valleys without rivers are flat).
If elevation quality is fine, skip entirely.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| JRC 255 no-data classified as water | High | Mask before threshold (already implemented in code) |
| Water channel drives loss to ignore elevation | Medium | Log per-channel loss; add explicit channel weight if needed |
| Tile seam artifacts in water mask | Low | Already handled by weight-window blending in `_build_decoder_stage` |
| Water never learned (model outputs zeros) | Medium | Check normalization; try raw 0–1 instead of mean/std normalize |
| Training significantly slower (2× output channels) | Low | Crop size 128 already fits in 24GB |
