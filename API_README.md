# Terrain API

Flask-based REST API for serving terrain and climate data from the terrain diffusion pipeline.

## Starting the Server

```bash
python -m terrain_diffusion.inference.api --hdf5-file world.h5 --port 8000
```

### Configuration Options

- `--hdf5-file`: Path to HDF5 file (default: `world.h5`, use `TEMP` for temporary file)
- `--port`: Server port (default: 8000, or `PORT` env var)
- `--host`: Server host (default: `0.0.0.0`)
- `--seed`: Random seed (default: from file or random)
- `--device`: Device (`cuda`/`cpu`, default: auto-detect, or `TERRAIN_DEVICE` env var)
- `--drop-water-pct`: Drop water percentage (default: 0.5)
- `--frequency-mult`: Frequency multipliers as JSON array (default: `[1.0, 1.0, 1.0, 1.0, 1.0]`)
- `--cond-snr`: Conditioning SNR as JSON array (default: `[0.5, 0.5, 0.5, 0.5, 0.5]`)
- `--histogram-raw`: Pre-softmax beauty histogram values as JSON array (default: `[0.0, 0.0, 0.0, 1.0, 1.5]`)
- `--latents-batch-size`: Batch size for latent generation (default: 4)
- `--log-mode`: Logging mode (`info` or `verbose`, default: `verbose`)

## `GET /terrain`

Get terrain elevation and climate data for a bounding box.

**Query Parameters:**
- `i1`, `j1`, `i2`, `j2`: Bounding box coordinates in target resolution (required)
- `scale`: Integer scale factor relative to 90m resolution (default: 1)
  - `1` = 90m per pixel
  - `2` = 45m per pixel
  - `4` = 22.5m per pixel
  - `8` = 11.25m per pixel

**Example:**
```
GET /terrain?i1=0&j1=0&i2=256&j2=256&scale=1
```

**Response:**

Binary data with:
- **Elevation**: `int16` little-endian (H×W×2 bytes)
  - Values are in meters (floored) and clamped to [-32768, 32767]
- **Climate**: `float32` little-endian interleaved (H×W×4×4 bytes)
  - Channels: `temp`, `t_season`, `precip`, `p_cv`
  - `temp`: Annual Mean Temperature (C)
  - `t_season`: Temperature Seasonality (standard deviation ×100)
  - `precip`: Annual Precipitation (mm/year)
  - `p_cv`: Precipitation Seasonality (Coefficient of Variation, Percentage)
  - Equivalent to BIO1, BIO4, BIO12, and BIO15 from [WorldClim](https://www.worldclim.org/data/bioclim.html).
  - Layout: (H, W, 4) interleaved

**Response Headers:**
- `X-Height`: Output height in pixels
- `X-Width`: Output width in pixels

**Error Response:**
```json
{"error": "error message"}
```

## Usage Example

```python
import requests
import numpy as np

# Request terrain data
response = requests.get(
    "http://localhost:8000/terrain",
    params={"i1": 0, "j1": 0, "i2": 256, "j2": 256, "scale": 1}
)

# Parse headers
h = int(response.headers["X-Height"])
w = int(response.headers["X-Width"])

# Parse elevation (int16)
elev_bytes = response.content[:h * w * 2]
elevation = np.frombuffer(elev_bytes, dtype="<i2").reshape(h, w)

# Parse climate (float32, 4 channels)
climate_bytes = response.content[h * w * 2:]
climate = np.frombuffer(climate_bytes, dtype="<f4").reshape(h, w, 4)
```
