# Terrain Diffusion

A practical, learned successor to Perlin noise for infinite, seed consistent, real time terrain generation.

Terrain Diffusion provides:
- InfiniteDiffusion, an algorithm for unbounded diffusion sampling with constant time random access. Utilizes [infinite-tensor](https://github.com/xandergos/infinite-tensor).
- A hierarchical stack of models for generating planetary terrain
- Real time streaming of terrain and climate data
- API for a pretty cool Minecraft mod

## Related Repositories

**Infinite Tensor**  
Python library for managing infinite-dimensional tensors

https://github.com/xandergos/infinite-tensor

**Minecraft Mod (For minecraft integration)**  
Fabric mod that replaces Minecraft's world generator.

https://github.com/xandergos/terrain-diffusion-mc

## Installation

```bash
git clone https://github.com/xandergos/terrain-diffusion
cd terrain-diffusion
pip install -r requirements.txt
```

### Install with pip
`pip install git+https://github.com/xandergos/terrain-diffusion.git`

## GPU Acceleration (CUDA)

It is strongly recommended to ensure that PyTorch is installed with CUDA support for GPU acceleration.
Terrain Diffusion is quite fast, and can run on a CPU as well, but it will be much slower.

To use GPU acceleration you only need
- An NVIDIA GPU
- Updated NVIDIA drivers
- A PyTorch build with CUDA
- Mac is CPU-only.

### Steps on Windows or Linux

1. Install latest NVIDIA driver

2. Install PyTorch with CUDA:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Verify (Optional):

```
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start

### Explore the World

`python -m terrain_diffusion explore`

### API for Minecraft

`python -m terrain_diffusion mc-api`

## Training from scratch

See [TRAINING.md](TRAINING.md) for a step-by-step guide. This is, of course, pretty lengthy.

## Modifying world generation (Advanced)

There are two ways to modify world generation without training from scratch.

### Modifying the synthetic map

The code for generating the base map used for everything is at `terrain_diffusion\inference\synthetic_map.py`. It is basically just a bunch of perlin noise with some transformations to have the same statistics as real world data, and make sure the climate is at least kind of reasonable. You can modify the file directly to change how the world is generated. While testing, it's recommended to use `--hdf5-file TEMP` so you don't have to delete the HDF5 file cache every time you make a change.

### Retraining the coarse model

The coarse model is tiny, so you can feasibly play around with the model parameters or the dataset to make new kinds of worlds. For example, you may over-sample crops that have harsher gradients. The dataset is in `terrain_diffusion\training\datasets\coarse_dataset.py`. You can also modify the config to create more or less powerful coarse models.

1. Download ETOPO

Download the "30 Arc-Second Resolution GeoTIFF" [here](https://www.ncei.noaa.gov/products/etopo-global-relief-model) and place it in `data/global`.

2. Download WorldClim data

Download `bio 30s` [here](https://www.worldclim.org/data/worldclim21.html). Extract all into `data/global`.

3. Train with:

```
accelerate launch -m terrain_diffusion train --config ./configs/diffusion_coarse/diffusion_coarse.cfg
```

4. Save the model:

```
python -m terrain_diffusion.training.save_model -c checkpoints/diffusion_coarse/latest_checkpoint -s 0.05
```

Move the output folder (Probably `checkpoints/diffusion_coarse/latest_checkpoint/saved_model`) to `checkpoints/models/diffusion_coarse`