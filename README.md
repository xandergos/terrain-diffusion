# Terrain Diffusion

A practical, learned successor to Perlin noise for infinite, seed consistent, real time terrain generation.

Terrain Diffusion provides:
- InfiniteDiffusion, an algorithm for unbounded diffusion sampling with constant time random access. Utilizes [infinite-tensor](https://github.com/xandergos/infinite-tensor).
- A hierarchical stack of models for generating planetary terrain
- Real time streaming of terrain and climate data
- API for a pretty cool Minecraft mod

---

## Related Repositories

**Infinite Tensor**  
Python library for managing infinite-dimensional tensors

https://github.com/xandergos/infinite-tensor

**Minecraft Mod (For minecraft integration)**  
Fabric mod that replaces Minecraft's world generator.

https://github.com/xandergos/terrain-diffusion-mc

---

## Installation

```bash
git clone https://github.com/xandergos/terrain-diffusion
cd terrain-diffusion
pip install -r requirements.txt
````

### Install with pip
`pip install git+https://github.com/xandergos/terrain-diffusion.git`

---

## Quick Start

### Explore the World

`python -m terrain_diffusion explore`

### API for Minecraft

`python -m terrain_diffusion mc-api`

---

## Training from scratch

See [TRAINING.md](TRAINING.md) for a step-by-step guide. This is, of course, pretty lengthy.