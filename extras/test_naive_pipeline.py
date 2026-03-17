"""Visual test for the naive pipeline's combined latent+decoder stage."""
import random
import time
import torch
import matplotlib.pyplot as plt
from naive_pipeline import WorldPipeline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'fp16' if DEVICE == 'cuda' else None

print(f"Loading pipeline (device={DEVICE}, dtype={DTYPE})...")
t0 = time.time()
world = WorldPipeline.from_local_models(
    seed=42,
    caching_strategy='direct',
    cache_limit=None,
    dtype=DTYPE,
    decoder_tile_stride=512,
)
world.to(DEVICE)
world.bind("TEMP")
print(f"Loaded in {time.time() - t0:.1f}s")

i0 = random.randint(-10000, 10000)
j0 = random.randint(-10000, 10000)
size = 1024
print(f"Generating 1024x1024 at ({i0}, {j0})...")
t0 = time.time()
result = world.get(i0, j0, i0 + size, j0 + size, with_climate=False)
print(f"Done in {time.time() - t0:.1f}s")

elev = result['elev'].numpy()
print(f"Elev range: [{elev.min():.1f}, {elev.max():.1f}] m")

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(elev, cmap='terrain')
ax.set_title(f"1024x1024 at ({i0}, {j0})")
plt.colorbar(im, ax=ax, label='Elevation (m)')
fig.tight_layout()
plt.show()
