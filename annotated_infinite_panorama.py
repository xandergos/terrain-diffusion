"""Self-contained demo: generate an infinite-width panorama with Stable
Diffusion v1.5 and the infinite-tensor framework.

Install:
    pip install torch diffusers transformers accelerate infinite-tensor pillow numpy

Run:
    python infinite_panorama.py

Method:
    1. Build a deterministic 1D-tiled Gaussian noise field so any sub-window
       of the (infinite) panorama sees the same initial noise.
    2. Denoise the latents through several *phases*, each covering a range
       of diffusion timesteps. Between phases, overlapping tiles are blended
       with a linear weight kernel (handled automatically by
       infinite-tensor summing overlapping window outputs).
    3. VAE-decode the final latents to pixels, again with overlap blending.
    4. Slice the resulting infinite pixel tensor to the desired width.

The panorama is unbounded. We crop `CROP_PIXEL_WIDTH` pixels
and save as ``output.png`` next to this file.
"""

import os

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline
from infinite_tensor import MemoryTileStore, TensorWindow

# -----------------------------------------------------------------------------
# Tweakables
# -----------------------------------------------------------------------------
PROMPT = "a photo of a mountain range at sunset"
# Diffusion-timestep boundaries separating the phases. More thresholds =>
# more blending rounds => stronger global consistency (and more compute).
INTERMEDIATE_TIMESTEPS = (400, 600, 750, 900)
CROP_PIXEL_WIDTH = 2048        # Width of the final cropped region (pixels).
LATENT_STRIDE = 32             # Overlap stride in latent space (tile = 64).
PIXEL_STRIDE = 384             # Overlap stride in pixel space (tile = 512).
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
SEED = 0

# SDv1.5 constants.
LATENT_TILE = 64
PIXEL_TILE = 512
LATENT_CHANNELS = 4

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.png")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def tiled_gaussian_noise(seed, x0, width, channels=LATENT_CHANNELS, height=LATENT_TILE, tile=256):
    """Sample a ``(channels, height, width)`` patch from a deterministic 1D-tiled
    Gaussian noise field. The value at column ``x`` depends only on
    ``(seed, x // tile)``, so overlapping tile requests agree, giving a
    self-consistent infinite noise field."""
    out = np.empty((channels, height, width), dtype=np.float32)
    first_tx = x0 // tile
    last_tx = (x0 + width - 1) // tile
    for tx in range(first_tx, last_tx + 1):
        tile_x0 = tx * tile
        ox0 = max(x0, tile_x0)
        ox1 = min(x0 + width, tile_x0 + tile)
        ss = np.random.SeedSequence(np.array([seed, tx], dtype=np.uint32))
        rng = np.random.Generator(np.random.PCG64DXSM(ss))
        tile_noise = rng.standard_normal((channels, height, tile), dtype=np.float32)
        out[:, :, ox0 - x0:ox1 - x0] = tile_noise[:, :, ox0 - tile_x0:ox1 - tile_x0]
    return out


def linear_kernel(height, width):
    """Separable linear blending weight, peak 1 at center, ~0 at edges."""
    x = torch.arange(width, dtype=torch.float32)
    mid = (width - 1) / 2
    w = 1 - 0.999 * torch.abs(x - mid) / mid
    return w[None, :].expand(height, -1).contiguous()


def build_timestep_ranges(all_timesteps, thresholds):
    """Partition descending ``all_timesteps`` into phases using ``thresholds``.
    Phase 0 gets ``t >= thresholds[0]``, the last phase gets
    ``t < thresholds[-1]``, intermediate phases fill the gaps."""
    thresholds = sorted(thresholds, reverse=True)
    if not thresholds:
        return [all_timesteps]
    ranges = []
    prev = None
    for t in thresholds:
        r = all_timesteps[all_timesteps >= t] if prev is None \
            else all_timesteps[(all_timesteps >= t) & (all_timesteps < prev)]
        if len(r) > 0:
            ranges.append(r)
        prev = t
    tail = all_timesteps[all_timesteps < thresholds[-1]]
    if len(tail) > 0:
        ranges.append(tail)
    return ranges


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    def encode(text):
        """Encode a string into CLIP text embeddings."""
        toks = pipe.tokenizer(
            text, padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(pipe.device)
        return pipe.text_encoder(toks)[0]

    text_emb = torch.cat([encode(""), encode(PROMPT)])  # [uncond, cond]

    def denoise(latent, timesteps):
        """Run classifier-free-guided DDIM steps on a ``(1, C, H, W)`` latent."""
        for t in timesteps:
            inp = pipe.scheduler.scale_model_input(torch.cat([latent] * 2), t)
            with torch.no_grad():
                pred = pipe.unet(inp, t, encoder_hidden_states=text_emb).sample
            uncond, cond = pred.chunk(2)
            pred = uncond + GUIDANCE_SCALE * (cond - uncond)
            latent = pipe.scheduler.step(pred, t, latent).prev_sample
        return latent

    latent_weight = linear_kernel(LATENT_TILE, LATENT_TILE)
    pixel_weight = linear_kernel(PIXEL_TILE, PIXEL_TILE)
    phase_timesteps = build_timestep_ranges(pipe.scheduler.timesteps, INTERMEDIATE_TIMESTEPS)
    init_sigma = pipe.scheduler.init_noise_sigma

    # Each latent/pixel tensor carries C+1 channels: C weighted values plus a
    # weight channel. The infinite-tensor framework *sums* overlapping window
    # outputs; dividing the first C channels by the last recovers the weighted
    # average across overlapping tiles. That division is `normalize` below.
    def normalize(weighted):
        return weighted[:-1] / weighted[-1:].clamp(min=1e-6)

    def pack(values_chw, weight_hw):
        """``(C, H, W) + (H, W) -> (C+1, H, W)`` weighted output for infinite-tensor."""
        return torch.cat([values_chw * weight_hw[None], weight_hw[None]], dim=0)

    # Window moves `LATENT_STRIDE` columns per step; tiles overlap heavily.
    latent_window = TensorWindow(
        size=(LATENT_CHANNELS + 1, LATENT_TILE, LATENT_TILE),
        stride=(LATENT_CHANNELS + 1, LATENT_TILE, LATENT_STRIDE),
    )
    # For VAE decode we re-sample the latent tensor at the pixel-output cadence
    # (one latent tile per pixel tile; `PIXEL_STRIDE // 8` in latent units).
    latent_decode_window = TensorWindow(
        size=(LATENT_CHANNELS + 1, LATENT_TILE, LATENT_TILE),
        stride=(LATENT_CHANNELS + 1, LATENT_TILE, PIXEL_STRIDE // 8),
    )
    pixel_window = TensorWindow(
        size=(3 + 1, PIXEL_TILE, PIXEL_TILE),
        stride=(3 + 1, PIXEL_TILE, PIXEL_STRIDE),
    )

    store = MemoryTileStore()

    # Phase numbering tracks the noise level of the *latent*. With
    # T = len(INTERMEDIATE_TIMESTEPS) + 1, the (virtual) pure-noise latent
    # sits at level T; each denoising range drops the level by one; phase 0
    # is the fully denoised latent. The VAE decode is a separate stage.
    T = len(phase_timesteps)

    def initial_phase(ctx):
        """Phase T-1: sample pure noise at this window column, denoise the highest-t range."""
        x = ctx[2] * LATENT_STRIDE
        noise = torch.as_tensor(tiled_gaussian_noise(SEED, x, LATENT_TILE)) * init_sigma
        noise = noise.to(pipe.device, dtype=torch.float16).unsqueeze(0)
        latent = denoise(noise, phase_timesteps[0])[0].cpu().float()
        return pack(latent, latent_weight)

    def make_continuation_phase(timesteps):
        """Phases T-2..0: read blended tile from previous phase, denoise further."""
        def phase(ctx, prev):
            latent = normalize(prev).to(pipe.device, dtype=torch.float16).unsqueeze(0)
            latent = denoise(latent, timesteps)[0].cpu().float()
            return pack(latent, latent_weight)
        return phase

    def decode(ctx, prev):
        """VAE-decode the fully denoised (phase 0) latent tile; re-weight for pixel blending."""
        latent = normalize(prev).to(pipe.device, dtype=torch.float16).unsqueeze(0)
        latent = latent / pipe.vae.config.scaling_factor
        with torch.no_grad():
            img = pipe.vae.decode(latent).sample
        img = (img / 2 + 0.5).clamp(0, 1)[0].cpu().float()
        return pack(img, pixel_weight)

    latents = store.get_or_create(
        f"phase{T - 1}",
        shape=(LATENT_CHANNELS + 1, LATENT_TILE, None),
        f=initial_phase,
        output_window=latent_window,
        cache_limit=None,
    )
    for i, timesteps in enumerate(phase_timesteps[1:], start=1):
        latents = store.get_or_create(
            f"phase{T - 1 - i}",
            shape=(LATENT_CHANNELS + 1, LATENT_TILE, None),
            f=make_continuation_phase(timesteps),
            output_window=latent_window,
            args=(latents,),
            args_windows=(latent_window,),
            cache_limit=None,
        )
    pixels = store.get_or_create(
        "image",
        shape=(3 + 1, PIXEL_TILE, None),
        f=decode,
        output_window=pixel_window,
        args=(latents,),
        args_windows=(latent_decode_window,),
        cache_limit=None,
    )

    region = normalize(torch.as_tensor(pixels[:, :, 0:CROP_PIXEL_WIDTH]))
    arr = (region.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
