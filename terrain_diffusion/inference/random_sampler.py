"""Automated random terrain sampling with debug overlay."""

import os
import random
import click
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from terrain_diffusion.inference.world_pipeline import WorldPipeline, normalize_tensor, resolve_hdf5_path
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.common.cli_helpers import parse_kwargs, parse_cache_size

CHANNEL_NAMES = ['Elev', 'p5', 'Temp', 'T std', 'Precip', 'Precip CV']


def add_debug_overlay(image: np.ndarray, climate_info: dict, coords: dict) -> np.ndarray:
    """Add a transparent black debug box with climate info to the top-left corner."""
    img = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Build debug text
    lines = [
        f"Elev: {climate_info['elev_min']:.0f} - {climate_info['elev_max']:.0f} m",
        f"Temp: {climate_info['temp']:.1f} °C",
        f"Temp std: {climate_info['temp_std']/100:.1f} °C",
        f"Precip: {climate_info['precip']:.0f} mm",
        f"Precip CV: {climate_info['precip_cv']:.1f}%",
    ]
    text = "\n".join(lines)
    
    # Try to use a monospace font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 28)
        except (IOError, OSError):
            font = ImageFont.load_default()
    
    # Calculate text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw semi-transparent black box
    padding = 10
    box_coords = [padding, padding, padding + text_width + 20, padding + text_height + 20]
    draw.rectangle(box_coords, fill=(0, 0, 0, 180))
    
    # Draw text
    draw.text((padding + 10, padding + 10), text, fill=(255, 255, 255, 255), font=font)
    
    # Composite overlay onto image
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert('RGB')).astype(np.float32) / 255.0


def sample_land_tiles(world: WorldPipeline, coarse_window: int, detail_size: int, min_land_frac: float = 0.5, n_samples: int = 10) -> list:
    """Sample random coarse tiles with at least min_land_frac land coverage.
    
    Args:
        world: WorldPipeline instance
        coarse_window: Size of coarse window to sample from
        detail_size: Size of detail tile in native pixels (used to determine coarse tile coverage)
        min_land_frac: Minimum fraction of land (elevation > 0) in the tile
        n_samples: Number of samples to return
    """
    ci0, ci1 = -coarse_window, coarse_window
    cj0, cj1 = -coarse_window, coarse_window
    
    # Get normalized coarse map
    coarse = normalize_tensor(world.coarse[:, ci0:ci1, cj0:cj1], dim=0)
    coarse_elev_ss = coarse[0]  # signed-sqrt elevation
    coarse_elev_m = torch.sign(coarse_elev_ss) * torch.square(coarse_elev_ss)
    
    land_mask = (coarse_elev_m > 0).float()
    H, W = land_mask.shape
    
    # Each coarse pixel covers 256 native pixels, so detail_size covers this many coarse pixels
    tile_coarse_size = detail_size // 256
    half = tile_coarse_size // 2
    
    valid_tiles = []
    for i in range(half, H - half):
        for j in range(half, W - half):
            # Check land fraction in the coarse pixels covered by this detail tile
            tile_land = land_mask[i - half:i + half, j - half:j + half]
            land_frac = tile_land.mean().item()
            if land_frac >= min_land_frac:
                valid_tiles.append((ci0 + i, cj0 + j))
    
    if len(valid_tiles) < n_samples:
        print(f"Warning: Only found {len(valid_tiles)} valid land tiles (requested {n_samples})")
        return valid_tiles
    
    return random.sample(valid_tiles, n_samples)


def get_coarse_climate_info(world: WorldPipeline, ci: int, cj: int) -> dict:
    """Extract climate info from coarse map for a tile."""
    # Get a small window around the tile
    coarse = normalize_tensor(world.coarse[:, ci:ci+1, cj:cj+1], dim=0)
    
    # Convert to real units
    elev_ss = coarse[0].item()
    elev_m = np.sign(elev_ss) * elev_ss ** 2
    
    p5_ss = coarse[1].item()
    p5_m = np.sign(p5_ss) * p5_ss ** 2
    
    return {
        'temp': coarse[2].item(),
        'temp_std': coarse[3].item(),
        'precip': coarse[4].item(),
        'precip_cv': coarse[5].item(),
    }


def run_sampler(
    model_path: str,
    output_dir: str,
    n_samples: int = 10,
    detail_size: int = 1024,
    coarse_window: int = 50,
    seed: int | None = None,
    device: str | None = None,
    min_land_frac: float = 0.5,
    caching_strategy: str = 'direct',
    **kwargs
) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Warning: Using CPU (CUDA not available).")
    
    os.makedirs(output_dir, exist_ok=True)
    
    world = WorldPipeline.from_pretrained(model_path, seed=seed, caching_strategy=caching_strategy, **kwargs)
    world.to(device)
    world.bind(hdf5_file='TEMP')
    
    with world:
        print(f"World seed: {world.seed}")
        
        # Sample random land tiles
        print(f"Sampling {n_samples} land tiles (min {min_land_frac*100:.0f}% land)...")
        tiles = sample_land_tiles(world, coarse_window, detail_size, min_land_frac, n_samples)
        print(f"Found {len(tiles)} tiles to process")
        
        for idx, (ci, cj) in enumerate(tiles):
            print(f"\n[{idx+1}/{len(tiles)}] Processing tile (ci={ci}, cj={cj})...")
            
            # Get climate info from coarse map
            climate_info = get_coarse_climate_info(world, ci, cj)
            
            # Map coarse tile to native resolution
            center_i = ci * 256
            center_j = cj * 256
            half = detail_size // 2
            i1, i2 = center_i - half, center_i + half
            j1, j2 = center_j - half, center_j + half
            
            # Generate terrain
            region_dict = world.get(i1, j1, i2, j2)
            elev = region_dict['elev'].cpu().numpy()
            
            # Add elevation stats to climate info
            climate_info['elev_min'] = elev.min()
            climate_info['elev_max'] = elev.max()
            
            # Generate relief map
            relief_rgb = get_relief_map(elev, None, None, None, resolution=world.native_resolution)
            
            # Add debug overlay
            coords = {'ci': ci, 'cj': cj, 'seed': world.seed}
            relief_with_overlay = add_debug_overlay(relief_rgb, climate_info, coords)
            
            # Save
            filename = f"{idx}.png"
            filepath = os.path.join(output_dir, filename)
            Image.fromarray((np.clip(relief_with_overlay, 0, 1) * 255).astype(np.uint8)).save(filepath)
            print(f"  Saved: {filepath}")
        
        print(f"\nDone! Generated {len(tiles)} samples in {output_dir}")


@click.command()
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.option("--output-dir", default="results/random_images", help="Output directory for images")
@click.option("--n-samples", type=int, default=100, help="Number of random samples to generate")
@click.option("--detail-size", type=int, default=1024, help="Size of each detail patch in native pixels")
@click.option("--coarse-window", type=int, default=300, help="Coarse window size for sampling")
@click.option("--seed", type=int, default=None, help="Random seed (default: random)")
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--min-land-frac", type=float, default=0.5, help="Minimum land fraction for tile selection")
@click.option("--caching-strategy", type=click.Choice(["indirect", "direct"]), default="direct")
@click.option("--cache-size", default="1G", help="Cache size for direct caching")
@click.option("--batch-size", type=str, default="1,2,4,8,16,32", help="Batch size(s) for latent generation")
@click.option("--compile/--no-compile", "torch_compile", default=True, help="Use torch.compile")
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32", help="Model dtype")
@click.option("--kwarg", "extra_kwargs", multiple=True, help="Additional key=value kwargs")
def main(
    model_path, output_dir, n_samples, detail_size, coarse_window, seed, device,
    min_land_frac, caching_strategy, cache_size, batch_size, torch_compile, dtype, extra_kwargs
):
    """Generate random terrain samples with debug overlays."""
    # Parse batch size(s)
    if ',' in batch_size:
        batch_sizes = [int(x.strip()) for x in batch_size.split(',')]
    else:
        batch_sizes = int(batch_size)
    
    if dtype == 'fp32':
        dtype = None
    
    run_sampler(
        model_path,
        output_dir,
        n_samples=n_samples,
        detail_size=detail_size,
        coarse_window=coarse_window,
        seed=seed,
        device=device,
        min_land_frac=min_land_frac,
        caching_strategy=caching_strategy,
        cache_limit=parse_cache_size(cache_size),
        latents_batch_size=batch_sizes,
        torch_compile=torch_compile,
        dtype=dtype,
        **parse_kwargs(extra_kwargs),
    )


if __name__ == '__main__':
    main()

