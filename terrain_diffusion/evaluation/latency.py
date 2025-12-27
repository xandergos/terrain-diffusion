"""
Measure generation latency (TTFT and TTST).

TTFT: Time to First Tile - delay from initialization to first tile
TTST: Time to Second Tile - time to generate adjacent tile thereafter
"""
import math
import time
import random
from tqdm import tqdm
import click
import torch

from terrain_diffusion.inference.world_pipeline import WorldPipeline

separation = 200 * 256  # Separation between tiles to guarantee no cache overlap


def measure_latency(
    device: str = 'cuda',
    seed: int = 42,
    onestep_latent: bool = False,
    tile_size: int = 512,
    grid_aligned: bool = False,
    num_runs: int = 100,
    decoder_tile_size: int = 512,
    decoder_tile_stride: int = 384,
    max_batch_size: int = 16,
) -> dict:
    """Measure TTFT and TTST."""
    # Reset peak memory before warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    assert 2**round(math.log2(max_batch_size)) == max_batch_size
        
    use_cuda = device == 'cuda'
    world = WorldPipeline.from_local_models(
        seed=seed,
        latents_batch_size=[2**i for i in range(round(math.log2(max_batch_size)) + 1)],
        torch_compile=use_cuda, 
        dtype='fp32',
        caching_strategy='direct',
        cache_limit=None,
        onestep_latent=onestep_latent,
        decoder_tile_size=decoder_tile_size,
        decoder_tile_stride=decoder_tile_stride,
    )
    world.to(device)
    world.bind("TEMP")
    
    # Warmup: generate one tile to initialize all models
    _ = world.get(0, 0, tile_size, tile_size, with_climate=False)
    if use_cuda:
        torch.cuda.synchronize()
    
    ttft_times = []
    ttst_times = []
    
    pbar = tqdm(total=num_runs, desc="Measuring latency")
    for run in range(num_runs):
        # Random base location, far from previous runs
        if grid_aligned:
            base_i = ((run + 1) * separation // tile_size) * tile_size + random.randint(0, separation // (10 * tile_size)) * tile_size
            base_j = random.randint(0, separation // tile_size) * tile_size
        else:
            base_i = (run + 1) * separation + random.randint(0, separation // 10)
            base_j = random.randint(0, separation)
        
        # TTFT: first tile
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = world.get(base_i, base_j, base_i + tile_size, base_j + tile_size, with_climate=False)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ttft_times.append(t1 - t0)
        
        # TTST: adjacent tile (shifted by tile_size in j direction)
        adj_j = base_j + tile_size
        if use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        _ = world.get(base_i, adj_j, base_i + tile_size, adj_j + tile_size, with_climate=False)
        if use_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        ttst_times.append(t3 - t2)
        
        # Clear cache after each run
        world.empty_cache()

        pbar.update(1)
        pbar.set_postfix({
            'TTFT': sum(ttft_times) / len(ttft_times),
            'TTST': sum(ttst_times) / len(ttst_times),
        })
    pbar.close()
    
    # Get peak VRAM usage
    peak_vram_mb = None
    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    world.close()
    
    def percentile(data, p):
        s = sorted(data)
        k = int((len(s) - 1) * p / 100 + 0.5)
        return s[k]
    
    return {
        'ttft_mean': sum(ttft_times) / len(ttft_times),
        'ttst_mean': sum(ttst_times) / len(ttst_times),
        'ttft_std': (sum((t - sum(ttft_times)/len(ttft_times))**2 for t in ttft_times) / len(ttft_times)) ** 0.5,
        'ttst_std': (sum((t - sum(ttst_times)/len(ttst_times))**2 for t in ttst_times) / len(ttst_times)) ** 0.5,
        'ttft_p5': percentile(ttft_times, 5),
        'ttft_p50': percentile(ttft_times, 50),
        'ttft_p95': percentile(ttft_times, 95),
        'ttst_p5': percentile(ttst_times, 5),
        'ttst_p50': percentile(ttst_times, 50),
        'ttst_p95': percentile(ttst_times, 95),
        'peak_vram_mb': peak_vram_mb,
    }


@click.command()
@click.option('--onestep-latent', is_flag=True, default=False, help='Use 1-step latent model instead of 2-step')
@click.option('--cpu', is_flag=True, default=False, help='Use CPU instead of GPU (disables compilation)')
@click.option('--tile-size', default=512, type=int, help='Size of tiles to generate')
@click.option('--grid-aligned', is_flag=True, default=False, help='Align base coordinates to tile_size multiples')
@click.option('-n', '--num-runs', default=100, type=int, help='Number of measurement runs')
@click.option('--decoder-tile-size', default=512, type=int, help='Decoder tile size')
@click.option('--decoder-stride', default=384, type=int, help='Decoder tile stride')
@click.option('--max-batch-size', default=16, type=int, help='Maximum batch size')
def main(onestep_latent, cpu, tile_size, grid_aligned, num_runs, decoder_tile_size, decoder_stride, max_batch_size):
    if cpu:
        device = 'cpu'
        print("Note: torch.compile disabled on CPU")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Note: CUDA not available, falling back to CPU (no torch.compile)")
    
    print(f"Using device: {device}")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Decoder tile: {decoder_tile_size}x{decoder_tile_size}, stride: {decoder_stride}")
    print(f"Number of runs: {num_runs}")
    print(f"Latent steps: {'1-step' if onestep_latent else '2-step'}")
    print(f"Grid aligned: {grid_aligned}\n")
    
    result = measure_latency(
        device=device,
        onestep_latent=onestep_latent,
        tile_size=tile_size,
        grid_aligned=grid_aligned,
        num_runs=num_runs,
        decoder_tile_size=decoder_tile_size,
        decoder_tile_stride=decoder_stride,
        max_batch_size=max_batch_size,
    )
    print(f"\nTTFT: {result['ttft_mean']:.2f}s ± {result['ttft_std']:.2f}s (p5={result['ttft_p5']:.2f}, p50={result['ttft_p50']:.2f}, p95={result['ttft_p95']:.2f})")
    print(f"TTST: {result['ttst_mean']:.2f}s ± {result['ttst_std']:.2f}s (p5={result['ttst_p5']:.2f}, p50={result['ttst_p50']:.2f}, p95={result['ttst_p95']:.2f})")
    if result['peak_vram_mb'] is not None:
        print(f"Peak VRAM: {result['peak_vram_mb']:.0f} MB")


if __name__ == '__main__':
    main()
