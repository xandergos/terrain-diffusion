"""
Measure generation latency (TTFT and TTST).

TTFT: Time to First Tile - delay from initialization to first 512x512 tile
TTST: Time to Second Tile - time to generate adjacent 512x512 tile thereafter
"""
import time
import random
from tqdm import tqdm
import torch

from terrain_diffusion.inference.world_pipeline import WorldPipeline


TILE_SIZE = 512
NUM_RUNS = 100
SEPARATION = 200 * 256  # Separation between tiles to avoid cache overlap


def measure_latency(device: str = 'cuda', seed: int = 42) -> dict:
    """Measure TTFT and TTST."""
    world = WorldPipeline.from_local_models(
        seed=seed,
        latents_batch_size=[1, 2, 4, 8, 16, 32],
        torch_compile=True,
        dtype='fp16',
        caching_strategy='direct',
        cache_limit=None,
    )
    world.to(device)
    world.bind("TEMP")
    
    # Warmup: generate one tile to initialize all models
    _ = world.get(0, 0, TILE_SIZE, TILE_SIZE, with_climate=False)
    torch.cuda.synchronize()
    
    ttft_times = []
    ttst_times = []
    
    pbar = tqdm(total=NUM_RUNS, desc="Measuring latency")
    for run in range(NUM_RUNS):
        # Random base location, far from previous runs
        base_i = (run + 1) * SEPARATION + random.randint(0, SEPARATION // 10)
        base_j = random.randint(0, SEPARATION)
        
        # TTFT: first tile
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = world.get(base_i, base_j, base_i + TILE_SIZE, base_j + TILE_SIZE, with_climate=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ttft_times.append(t1 - t0)
        
        # TTST: adjacent tile (shifted by TILE_SIZE in j direction)
        adj_j = base_j + TILE_SIZE
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        _ = world.get(base_i, adj_j, base_i + TILE_SIZE, adj_j + TILE_SIZE, with_climate=False)
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
    world.close()
    
    return {
        'ttft_mean': sum(ttft_times) / len(ttft_times),
        'ttst_mean': sum(ttst_times) / len(ttst_times),
        'ttft_std': (sum((t - sum(ttft_times)/len(ttft_times))**2 for t in ttft_times) / len(ttft_times)) ** 0.5,
        'ttst_std': (sum((t - sum(ttst_times)/len(ttst_times))**2 for t in ttst_times) / len(ttst_times)) ** 0.5,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"Number of runs: {NUM_RUNS}\n")
    
    result = measure_latency(device=device)
    print(f"\nTTFT: {result['ttft_mean']:.2f}s ± {result['ttft_std']:.2f}s")
    print(f"TTST: {result['ttst_mean']:.2f}s ± {result['ttst_std']:.2f}s")


if __name__ == '__main__':
    main()
