"""
Measure generation latency (TTFT and TTST) across resolutions.

TTFT: Time to First Tile - delay from initialization to first 512x512 tile
TTST: Time to Second Tile - time to generate adjacent 512x512 tile thereafter
"""
import time
import random
from tempfile import NamedTemporaryFile
from tqdm import tqdm
import torch

from terrain_diffusion.inference.world_pipeline import WorldPipeline


TILE_SIZE = 512
NUM_RUNS = 100


def measure_latency(resolution: int, device: str = 'cuda', seed: int = 42) -> dict:
    """Measure TTFT and TTST for a given resolution."""
    
    # Resolution to method mapping
    get_fn_name = {90: 'get_90', 45: 'get_45', 22: 'get_22', 11: 'get_11'}[resolution]
    
    # Separation between tiles to avoid cache overlap
    separation = int(200 * 256 * 90 / resolution)
    
    with NamedTemporaryFile(suffix='.h5') as tmp_file:
        world = WorldPipeline(
            tmp_file.name,
            device=device,
            seed=seed,
            mode="a",
            latents_batch_size=32,
        )
        get_fn = getattr(world, get_fn_name)
        
        # Warmup: generate one tile to initialize all models
        _ = get_fn(0, 0, TILE_SIZE, TILE_SIZE, with_climate=False)
        torch.cuda.synchronize()
        
        ttft_times = []
        ttst_times = []
        
        pbar = tqdm(total=NUM_RUNS, desc=f"Measuring {resolution}m resolution")
        for run in range(NUM_RUNS):
            # Random base location, far from previous runs
            base_i = (run + 1) * separation + random.randint(0, separation // 10)
            base_j = random.randint(0, separation)
            
            # TTFT: first tile
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = get_fn(base_i, base_j, base_i + TILE_SIZE, base_j + TILE_SIZE, with_climate=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ttft_times.append(t1 - t0)
            
            # TTST: adjacent tile (shifted by TILE_SIZE in j direction)
            adj_j = base_j + TILE_SIZE
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            _ = get_fn(base_i, adj_j, base_i + TILE_SIZE, adj_j + TILE_SIZE, with_climate=False)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            ttst_times.append(t3 - t2)

            pbar.update(1)
            pbar.set_postfix({
                'TTFT': sum(ttft_times) / len(ttft_times),
                'TTST': sum(ttst_times) / len(ttst_times),
            })
        pbar.close()
        world.close()
    
    return {
        'resolution': resolution,
        'ttft_mean': sum(ttft_times) / len(ttft_times),
        'ttst_mean': sum(ttst_times) / len(ttst_times),
        'ttft_std': (sum((t - sum(ttft_times)/len(ttft_times))**2 for t in ttft_times) / len(ttft_times)) ** 0.5,
        'ttst_std': (sum((t - sum(ttst_times)/len(ttst_times))**2 for t in ttst_times) / len(ttst_times)) ** 0.5,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"Number of runs per resolution: {NUM_RUNS}\n")
    
    resolutions = [90, 45, 22, 11]
    results = []
    
    for res in resolutions:
        print(f"Measuring {res}m resolution...")
        result = measure_latency(res, device=device)
        results.append(result)
        print(f"  TTFT: {result['ttft_mean']:.2f}s ± {result['ttft_std']:.2f}s")
        print(f"  TTST: {result['ttst_mean']:.2f}s ± {result['ttst_std']:.2f}s\n")
    
    # Print table
    print("\n" + "=" * 50)
    print("Generation latency at different resolutions")
    print("=" * 50)
    print(f"{'Resolution':<12} {'TTFT (s)':<15} {'TTST (s)':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['resolution']}m{'':<9} {r['ttft_mean']:.1f}s{'':<12} {r['ttst_mean']:.1f}s")
    print("=" * 50)


if __name__ == '__main__':
    main()

