import json
import click
import torch
from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path
from terrain_diffusion.common.cli_helpers import parse_kwargs
from tqdm import tqdm


def generate_world(hdf5_file: str, seed: int | None = None, coarse_window: int = 64, device: str | None = None, **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Warning: Using CPU (CUDA not available).")

    world = WorldPipeline.from_local_models(seed=seed, **kwargs)
    world.to(device)
    world.bind(hdf5_file)
    with world:
        print(f"World seed: {world.seed}")
        ci0, ci1 = -coarse_window, coarse_window
        cj0, cj1 = -coarse_window, coarse_window
        
        tile_size = 2048
        pbar = tqdm(total=((ci1-ci0)//8)*((cj1-cj0)//8), desc="Generating world")
        for i in range(ci0, ci1, tile_size//256):
            for j in range(cj0, cj1, tile_size//256):
                world.residual[:, i*256:i*256+tile_size, j*256:j*256+tile_size]
                pbar.update(1)


@click.command()
@click.option("--hdf5-file", default="world.h5", help="Output HDF5 file path (use 'TEMP' for temporary file)")
@click.option("--seed", type=int, default=None, help="Random seed (default: random or from file)")
@click.option("--coarse-window", type=int, default=50, help="Coarse window size")
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--drop-water-pct", type=float, default=0.5, help="Drop water percentage")
@click.option("--frequency-mult", default="[1.0, 1.0, 1.0, 1.0, 1.0]", help="Frequency multipliers (JSON)")
@click.option("--cond-snr", default="[0.5, 0.5, 0.5, 0.5, 0.5]", help="Conditioning SNR (JSON)")
@click.option("--histogram-raw", default="[0.0, 0.0, 0.0, 1.0, 1.5]", help="Histogram raw values (JSON)")
@click.option("--latents-batch-size", type=int, default=4, help="Batch size for latent generation")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose", help="Logging mode")
@click.option("--kwarg", "extra_kwargs", multiple=True, help="Additional key=value kwargs (e.g. --kwarg coarse_pooling=2)")
def main(hdf5_file, seed, coarse_window, device, drop_water_pct, frequency_mult, cond_snr, histogram_raw, latents_batch_size, log_mode, extra_kwargs):
    """Generate a world using the terrain diffusion pipeline"""
    hdf5_file = resolve_hdf5_path(hdf5_file)
    generate_world(
        hdf5_file,
        seed=seed,
        coarse_window=coarse_window,
        device=device,
        drop_water_pct=drop_water_pct,
        frequency_mult=json.loads(frequency_mult),
        cond_snr=json.loads(cond_snr),
        histogram_raw=json.loads(histogram_raw),
        latents_batch_size=latents_batch_size,
        log_mode=log_mode,
        **parse_kwargs(extra_kwargs),
    )


if __name__ == "__main__":
    main()