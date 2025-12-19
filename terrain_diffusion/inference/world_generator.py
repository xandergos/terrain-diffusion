import click
import torch
from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path
from terrain_diffusion.common.cli_helpers import parse_kwargs
from tqdm import tqdm


def generate_world(model_path: str, hdf5_file: str, seed: int | None = None, coarse_window: int = 64, device: str | None = None, **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Warning: Using CPU (CUDA not available).")

    world = WorldPipeline.from_pretrained(model_path, seed=seed, **kwargs)
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
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.option("--hdf5-file", default="TEMP", help="Output HDF5 file path (use 'TEMP' for temporary file)")
@click.option("--seed", type=int, default=None, help="Random seed (default: random or from file)")
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--batch-size", type=str, default="4", help="Batch size(s) for latent generation (e.g. '4' or '1,2,4,8')")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose", help="Logging mode")
@click.option("--coarse-window", type=int, default=50, help="Coarse window size")
@click.option("--compile", "torch_compile", is_flag=True, help="Use torch.compile for faster inference")
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default=None, help="Model dtype (default: fp32)")
@click.option("--kwarg", "extra_kwargs", multiple=True, help="Additional key=value kwargs (e.g. --kwarg coarse_pooling=2)")
def main(model_path, hdf5_file, seed, device, batch_size, log_mode, coarse_window, torch_compile, dtype, extra_kwargs):
    """Generate a world using the terrain diffusion pipeline"""
    hdf5_file = resolve_hdf5_path(hdf5_file)
    # Parse batch size(s)
    if ',' in batch_size:
        batch_sizes = [int(x.strip()) for x in batch_size.split(',')]
    else:
        batch_sizes = int(batch_size)
    # Normalize dtype
    if dtype == 'fp32':
        dtype = None
    generate_world(
        model_path,
        hdf5_file,
        seed=seed,
        coarse_window=coarse_window,
        device=device,
        latents_batch_size=batch_sizes,
        log_mode=log_mode,
        torch_compile=torch_compile,
        dtype=dtype,
        **parse_kwargs(extra_kwargs),
    )


if __name__ == "__main__":
    main()