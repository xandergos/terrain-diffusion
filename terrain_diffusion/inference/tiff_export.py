"""
Export terrain to GeoTIFF from a conditioning TIFF directory (e.g. azgaar-output/).

Imports all conditioning channels (elevation, temperature, precipitation, etc.) with
64-cell edge padding so the model has smooth context at the borders. The padding is
stripped from the output.

Usage:
  python -m terrain_diffusion.inference.tiff_export azgaar-output/ output.tif
  python -m terrain_diffusion.inference.tiff_export azgaar-output/ output.tif --snr 1.0,0.5,2.0,0.5,2.0
"""

import click
import numpy as np
import rasterio
import torch
from pathlib import Path
from rasterio.transform import Affine
from tqdm import tqdm

from terrain_diffusion.common.cli_helpers import parse_cache_size
from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path

PADDING = 64
PIXELS_PER_CELL = 256

# (filename, channel_index, internal_scale, default_value)
# internal_scale: multiplier to convert TIFF units to pipeline internal units
#   T std (ch 2) is stored as °C×100 internally but TIFFs are in °C, so scale=100
# default_value: fill for out-of-bounds conditioning (elevation uses -1000 = deep ocean)
CHANNEL_FILES = [
    ("heightmap.tif",        0, 1.0,   -1000.0),
    ("temperature.tif",      1, 1.0,   None),
    ("temperature_std.tif",  2, 100.0, None),
    ("precipitation.tif",    3, 1.0,   None),
    ("precipitation_cv.tif", 4, 1.0,   None),
]


def _load_and_pad(path: Path, channel: int, internal_scale: float, default_value: float | None) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    fill = default_value if default_value is not None else 0.0
    arr = np.where(np.isfinite(arr), arr, fill)

    if internal_scale != 1.0:
        arr = arr * internal_scale

    return np.pad(arr, PADDING, mode="edge")


@click.command()
@click.argument("tiff_dir", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.option(
    "--snr",
    metavar="ELEV,TEMP,T_STD,PRECIP,P_CV",
    help=(
        "Conditioning strength per channel (same as refinement strength in the editor). "
        "Exactly 5 comma-separated values, e.g. 1.0,0.5,2.0,0.5,2.0"
    ),
)
@click.option("--hdf5-file", default=None, help="HDF5 cache file ('TEMP' for temporary)")
@click.option("--cache-size", default="100M", help="Cache size for direct caching (e.g. 100M, 1G)")
@click.option("--seed", type=int, default=None)
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--batch-size", default="1,2,4,8,16")
@click.option("--compile/--no-compile", "torch_compile", default=True)
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32")
@click.option("--caching-strategy", type=click.Choice(["indirect", "direct"]), default="direct")
def main(tiff_dir, output, model_path, snr, hdf5_file, cache_size, seed, device,
         batch_size, torch_compile, dtype, caching_strategy):
    """Generate terrain from conditioning TIFFs and export to GeoTIFF."""
    tiff_dir = Path(tiff_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: Using CPU (CUDA not available).")

    batch_sizes = [int(x) for x in batch_size.split(",")] if "," in batch_size else int(batch_size)
    if dtype == "fp32":
        dtype = None

    world = WorldPipeline.from_pretrained(
        model_path,
        seed=seed,
        latents_batch_size=batch_sizes,
        torch_compile=torch_compile,
        dtype=dtype,
        caching_strategy=caching_strategy,
        cache_limit=parse_cache_size(cache_size),
    )
    world.to(device)

    if snr:
        try:
            snr_vals = [float(x.strip()) for x in snr.split(",")]
        except ValueError:
            raise click.UsageError("--snr values must be numbers (e.g. 1.0,0.5,2.0,0.5,2.0).")
        if len(snr_vals) != 5:
            raise click.UsageError("--snr must have exactly 5 comma-separated values (e.g. 1.0,0.5,2.0,0.5,2.0).")
        world.set_cond_snr(snr_vals)

    if caching_strategy == "direct":
        world.bind(hdf5_file=resolve_hdf5_path(hdf5_file) if hdf5_file else None)
    else:
        world.bind(resolve_hdf5_path(hdf5_file) if hdf5_file else "TEMP")

    print(f"World seed: {world.seed}")

    ref_transform = None
    ref_crs = None
    H_orig = W_orig = None

    for filename, channel, internal_scale, default_value in CHANNEL_FILES:
        path = tiff_dir / filename
        if not path.exists():
            print(f"  Skipping {filename} (not found). Perlin noise will be used instead.")
            continue

        with rasterio.open(path) as ds:
            if ref_transform is None:
                ref_transform = ds.transform
                ref_crs = ds.crs
                H_orig, W_orig = ds.height, ds.width

        padded = _load_and_pad(path, channel, internal_scale, default_value)
        world.set_custom_conditioning_import(channel, padded, 0, 0, default_value=default_value)
        print(f"  Imported {filename} → channel {channel}, padded shape: {padded.shape}")

    if ref_transform is None:
        raise click.UsageError("No conditioning TIFFs found in the directory.")

    out_h = H_orig * PIXELS_PER_CELL
    out_w = W_orig * PIXELS_PER_CELL
    out_transform = Affine(
        ref_transform.a / PIXELS_PER_CELL, ref_transform.b, ref_transform.c,
        ref_transform.d, ref_transform.e / PIXELS_PER_CELL, ref_transform.f,
    )

    print(f"Output: {output} ({out_w}x{out_h} px)")

    chunk_cells = 8
    chunk_px = chunk_cells * PIXELS_PER_CELL
    row_chunks = (H_orig + chunk_cells - 1) // chunk_cells
    col_chunks = (W_orig + chunk_cells - 1) // chunk_cells

    with world:
        with rasterio.open(
            output, "w",
            driver="GTiff", height=out_h, width=out_w,
            count=1, dtype="float32",
            crs=ref_crs, transform=out_transform,
            compress="lzw", tiled=True, blockxsize=256, blockysize=256,
        ) as dst:
            with tqdm(total=row_chunks * col_chunks, desc="Generating") as pbar:
                for ci in range(0, H_orig, chunk_cells):
                    for cj in range(0, W_orig, chunk_cells):
                        ci2 = min(ci + chunk_cells, H_orig)
                        cj2 = min(cj + chunk_cells, W_orig)

                        pi1 = (PADDING + ci) * PIXELS_PER_CELL
                        pi2 = (PADDING + ci2) * PIXELS_PER_CELL
                        pj1 = (PADDING + cj) * PIXELS_PER_CELL
                        pj2 = (PADDING + cj2) * PIXELS_PER_CELL

                        result = world.get(pi1, pj1, pi2, pj2, with_climate=False)
                        elev = result["elev"].numpy().astype(np.float32)

                        window = rasterio.windows.Window(
                            cj * PIXELS_PER_CELL, ci * PIXELS_PER_CELL,
                            elev.shape[1], elev.shape[0],
                        )
                        dst.write(elev, 1, window=window)
                        pbar.update(1)


if __name__ == "__main__":
    main()
