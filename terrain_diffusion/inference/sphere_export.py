import json
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch

from terrain_diffusion.common.cli_helpers import parse_cache_size, parse_kwargs
from terrain_diffusion.inference.faces_to_obj import _cube_face_directions
from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _sample_from_indices(
    world: WorldPipeline,
    i_abs: np.ndarray,
    j_abs: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (ii, jj) in enumerate(zip(i_abs, j_abs)):
        groups[(int(ii // tile_size), int(jj // tile_size))].append(idx)

    elev = np.zeros((i_abs.shape[0],), dtype=np.float32)
    for (ti, tj), indices in groups.items():
        i0 = ti * tile_size
        j0 = tj * tile_size
        i1 = i0 + tile_size
        j1 = j0 + tile_size
        tile = world.get(i0, j0, i1, j1, with_climate=False)["elev"].detach().cpu().numpy()
        for idx in indices:
            li = i_abs[idx] - i0
            lj = j_abs[idx] - j0
            elev[idx] = float(tile[li, lj])
    return elev


def _sample_from_indices_bilinear(
    world: WorldPipeline,
    i_float: np.ndarray,
    j_float: np.ndarray,
    tile_size: int,
    j_period: int | None = None,
) -> np.ndarray:
    """Sample elevation with bilinear interpolation at fractional coordinates."""
    i0 = np.floor(i_float).astype(np.int64)
    j0 = np.floor(j_float).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1

    fi = (i_float - i0.astype(np.float64)).astype(np.float32)
    fj = (j_float - j0.astype(np.float64)).astype(np.float32)

    if j_period is not None:
        j0 = j0 % j_period
        j1 = j1 % j_period

    n = len(i_float)
    all_i = np.concatenate([i0, i0, i1, i1])
    all_j = np.concatenate([j0, j1, j0, j1])
    all_elev = _sample_from_indices(world, all_i, all_j, tile_size)

    e00 = all_elev[:n]
    e01 = all_elev[n : 2 * n]
    e10 = all_elev[2 * n : 3 * n]
    e11 = all_elev[3 * n :]

    return (
        e00 * (1 - fi) * (1 - fj)
        + e01 * (1 - fi) * fj
        + e10 * fi * (1 - fj)
        + e11 * fi * fj
    )


# Cube edge adjacency: (face_a, side_a, face_b, side_b, reversed)
# Tells which border of face_a matches which border of face_b.
_CUBE_EDGES = [
    (0, "top",    2, "right",  True),
    (0, "bottom", 3, "right",  False),
    (0, "left",   4, "right",  False),
    (0, "right",  5, "left",   False),
    (1, "top",    2, "left",   False),
    (1, "bottom", 3, "left",   True),
    (1, "left",   5, "right",  False),
    (1, "right",  4, "left",   False),
    (2, "top",    5, "top",    True),
    (2, "bottom", 4, "top",    False),
    (3, "top",    4, "bottom", False),
    (3, "bottom", 5, "bottom", True),
]

# Build lookup: for face_id and side, which (other_face, other_side, reversed)
_BORDER_NEIGHBOR: dict[tuple[int, str], tuple[int, str, bool]] = {}
for _fa, _sa, _fb, _sb, _rev in _CUBE_EDGES:
    _BORDER_NEIGHBOR[(_fa, _sa)] = (_fb, _sb, _rev)
    _BORDER_NEIGHBOR[(_fb, _sb)] = (_fa, _sa, _rev)


def _cross_face_atlas(
    face_id: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Atlas (i_float, j_float) for the cross layout."""
    eps = 1e-8
    s = fs - 1  # max pixel index within a face

    if face_id == 0:  # +X, col=2, row=1
        d = np.maximum(np.abs(x), eps)
        u = (-z / d + 1.0) * 0.5
        v = (y / d + 1.0) * 0.5
        j = 2 * fs + u * s
        i = (2 * fs - 1) - v * s
    elif face_id == 1:  # -X, col=0, row=1
        d = np.maximum(np.abs(x), eps)
        u = (z / d + 1.0) * 0.5
        v = (y / d + 1.0) * 0.5
        j = u * s
        i = (2 * fs - 1) - v * s
    elif face_id == 2:  # +Y, col=1, row=0
        d = np.maximum(np.abs(y), eps)
        u = (x / d + 1.0) * 0.5
        v = (z / d + 1.0) * 0.5
        j = fs + u * s
        i = v * s
    elif face_id == 3:  # -Y, col=1, row=2
        d = np.maximum(np.abs(y), eps)
        u = (x / d + 1.0) * 0.5
        v = (1.0 - z / d) * 0.5
        j = fs + u * s
        i = 2 * fs + v * s
    elif face_id == 4:  # +Z, col=1, row=1
        d = np.maximum(np.abs(z), eps)
        u = (x / d + 1.0) * 0.5
        v = (y / d + 1.0) * 0.5
        j = fs + u * s
        i = (2 * fs - 1) - v * s
    else:  # -Z, col=3, row=1
        d = np.maximum(np.abs(z), eps)
        u = (-x / d + 1.0) * 0.5
        v = (y / d + 1.0) * 0.5
        j = 3 * fs + u * s
        i = (2 * fs - 1) - v * s

    return i, j


def _sample_cubesphere_strip(
    world: WorldPipeline,
    dirs: np.ndarray,
    period: int,
    i_offset: int,
    j_offset: int,
    tile_size: int,
    face_size: int,
    blend_power: float = 20.0,
) -> np.ndarray:
    """Sample elevation for a strip of sphere directions using blend-weighted cubesphere."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    n = len(x)

    components = np.stack([x, -x, y, -y, z, -z], axis=1)
    raw = np.maximum(0.0, components) ** blend_power
    wsum = np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    weights = raw / wsum

    threshold = 0.005
    elev = np.zeros(n, dtype=np.float32)

    for proj_face_id in range(6):
        w = weights[:, proj_face_id]
        mask = w > threshold
        if not np.any(mask):
            continue

        i_f, j_f = _cross_face_atlas(
            proj_face_id, x[mask], y[mask], z[mask], face_size
        )

        face_elev = _sample_from_indices_bilinear(
            world, i_offset + i_f, j_offset + j_f, tile_size, j_period=period
        )
        elev[mask] += w[mask] * face_elev

    return elev


def _border_directions(face_id: int, res: int, side: str) -> np.ndarray:
    """Return (res, 3) unit sphere directions for one border of a face."""
    t = np.linspace(0.0, 1.0, res, dtype=np.float32)

    if side == "top":
        vr, uc = np.full(res, t[0]), t
    elif side == "bottom":
        vr, uc = np.full(res, t[-1]), t
    elif side == "left":
        vr, uc = t, np.full(res, t[0])
    else:  # right
        vr, uc = t, np.full(res, t[-1])

    if face_id in (0, 1, 4, 5):
        v = 1.0 - vr
    else:
        v = vr
    u = uc

    if face_id == 0:
        x, y, z = np.ones_like(u), 2*v - 1, 1 - 2*u
    elif face_id == 1:
        x, y, z = -np.ones_like(u), 2*v - 1, 2*u - 1
    elif face_id == 2:
        x, y, z = 2*u - 1, np.ones_like(u), 2*v - 1
    elif face_id == 3:
        x, y, z = 2*u - 1, -np.ones_like(u), -(2*v - 1)
    elif face_id == 4:
        x, y, z = 2*u - 1, 2*v - 1, np.ones_like(u)
    else:
        x, y, z = 1 - 2*u, 2*v - 1, -np.ones_like(u)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def _set_face_border(face: np.ndarray, side: str, vals: np.ndarray) -> None:
    """Write a 1D array to one border of a face grid."""
    if side == "top":      face[0, :] = vals
    elif side == "bottom": face[-1, :] = vals
    elif side == "left":   face[:, 0] = vals
    else:                  face[:, -1] = vals


def _export_cube_faces(
    world: WorldPipeline,
    period: int,
    i_offset: int,
    j_offset: int,
    tile_size: int,
    output_stem: Path,
    face_resolution: int | None = None,
    face_format: str = "png",
    diameter_m: float = 2000.0,
) -> dict:
    """Export 6 cube faces as heightmap images.

    Formats:
        png  — 16-bit lossless PNG (elevation normalized to uint16 per face,
               reconstruct via: elev = pixel / 65535 * (max - min) + min).
        tiff — 32-bit float TIFF (raw elevation in meters, truly lossless).

    Samples via blend-weighted cubesphere projection so all 12 cube edges
    are seamless (the cross-layout atlas only has 6 adjacent edges).
    """
    face_size = period // 4
    res = face_resolution or face_size
    fmt = face_format.lower()

    face_names = ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"]

    face_meta = {}
    strip_h = 256  # rows per strip for memory efficiency

    # Pass 1: Sample all faces (base elevation, no noise)
    all_elevations: dict[int, np.ndarray] = {}
    for face_id, name in enumerate(face_names):
        elev_grid = np.zeros((res, res), dtype=np.float32)
        for y0 in range(0, res, strip_h):
            y1 = min(y0 + strip_h, res)
            strip_dirs = _cube_face_directions(face_id, res)[y0 * res : y1 * res]
            elev_grid[y0:y1, :] = _sample_cubesphere_strip(
                world, strip_dirs, period, i_offset, j_offset, tile_size,
                face_size,
            ).reshape(y1 - y0, res)
        all_elevations[face_id] = elev_grid
        print(f"  {name}: sampled ({res}x{res})")

    # Enforce border consistency: sample each shared edge ONCE and assign
    # to both faces. The WorldPipeline's GPU diffusion is non-deterministic,
    # so sampling the same direction twice can give different values. By
    # sampling once and sharing, borders match exactly.
    print("  Enforcing border consistency...")
    for fa_id, sa, fb_id, sb, rev in _CUBE_EDGES:
        dirs = _border_directions(fa_id, res, sa)
        border_elev = _sample_cubesphere_strip(
            world, dirs, period, i_offset, j_offset, tile_size, face_size,
        )
        _set_face_border(all_elevations[fa_id], sa, border_elev)
        if rev:
            border_elev = border_elev[::-1]
        _set_face_border(all_elevations[fb_id], sb, border_elev)

    # Export faces
    for face_id, name in enumerate(face_names):
        elev_grid = all_elevations[face_id]
        emin, emax = float(elev_grid.min()), float(elev_grid.max())

        if fmt == "tiff":
            import tifffile
            face_path = output_stem.parent / f"{output_stem.stem}_face_{name}.tiff"
            tifffile.imwrite(str(face_path), elev_grid.astype(np.float32))
        else:
            from PIL import Image
            face_path = output_stem.parent / f"{output_stem.stem}_face_{name}.png"
            if emax > emin:
                normalized = ((elev_grid - emin) / (emax - emin) * 65535).astype(np.uint16)
            else:
                normalized = np.full((res, res), 32768, dtype=np.uint16)
            Image.fromarray(normalized).save(str(face_path), compress_level=9)

        face_meta[name] = {
            "file": face_path.name,
            "format": fmt,
            "resolution": res,
            "elev_min": emin,
            "elev_max": emax,
        }
        print(f"  {name}: {face_path.name} ({res}x{res}, [{emin:.1f}, {emax:.1f}]m)")

    del all_elevations
    return face_meta


def export_faces(
    model_path: str,
    output: str,
    hdf5_file: str | None = None,
    seed: int | None = None,
    diameter_m: float = 2000.0,
    i_offset: int = 0,
    j_offset: int = 0,
    device: str | None = None,
    caching_strategy: str = "indirect",
    tile_size: int = 256,
    face_resolution: int | None = None,
    face_format: str = "png",
    **kwargs,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: Using CPU (CUDA not available).")

    world = WorldPipeline.from_pretrained(model_path, seed=seed, caching_strategy=caching_strategy, **kwargs)
    world.to(device)
    if caching_strategy == "direct":
        world.bind(hdf5_file=hdf5_file)
    else:
        world.bind(hdf5_file or "TEMP")

    with world:
        period = kwargs.get("planet_period", world.kwargs.get("planet_period"))
        if period is None:
            raise ValueError("planet_period must be set for seamless spherical export.")
        period = int(period)
        if period <= 0 or period % 2 != 0:
            raise ValueError("planet_period must be a positive even integer.")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")

        out_path = Path(output)
        output_stem = out_path.with_suffix("") if out_path.suffix == ".json" else out_path

        print("Exporting cube faces...")
        face_meta = _export_cube_faces(
            world=world,
            period=period,
            i_offset=i_offset,
            j_offset=j_offset,
            tile_size=tile_size,
            output_stem=output_stem,
            face_resolution=face_resolution,
            face_format=face_format,
            diameter_m=diameter_m,
        )

        # Face-only JSON sidecar (raw terrain metadata)
        meta = {
            "seed": world.seed,
            "diameter_m": diameter_m,
            "planet_period": period,
            "native_resolution": world.native_resolution,
            "face_resolution": face_resolution or (period // 4),
            "face_format": face_format,
            "faces": face_meta,
        }
        meta_path = output_stem.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))

        print(f"World seed: {world.seed}")
        print(f"Wrote: {meta_path}")
        print(f"planet_period={period} (native {world.native_resolution}m)")


@click.command()
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.option("--output", default="outputs/planet.json", help="Output JSON path (face images use same stem)")
@click.option("--caching-strategy", type=click.Choice(["indirect", "direct"]), default="indirect", help="Caching strategy")
@click.option("--hdf5-file", default=None, help="HDF5 file path")
@click.option("--cache-size", default="100M", help="Cache size for direct caching")
@click.option("--seed", type=int, default=None, help="Random seed")
@click.option("--device", default=None, help="Device (cuda/cpu, default auto)")
@click.option("--batch-size", type=str, default="1,4", help="Latent batch size(s), e.g. '4' or '1,2,4,8'")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="info", help="Logging mode")
@click.option("--compile/--no-compile", "torch_compile", default=True, help="Use torch.compile")
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32", help="Model dtype")
@click.option("--diameter-m", type=float, default=2000.0, help="Sphere diameter in meters")
@click.option("--i-offset", type=int, default=0, help="Sampling offset in i (north-south) at native resolution")
@click.option("--j-offset", type=int, default=0, help="Sampling offset in j (east-west) at native resolution")
@click.option("--tile-size", type=int, default=256, help="Tile fetch size while sampling")
@click.option("--face-resolution", type=int, default=None, help="Cube face image resolution in pixels (default: native face_size = planet_period/4)")
@click.option("--face-format", type=click.Choice(["png", "tiff"]), default="png", help="Face image format: png (16-bit lossless) or tiff (float32 lossless)")
@click.option("--kwarg", "extra_kwargs", multiple=True, help="Additional key=value kwargs (e.g. --kwarg planet_period=16384)")
def main(
    model_path,
    output,
    caching_strategy,
    hdf5_file,
    cache_size,
    seed,
    device,
    batch_size,
    log_mode,
    torch_compile,
    dtype,
    diameter_m,
    i_offset,
    j_offset,
    tile_size,
    face_resolution,
    face_format,
    extra_kwargs,
):
    """Export cube face heightmaps using the Terrain Diffusion 90m world pipeline."""
    if caching_strategy == "indirect" and hdf5_file is None:
        hdf5_file = "TEMP"
    if hdf5_file is not None:
        hdf5_file = resolve_hdf5_path(hdf5_file)

    if "," in batch_size:
        batch_sizes = [int(x.strip()) for x in batch_size.split(",")]
    else:
        batch_sizes = int(batch_size)

    if dtype == "fp32":
        dtype = None

    export_faces(
        model_path=model_path,
        output=output,
        hdf5_file=hdf5_file,
        seed=seed,
        diameter_m=diameter_m,
        i_offset=i_offset,
        j_offset=j_offset,
        device=device,
        face_resolution=face_resolution,
        face_format=face_format,
        latents_batch_size=batch_sizes,
        log_mode=log_mode,
        torch_compile=torch_compile,
        dtype=dtype,
        caching_strategy=caching_strategy,
        cache_limit=parse_cache_size(cache_size),
        tile_size=tile_size,
        **parse_kwargs(extra_kwargs),
    )


if __name__ == "__main__":
    main()
