import json
import math
from pathlib import Path

import click
import numpy as np
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType

from terrain_diffusion.inference.faces_to_obj import _load_face_elevation, _FACE_NAMES

# Perlin noise for elevation detail (same seeds/config as minecraft_api.py)
# Coarse layer: ~24-pixel wavelength, 3-octave FBm
_FACE_NOISE_COARSE = FastNoiseLite(seed=99999)
_FACE_NOISE_COARSE.noise_type = NoiseType.NoiseType_Perlin
_FACE_NOISE_COARSE.frequency = 1.0 / 24.0
_FACE_NOISE_COARSE.fractal_type = FractalType.FractalType_FBm
_FACE_NOISE_COARSE.fractal_octaves = 3
_FACE_NOISE_COARSE.fractal_lacunarity = 2.0
_FACE_NOISE_COARSE.fractal_gain = 0.5

# Fine layer: ~6-pixel wavelength, 2-octave FBm
_FACE_NOISE_FINE = FastNoiseLite(seed=88888)
_FACE_NOISE_FINE.noise_type = NoiseType.NoiseType_Perlin
_FACE_NOISE_FINE.frequency = 1.0 / 6.0
_FACE_NOISE_FINE.fractal_type = FractalType.FractalType_FBm
_FACE_NOISE_FINE.fractal_octaves = 2
_FACE_NOISE_FINE.fractal_lacunarity = 2.0
_FACE_NOISE_FINE.fractal_gain = 0.6

# Cube edge adjacency: (face_a, side_a, face_b, side_b, reversed)
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

_BORDER_NEIGHBOR: dict[tuple[int, str], tuple[int, str, bool]] = {}
for _fa, _sa, _fb, _sb, _rev in _CUBE_EDGES:
    _BORDER_NEIGHBOR[(_fa, _sa)] = (_fb, _sb, _rev)
    _BORDER_NEIGHBOR[(_fb, _sb)] = (_fa, _sa, _rev)


def _get_adjacent_row(
    all_elevations: dict[int, np.ndarray],
    face_id: int,
    side: str,
) -> np.ndarray:
    """Get the row of pixels just inside the border of the adjacent face."""
    other_face, other_side, rev = _BORDER_NEIGHBOR[(face_id, side)]
    other_elev = all_elevations[other_face]
    if other_side == "top":      row = other_elev[1, :]
    elif other_side == "bottom": row = other_elev[-2, :]
    elif other_side == "left":   row = other_elev[:, 1]
    else:                        row = other_elev[:, -1 - 1]
    if rev:
        row = row[::-1]
    return row


def _pad_face_with_neighbors(
    face_elev: np.ndarray,
    face_id: int,
    all_elevations: dict[int, np.ndarray],
) -> np.ndarray:
    """Pad a face elevation grid with 1 pixel from adjacent faces at all 4 borders."""
    res = face_elev.shape[0]
    padded = np.zeros((res + 2, res + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = face_elev

    padded[0, 1:-1] = _get_adjacent_row(all_elevations, face_id, "top")
    padded[-1, 1:-1] = _get_adjacent_row(all_elevations, face_id, "bottom")
    padded[1:-1, 0] = _get_adjacent_row(all_elevations, face_id, "left")
    padded[1:-1, -1] = _get_adjacent_row(all_elevations, face_id, "right")

    padded[0, 0] = (padded[0, 1] + padded[1, 0]) * 0.5
    padded[0, -1] = (padded[0, -2] + padded[1, -1]) * 0.5
    padded[-1, 0] = (padded[-1, 1] + padded[-2, 0]) * 0.5
    padded[-1, -1] = (padded[-1, -2] + padded[-2, -1]) * 0.5

    return padded


def _compute_slope_factor(
    elev_grid: np.ndarray,
    face_id: int,
    all_elevations: dict[int, np.ndarray],
    diameter_m: float,
    res: int,
) -> np.ndarray:
    """Compute slope factor for noise amplitude scaling with cross-face gradient context."""
    pixel_size_m = diameter_m * math.pi / (4.0 * res)
    grad_threshold = 40.0 * pixel_size_m / 90.0

    padded = _pad_face_with_neighbors(elev_grid, face_id, all_elevations)
    gy, gx = np.gradient(padded)
    gy = gy[1:-1, 1:-1]
    gx = gx[1:-1, 1:-1]

    gradient = np.sqrt(gx**2 + gy**2)
    return np.clip(gradient / grad_threshold, 0.0, 1.0) ** 1.5


def _apply_noise_detail(
    elev_grid: np.ndarray,
    face_id: int,
    face_size: int,
    res: int,
    diameter_m: float,
    native_resolution: float,
    noise_scale: float,
    slope_factor_full: np.ndarray,
) -> np.ndarray:
    """Add slope-adaptive Perlin noise detail to a face heightmap."""
    pixel_size_m = diameter_m * math.pi / (4.0 * res)
    strip_h = 256
    noise_coord_scale = face_size / 2.0

    result = elev_grid.copy()

    t_col = np.linspace(0.0, 1.0, res, dtype=np.float32)
    t_row = np.linspace(0.0, 1.0, res, dtype=np.float32)

    for y0 in range(0, res, strip_h):
        y1 = min(y0 + strip_h, res)
        sh = y1 - y0

        uc, vr = np.meshgrid(t_col, t_row[y0:y1])
        if face_id in (0, 1, 4, 5):
            v = 1.0 - vr
        else:
            v = vr
        u = uc

        if face_id == 0:
            sx, sy, sz = np.ones_like(u), 2.0 * v - 1.0, 1.0 - 2.0 * u
        elif face_id == 1:
            sx, sy, sz = -np.ones_like(u), 2.0 * v - 1.0, 2.0 * u - 1.0
        elif face_id == 2:
            sx, sy, sz = 2.0 * u - 1.0, np.ones_like(u), 2.0 * v - 1.0
        elif face_id == 3:
            sx, sy, sz = 2.0 * u - 1.0, -np.ones_like(u), -(2.0 * v - 1.0)
        elif face_id == 4:
            sx, sy, sz = 2.0 * u - 1.0, 2.0 * v - 1.0, np.ones_like(u)
        else:
            sx, sy, sz = 1.0 - 2.0 * u, 2.0 * v - 1.0, -np.ones_like(u)

        norms = np.sqrt(sx**2 + sy**2 + sz**2)
        sx /= norms
        sy /= norms
        sz /= norms

        coords = np.array([
            (sx * noise_coord_scale).ravel(),
            (sy * noise_coord_scale).ravel(),
            (sz * noise_coord_scale).ravel(),
        ], dtype=np.float32)

        noise_coarse = _FACE_NOISE_COARSE.gen_from_coords(coords).astype(np.float32).reshape(sh, res)
        noise_fine = _FACE_NOISE_FINE.gen_from_coords(coords).astype(np.float32).reshape(sh, res)

        slope_factor = slope_factor_full[y0:y1, :]
        amp_coarse = noise_scale * 100.0 * slope_factor * pixel_size_m / native_resolution
        amp_fine = noise_scale * 70.0 * slope_factor * pixel_size_m / native_resolution

        is_land = result[y0:y1, :] >= 0.0
        result[y0:y1, :] += (noise_coarse * amp_coarse + noise_fine * amp_fine) * is_land

    return result


def apply_noise(
    input_json: str,
    output_json: str | None = None,
    noise_scale: float = 1.0,
    native_resolution: float | None = None,
) -> None:
    """Apply slope-adaptive Perlin noise to exported face heightmaps."""
    json_path = Path(input_json)
    with open(json_path) as f:
        meta = json.load(f)

    face_dir = json_path.parent

    diameter_m = meta.get("diameter_m")
    if diameter_m is None:
        radius = meta.get("radius")
        if radius is None:
            raise ValueError("JSON must contain 'diameter_m' or 'radius'.")
        diameter_m = radius * 2.0

    native_res = native_resolution or meta.get("native_resolution", 90.0)
    planet_period = meta.get("planet_period")
    face_resolution = meta["face_resolution"]

    if planet_period is not None:
        face_size = planet_period // 4
    else:
        # Derive from face resolution (assume face_resolution == face_size)
        face_size = face_resolution

    face_meta = meta["faces"]

    # Load all 6 face elevations (keyed by face index for cross-face gradient context)
    all_elevations: dict[int, np.ndarray] = {}
    for face_id, name in enumerate(_FACE_NAMES):
        all_elevations[face_id] = _load_face_elevation(face_meta[name], face_dir)

    res = all_elevations[0].shape[0]

    # Determine output paths
    if output_json is not None:
        out_json_path = Path(output_json)
    else:
        out_json_path = json_path.with_name(f"{json_path.stem}_noised.json")
    output_stem = out_json_path.with_suffix("")

    print(f"Applying noise (scale={noise_scale}) to {json_path.name}...")

    updated_face_meta = {}
    for face_id, name in enumerate(_FACE_NAMES):
        elev_grid = all_elevations[face_id]

        slope_factor = _compute_slope_factor(
            elev_grid, face_id, all_elevations, diameter_m, res,
        )
        elev_grid = _apply_noise_detail(
            elev_grid,
            face_id=face_id,
            face_size=face_size,
            res=res,
            diameter_m=diameter_m,
            native_resolution=native_res,
            noise_scale=noise_scale,
            slope_factor_full=slope_factor,
        )

        emin, emax = float(elev_grid.min()), float(elev_grid.max())

        # Write face in same format as input
        fmt = face_meta[name]["format"]
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

        updated_face_meta[name] = {
            "file": face_path.name,
            "format": fmt,
            "resolution": res,
            "elev_min": emin,
            "elev_max": emax,
        }
        print(f"  {name}: {face_path.name} [{emin:.1f}, {emax:.1f}]m")

    # Write output JSON with updated metadata
    out_meta = dict(meta)
    out_meta["faces"] = updated_face_meta
    out_meta["noise_scale"] = noise_scale
    out_json_path.write_text(json.dumps(out_meta, indent=2))
    print(f"Wrote: {out_json_path}")


@click.command()
@click.argument("input_json")
@click.option("--output", default=None, help="Output JSON path (default: <stem>_noised.json)")
@click.option("--noise-scale", type=float, default=1.0, help="Global noise amplitude multiplier (default 1.0)")
@click.option("--native-resolution", type=float, default=None, help="Override native resolution in meters (default: from JSON or 90.0)")
def main(input_json, output, noise_scale, native_resolution):
    """Apply slope-adaptive Perlin noise detail to exported face heightmaps."""
    apply_noise(
        input_json=input_json,
        output_json=output,
        noise_scale=noise_scale,
        native_resolution=native_resolution,
    )


if __name__ == "__main__":
    main()
