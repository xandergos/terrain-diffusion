import json
from pathlib import Path

import click
import numpy as np


def _cube_face_directions(face_id: int, res: int) -> np.ndarray:
    """Return (res*res, 3) unit sphere directions for a cube face grid.

    Matches the _cross_face_atlas mapping so mesh vertices align exactly
    with the face heightmap pixels.
    """
    t = np.linspace(0.0, 1.0, res, dtype=np.float32)
    uc, vr = np.meshgrid(t, t)  # uc = u (column), vr = v (row)

    # Faces in row 1 have inverted v (atlas i increases downward, v increases upward)
    if face_id in (0, 1, 4, 5):
        v = 1.0 - vr
    else:
        v = vr
    u = uc

    if face_id == 0:  # +X
        x = np.ones_like(u)
        z = 1.0 - 2.0 * u
        y = 2.0 * v - 1.0
    elif face_id == 1:  # -X
        x = -np.ones_like(u)
        z = 2.0 * u - 1.0
        y = 2.0 * v - 1.0
    elif face_id == 2:  # +Y
        y = np.ones_like(u)
        x = 2.0 * u - 1.0
        z = 2.0 * v - 1.0
    elif face_id == 3:  # -Y
        y = -np.ones_like(u)
        x = 2.0 * u - 1.0
        z = -(2.0 * v - 1.0)
    elif face_id == 4:  # +Z
        z = np.ones_like(u)
        x = 2.0 * u - 1.0
        y = 2.0 * v - 1.0
    else:  # -Z
        z = -np.ones_like(u)
        x = 1.0 - 2.0 * u
        y = 2.0 * v - 1.0

    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= norms
    return dirs


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("# Terrain Diffusion cube-sphere mesh\n")
        for vx, vy, vz in verts:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        if normals is not None:
            for nx, ny, nz in normals:
                f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("s 1\n")
        if normals is not None:
            for i0, i1, i2 in faces:
                f.write(f"f {i0+1}//{i0+1} {i1+1}//{i1+1} {i2+1}//{i2+1}\n")
        else:
            for i0, i1, i2 in faces:
                f.write(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n")


def _load_face_elevation(face_info: dict, face_dir: Path, mesh_res: int | None = None) -> np.ndarray:
    """Load a face heightmap and return elevation in meters as float32.

    For PNG: reconstructs from uint16 normalization using elev_min/elev_max.
    For TIFF: reads raw float32 values directly.
    Resamples to mesh_res if provided and different from image resolution.
    """
    face_path = face_dir / face_info["file"]
    fmt = face_info["format"]
    img_res = face_info["resolution"]

    if fmt == "tiff":
        import tifffile
        elev = tifffile.imread(str(face_path)).astype(np.float32)
    else:
        from PIL import Image
        img = Image.open(str(face_path))
        pixels = np.array(img, dtype=np.float32)
        emin = face_info["elev_min"]
        emax = face_info["elev_max"]
        if emax > emin:
            elev = pixels / 65535.0 * (emax - emin) + emin
        else:
            elev = np.full_like(pixels, emin)

    if mesh_res is not None and mesh_res != img_res:
        from scipy.ndimage import map_coordinates
        y_new = np.linspace(0, img_res - 1, mesh_res)
        x_new = np.linspace(0, img_res - 1, mesh_res)
        yy, xx = np.meshgrid(y_new, x_new, indexing="ij")
        elev = map_coordinates(elev, [yy, xx], order=1, mode="nearest").astype(np.float32)

    return elev


# Face ordering matches sphere_export.py face_defs
_FACE_NAMES = ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"]


def _merge_border_vertices(verts: np.ndarray, tris: np.ndarray, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices at cube face borders that map to the same sphere position.

    Adjacent faces produce separate vertices at shared edges. This merges them
    so the mesh is watertight and normals interpolate smoothly across edges.
    """
    n_face = res * res
    remap = np.arange(len(verts), dtype=np.int64)
    dir_to_idx: dict[tuple[float, float, float], int] = {}

    for face_id in range(6):
        base = face_id * n_face
        # Collect border vertex indices (top/bottom rows, left/right columns)
        borders = set()
        for i in range(res):
            borders.add(base + i)                    # top row
            borders.add(base + (res - 1) * res + i)  # bottom row
            borders.add(base + i * res)               # left col
            borders.add(base + i * res + (res - 1))   # right col

        for idx in borders:
            pos = verts[idx]
            norm = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            unit = pos / norm
            key = (round(float(unit[0]), 5), round(float(unit[1]), 5), round(float(unit[2]), 5))
            if key in dir_to_idx:
                canonical = dir_to_idx[key]
                remap[idx] = canonical
                # Average positions to smooth any tiny floating-point difference
                verts[canonical] = (verts[canonical] + verts[idx]) * 0.5
            else:
                dir_to_idx[key] = idx

    tris = remap[tris]
    return verts, tris


_EDGES = [
    ("pos_x", "top",    "pos_y", "right",  True),
    ("pos_x", "bottom", "neg_y", "right",  False),
    ("pos_x", "left",   "pos_z", "right",  False),
    ("pos_x", "right",  "neg_z", "left",   False),
    ("neg_x", "top",    "pos_y", "left",   False),
    ("neg_x", "bottom", "neg_y", "left",   True),
    ("neg_x", "left",   "neg_z", "right",  False),
    ("neg_x", "right",  "pos_z", "left",   False),
    ("pos_y", "top",    "neg_z", "top",    True),
    ("pos_y", "bottom", "pos_z", "top",    False),
    ("neg_y", "top",    "pos_z", "bottom", False),
    ("neg_y", "bottom", "neg_z", "bottom", True),
]


def _fix_border_seams(face_elevations: dict[str, np.ndarray], feather: int = 4) -> None:
    """Fix elevation mismatches at shared cube edges.

    For each shared edge, computes the mismatch (difference between the two
    faces' border values), then distributes the correction over `feather`
    pixels using a linear ramp. This avoids the V-shaped bump that single-
    pixel averaging creates.
    """
    def _get_border(face: np.ndarray, side: str) -> np.ndarray:
        if side == "top":    return face[0, :]
        if side == "bottom": return face[-1, :]
        if side == "left":   return face[:, 0]
        return face[:, -1]

    def _apply_correction(face: np.ndarray, side: str, correction: np.ndarray, feather: int) -> None:
        """Apply a correction that's full at the border and fades to 0 over feather pixels."""
        res = face.shape[0]
        w = min(feather, res // 4)
        for i in range(w):
            alpha = 1.0 - i / w  # 1.0 at border, 0.0 at feather distance
            if side == "top":      face[i, :] += correction * alpha
            elif side == "bottom": face[-(i+1), :] += correction * alpha
            elif side == "left":   face[:, i] += correction * alpha
            else:                  face[:, -(i+1)] += correction * alpha

    for fa, sa, fb, sb, rev in _EDGES:
        a = _get_border(face_elevations[fa], sa).copy()
        b = _get_border(face_elevations[fb], sb).copy()
        if rev:
            b = b[::-1]
        # Split the mismatch: each face corrects half toward the other
        mismatch = a - b  # positive means a is higher
        corr_a = -mismatch * 0.5  # lower face a
        corr_b = mismatch * 0.5   # raise face b
        _apply_correction(face_elevations[fa], sa, corr_a, feather)
        _apply_correction(face_elevations[fb], sb, corr_b[::-1] if rev else corr_b, feather)


def _build_cubesphere_mesh(
    face_elevations: dict[str, np.ndarray],
    radius: float,
    elevation_scale: float,
    max_relief_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a cube-sphere mesh from 6 face elevation grids.

    Returns (verts, tris) arrays ready for OBJ export.
    """
    res = next(iter(face_elevations.values())).shape[0]

    # Compute global median across all faces
    all_elev = np.concatenate([e.ravel() for e in face_elevations.values()])
    median_elev = float(np.median(all_elev))

    all_verts = []
    all_tris = []

    for face_id, name in enumerate(_FACE_NAMES):
        elev_grid = face_elevations[name]
        dirs = _cube_face_directions(face_id, res)
        elev = elev_grid.ravel()

        # Elevation processing: center, scale, clamp
        relief = (elev - median_elev) * elevation_scale
        relief = np.clip(relief, -max_relief_m, max_relief_m)
        displaced_r = np.maximum(radius + relief, radius * 0.5)

        verts = dirs * displaced_r[:, None]
        all_verts.append(verts)

        # Triangle grid: (res-1)^2 quads, 2 tris each
        offset = face_id * res * res
        rows = np.arange(res - 1)
        cols = np.arange(res - 1)
        cc, rr = np.meshgrid(cols, rows)
        v00 = offset + rr * res + cc
        v01 = offset + rr * res + cc + 1
        v10 = offset + (rr + 1) * res + cc
        v11 = offset + (rr + 1) * res + cc + 1
        tri1 = np.stack([v00, v10, v01], axis=-1).reshape(-1, 3)
        tri2 = np.stack([v01, v10, v11], axis=-1).reshape(-1, 3)
        all_tris.append(np.concatenate([tri1, tri2], axis=0))

    verts = np.concatenate(all_verts, axis=0)
    tris = np.concatenate(all_tris, axis=0)

    # Merge border vertices so adjacent faces share vertices and normals
    verts, tris = _merge_border_vertices(verts, tris, res)

    return verts, tris, median_elev


def faces_to_obj(
    input_json: str,
    output_obj: str | None = None,
    mesh_resolution: int | None = None,
    elevation_scale: float = 1.0,
    max_relief_m: float | None = None,
) -> None:
    """Build a cube-sphere OBJ mesh from exported face heightmaps."""
    json_path = Path(input_json)
    with open(json_path) as f:
        meta = json.load(f)

    face_dir = json_path.parent
    if "diameter_m" in meta:
        radius = meta["diameter_m"] * 0.5
    elif "radius" in meta:
        radius = meta["radius"]
    else:
        raise ValueError("JSON must contain 'diameter_m' or 'radius'.")
    seed = meta.get("seed")

    if "faces" in meta:
        face_meta = meta["faces"]
    else:
        # Infer face files from naming convention: <stem>_face_<name>.<ext>
        stem = json_path.stem
        face_meta = {}
        for name in _FACE_NAMES:
            for ext in ("tiff", "png"):
                candidate = face_dir / f"{stem}_face_{name}.{ext}"
                if candidate.exists():
                    import tifffile
                    if ext == "tiff":
                        data = tifffile.imread(str(candidate)).astype(np.float32)
                    else:
                        from PIL import Image
                        data = np.array(Image.open(str(candidate)), dtype=np.float32)
                    face_meta[name] = {
                        "file": candidate.name,
                        "format": ext,
                        "resolution": data.shape[0],
                        "elev_min": float(data.min()),
                        "elev_max": float(data.max()),
                    }
                    del data
                    break
            else:
                raise FileNotFoundError(f"No face image found for {name} (tried {stem}_face_{name}.tiff/.png)")

    if max_relief_m is None:
        max_relief_m = max(1.0, radius * 0.1)

    # Determine mesh resolution
    first_face = next(iter(face_meta.values()))
    img_res = first_face["resolution"]
    res = mesh_resolution or img_res

    print(f"Loading faces from {json_path.name} ({img_res}x{img_res} images)")
    if res != img_res:
        print(f"  Resampling to {res}x{res} mesh resolution")

    # Load all face elevations
    face_elevations = {}
    for name in _FACE_NAMES:
        info = face_meta[name]
        elev = _load_face_elevation(info, face_dir, mesh_res=res if res != img_res else None)
        face_elevations[name] = elev
        emin, emax = float(elev.min()), float(elev.max())
        print(f"  {name}: [{emin:.1f}, {emax:.1f}]m")

    _fix_border_seams(face_elevations)

    print(f"Building cube-sphere mesh ({res}x{res} per face)...")
    verts, tris, median_elev = _build_cubesphere_mesh(
        face_elevations=face_elevations,
        radius=radius,
        elevation_scale=elevation_scale,
        max_relief_m=max_relief_m,
    )

    # Output path
    if output_obj is None:
        out_path = json_path.with_suffix(".obj")
    else:
        out_path = Path(output_obj)

    _write_obj(out_path, verts, tris)

    # Write OBJ sidecar JSON (compatible with view_planet.py)
    obj_meta = {
        "radius": radius,
        "median_elev": median_elev,
        "elevation_scale": elevation_scale,
        "max_relief_m": max_relief_m,
        "sampler": "cubesphere",
        "seed": seed,
    }
    obj_meta_path = out_path.with_suffix(".json")
    obj_meta_path.write_text(json.dumps(obj_meta, indent=2))

    print(f"Wrote: {out_path}")
    print(f"Vertices: {verts.shape[0]}")
    print(f"Triangles: {tris.shape[0]}")
    print(
        f"radius={radius}, elevation_scale={elevation_scale}, "
        f"max_relief_m={max_relief_m}, median_elev={median_elev:.2f}"
    )


@click.command()
@click.argument("input_json")
@click.option("--output", default=None, help="Output OBJ path (default: same stem as JSON)")
@click.option("--mesh-resolution", type=int, default=None, help="Mesh resolution per face (default: face image resolution)")
@click.option("--elevation-scale", type=float, default=1.0, help="Scale factor for elevation displacement")
@click.option("--max-relief-m", type=float, default=None, help="Clamp absolute relief in meters (default: 10% of radius)")
def main(input_json, output, mesh_resolution, elevation_scale, max_relief_m):
    """Build a cube-sphere OBJ mesh from exported face heightmaps."""
    faces_to_obj(
        input_json=input_json,
        output_obj=output,
        mesh_resolution=mesh_resolution,
        elevation_scale=elevation_scale,
        max_relief_m=max_relief_m,
    )


if __name__ == "__main__":
    main()
