"""
Convert an Azgaar Fantasy Map Builder full JSON export to GeoTIFF rasters.

Outputs:
  heightmap.tif        - elevation in meters (float32; uses Azgaar's (h-18)^exponent formula)
  temperature.tif      - mean temperature in °C (float32, from grid cells)
  temperature_std.tif  - temperature std deviation in °C (float32, derived from biome)
  precipitation.tif    - annual precipitation in mm (float32, grid prec * 100)
  precipitation_cv.tif - precipitation coefficient of variation % (float32, derived from biome)

Usage:
  python -m terrain_diffusion.inference.utils.azgaar_to_tiff \
      "Vigny Full.json" output_dir/ --scale 7
"""

import json
import warnings
from pathlib import Path

import click

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt

# Biome ID -> (temp_std_C, precip_cv_pct)
# std and CV are biome-characteristic; actual mean values come from grid cell data.
BIOME_VARIABILITY = {
    0:  (float("nan"), float("nan")),  # Marine
    1:  (5.0,  80.0),   # Hot Desert
    2:  (15.0, 33.0),   # Cold Desert
    3:  (5.0,  28.6),   # Savanna
    4:  (10.0, 25.0),   # Grassland
    5:  (3.0,  26.7),   # Tropical Seasonal Forest
    6:  (8.0,  22.2),   # Temperate Deciduous Forest
    7:  (2.0,  16.0),   # Tropical Rainforest
    8:  (6.0,  25.0),   # Temperate Rainforest
    9:  (15.0, 20.0),   # Taiga
    10: (15.0, 25.0),   # Tundra
    11: (10.0, 30.0),   # Glacier
    12: (8.0,  20.0),   # Wetland
}
TEMP_STD_IDX, PRECIP_CV_IDX = 0, 1



def load_map(path):
    with open(path) as f:
        data = json.load(f)

    info = data["info"]
    coords = data["mapCoordinates"]
    pack = data["pack"]
    grid = data["grid"]

    map_w = info["width"]
    map_h = info["height"]

    pack_verts = {v["i"]: v["p"] for v in pack["vertices"]}
    grid_verts = {v["i"]: v["p"] for v in grid["vertices"]}
    height_exponent = float(data["settings"]["heightExponent"])

    return map_w, map_h, coords, pack["cells"], pack_verts, grid["cells"], grid_verts, height_exponent


def h_to_meters(h, exponent, ocean_max_depth=4000.0, ocean_power=1.5):
    """Convert Azgaar internal height (0-100) to meters.

    Land (h >= 20) matches Azgaar's getHeight(): (h-18)^exponent
    Ocean (h < 20) uses a power curve: -ocean_max_depth * ((20-h)/20)^ocean_power
      h=0  -> -ocean_max_depth (deepest ocean)
      h=19 -> ~-45 m at defaults (coastal shelf)
    """
    if h < 20:
        return -ocean_max_depth * ((20 - h) / 20) ** ocean_power
    return float(h - 18) ** exponent


def build_shapes(cells, verts, scale_x, scale_y, value_fn):
    """Yield (geometry, value) for each cell, using the given vertex lookup."""
    for cell in cells:
        value = value_fn(cell)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            ring = [[px * scale_x, py * scale_y] for px, py in (verts[vi] for vi in cell["v"])]
        except KeyError:
            continue
        yield {"type": "Polygon", "coordinates": [ring]}, value


def rasterize_layer(cells, verts, scale_x, scale_y, shape, value_fn, dtype, fill):
    shapes = list(build_shapes(cells, verts, scale_x, scale_y, value_fn))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Invalid or empty shape")
        arr = rasterize(
            shapes,
            out_shape=shape,
            fill=fill,
            dtype=dtype,
            all_touched=False,
        )
    return arr


def fill_nodata(arr, nodata):
    """Replace nodata pixels with the value of the nearest valid pixel."""
    if isinstance(nodata, float) and np.isnan(nodata):
        mask = np.isnan(arr)
    else:
        mask = arr == nodata
    if not mask.any():
        return arr
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(indices)]


def write_tiff(path, arr, transform, crs="EPSG:4326", nodata=None):
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(arr, 1)


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--scale", default=100.0, show_default=True, help="Size of each output pixel in km")
@click.option("--ocean-max-depth", default=4000.0, show_default=True, help="Maximum ocean depth in meters (at h=0)")
@click.option("--ocean-power", default=1.5, show_default=True, help="Power curve exponent for ocean depth (higher = steeper near coast)")
def main(input, output_dir, scale, ocean_max_depth, ocean_power):
    """Convert an Azgaar full JSON export to GeoTIFF rasters."""
    input_path = Path(input)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    map_w, map_h, coords, pack_cells, pack_verts, grid_cells, grid_verts, height_exponent = load_map(input_path)
    print(f"  Map size: {map_w}x{map_h}, {len(grid_cells)} grid cells, {len(pack_cells)} pack cells, exponent={height_exponent}")

    lon_w, lon_e = coords["lonW"], coords["lonE"]
    lat_s, lat_n = coords["latS"], coords["latN"]

    mid_lat = np.radians((lat_n + lat_s) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mid_lat)
    out_w = max(1, round((lon_e - lon_w) * km_per_deg_lon / scale))
    out_h = max(1, round((lat_n - lat_s) * km_per_deg_lat / scale))

    scale_x = out_w / map_w
    scale_y = out_h / map_h
    print(f"  Output shape: {out_w}x{out_h} (WxH), {scale} km/px")

    transform = from_bounds(lon_w, lat_s, lon_e, lat_n, out_w, out_h)
    shape = (out_h, out_w)

    # Grid cells carry h/temp/prec and cover the full map including deep ocean.
    # Pack cells carry biome and are used only for biome-derived layers.
    grid_kw = dict(cells=grid_cells, verts=grid_verts, scale_x=scale_x, scale_y=scale_y, shape=shape)
    pack_kw = dict(cells=pack_cells, verts=pack_verts, scale_x=scale_x, scale_y=scale_y, shape=shape)

    # --- Heightmap ---
    print("Rasterizing heightmap...")
    arr = rasterize_layer(**grid_kw, dtype="float32", fill=np.nan,
                          value_fn=lambda c: h_to_meters(c.get("h", 0), height_exponent, ocean_max_depth, ocean_power))
    arr = fill_nodata(arr, np.nan)
    write_tiff(output_dir / "heightmap.tif", arr, transform)
    print(f"  height range: {arr.min():.0f} .. {arr.max():.0f} m")

    # --- Temperature ---
    print("Rasterizing temperature...")
    arr = rasterize_layer(**grid_kw, dtype="float32", fill=-9999.0,
                          value_fn=lambda c: float(c["temp"]) if "temp" in c else None)
    arr = fill_nodata(arr, -9999.0)
    write_tiff(output_dir / "temperature.tif", arr, transform)
    print(f"  temperature range: {arr.min():.1f} .. {arr.max():.1f} °C")

    # --- Temperature std (from biome, pack cells) ---
    print("Rasterizing temperature std...")
    arr = rasterize_layer(**pack_kw, dtype="float32", fill=-9999.0,
                          value_fn=lambda c: BIOME_VARIABILITY.get(c.get("biome", 0), (float("nan"), float("nan")))[TEMP_STD_IDX])
    arr = fill_nodata(arr, -9999.0)
    write_tiff(output_dir / "temperature_std.tif", arr, transform)
    print(f"  temperature std range: {arr.min():.1f} .. {arr.max():.1f} °C")

    # --- Precipitation ---
    print("Rasterizing precipitation...")
    arr = rasterize_layer(**grid_kw, dtype="float32", fill=-9999.0,
                          value_fn=lambda c: float(c["prec"]) * 100.0 if "prec" in c else None)
    arr = fill_nodata(arr, -9999.0)
    write_tiff(output_dir / "precipitation.tif", arr, transform)
    print(f"  precipitation range: {arr.min():.0f} .. {arr.max():.0f} mm/yr")

    # --- Precipitation CV (from biome, pack cells) ---
    print("Rasterizing precipitation CV...")
    arr = rasterize_layer(**pack_kw, dtype="float32", fill=-9999.0,
                          value_fn=lambda c: BIOME_VARIABILITY.get(c.get("biome", 0), (float("nan"), float("nan")))[PRECIP_CV_IDX])
    arr = fill_nodata(arr, -9999.0)
    write_tiff(output_dir / "precipitation_cv.tif", arr, transform)
    print(f"  precipitation CV range: {arr.min():.1f} .. {arr.max():.1f} %")

    print(f"\nWrote TIFFs to {output_dir}/")


if __name__ == "__main__":
    main()

