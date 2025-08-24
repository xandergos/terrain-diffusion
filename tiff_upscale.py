import rasterio
from rasterio.enums import Resampling
import numpy as np

# Parameters
input_path = 'heightmap.tif'
output_path = 'heightmap_upscaled.tif'
scale_factor = 9  # Change this to desired multiplier

# Open the input TIFF
with rasterio.open(input_path) as src:
    data = src.read(
        out_shape=(
            src.count,
            int(src.height * scale_factor),
            int(src.width * scale_factor)
        ),
        resampling=Resampling.cubic  # Options: nearest, bilinear, cubic, etc.
    )

    # Scale transform affine matrix
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )

    # Save the upscaled image
    profile = src.profile
    profile.update({
        'height': data.shape[1],
        'width': data.shape[2],
        'transform': transform
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)

print(f"Upscaled TIFF saved to {output_path}")
