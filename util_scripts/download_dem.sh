#!/bin/bash
python3.11 -m terrain_diffusion.data.downloading.data \
    --image dem \
    --output_dir data/dem_data \
    --output_size 2048 \
    --output_resolution 92.15 \
    --land_threshold 0.1