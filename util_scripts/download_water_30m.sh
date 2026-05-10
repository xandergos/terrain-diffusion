#!/bin/bash
# Download JRC Global Surface Water occurrence tiles at 30m native resolution.
# Tile grid matches the Copernicus DEM grid (30m variant).
# JRC occurrence values: 0-100 (percentage 1984-2021), 255 = no-data (outside Landsat coverage).
python3.11 -m terrain_diffusion.data.downloading.data \
    --image jrc \
    --output_dir data/jrc_data \
    --output_size 2048 \
    --output_resolution 30.71 \
    --land_threshold 0.1 \
    --num_workers 10
