#!/bin/bash

# Download landcover water data directly to local filesystem
python3.11 ./terrain_diffusion/data/downloading/data.py \
    --image landcover_water \
    --output_dir data/landcover_water \
    --output_size 4096 \
    --output_resolution 90 \
    --land_threshold 0.1