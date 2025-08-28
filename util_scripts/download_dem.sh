#!/bin/bash
python3.11 ./terrain_diffusion/data/downloading/data.py \
    --image dem \
    --output_dir data/dem_data \
    --output_size 4096 \
    --output_resolution 90 \
    --land_threshold 0.1