#!/bin/bash
python3.11 ./terrain_diffusion/data/downloading/data.py \
    --image landcover_class \
    --output_dir data/landcover_class \
    --output_size 4096 \
    --output_resolution 90 \
    --land_threshold 0.1