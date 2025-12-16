#!/bin/bash
python3.11 -m terrain_diffusion.data.downloading.data \
    --image copernicus \
    --output_dir data/copernicus_data \
    --output_size 2048 \
    --output_resolution 30.71 \
    --land_threshold 0.1 \
    --num_workers 10
