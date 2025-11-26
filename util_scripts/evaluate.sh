#!/bin/bash
set -e

# Evaluation script for Terrain Diffusion models
# All models are assumed to be in checkpoints/models/

# 1. Evaluate Diffusion Decoder
#echo "Evaluating Diffusion Decoder..."
#accelerate launch -m terrain_diffusion.evaluation.decoder_diffusion \
#    --model checkpoints/models/diffusion_decoder-64x3 \
#    --config configs/diffusion_decoder/diffusion_decoder_64-3.cfg \
#    --num-images 50000 \
#    --batch-size 16 \
#    --metric fid \
#    --guide-model checkpoints/models/diffusion_decoder_32x3 \
#    --guidance-scale 1.26

# 2. Evaluate Consistency Decoder
#echo "Evaluating Consistency Decoder..."
#accelerate launch -m terrain_diffusion.evaluation.decoder_consistency \
#    --model checkpoints/models/consistency_decoder-64x3 \
#    --config configs/diffusion_decoder/consistency_decoder_64x3.cfg \
#    --num-images 50000 \
#    --batch-size 16 \
#    --tile-size 512 \
#    --intermediate-sigma 0.065 \
#    --metric fid
#
## 3. Evaluate Diffusion Base
#echo "Evaluating Diffusion Base..."
#accelerate launch -m terrain_diffusion.evaluation.base_diffusion \
#    --model checkpoints/models/diffusion_base-192x3 \
#    --config configs/diffusion_base/diffusion_192-3.cfg \
#    --num-images 50000 \
#    --batch-size 16 \
#    --metric fid \
#    --guide-model checkpoints/models/diffusion_base_32x3 \
#    --guidance-scale 2.15

# 4. Evaluate Infinite Consistency Base
echo "Evaluating Infinite Consistency Base..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency

# 4. Evaluate Consistency Base
echo "Evaluating Consistency Base..."
accelerate launch -m terrain_diffusion.evaluation.base_consistency

echo "Calculating Real FID..."
python3.11 ./terrain_diffusion/training/sweeps/calc_real_kid.py --config ./configs/diffusion_base/diffusion_192-3.cfg --metric fid --max-images 50000