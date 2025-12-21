#!/bin/bash
set -e

# Evaluation script for Terrain Diffusion models
# All models are assumed to be in checkpoints/models/

# 1. Evaluate Consistency Decoder FID
echo "Evaluating Consistency Decoder FID..."
accelerate launch -m terrain_diffusion.evaluation.decoder_consistency

# 2. Evaluate Base Diffusion
echo "Evaluating Base Diffusion..."
accelerate launch -m terrain_diffusion.evaluation.base_diffusion

# 3. Evaluate Base Consistency
echo "Evaluating Base Consistency..."
accelerate launch -m terrain_diffusion.evaluation.base_consistency

# 4. Evaluate Infinite Consistency
echo "Evaluating Infinite Consistency..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency

# 5. Evaluate Consistency Diffusion without laplacian denoising
echo "Evaluating Base Consistency without laplacian denoising..."
accelerate launch -m terrain_diffusion.evaluation.base_consistency --disable-laplacian-denoising

# 6. Evaluate Infinite Consistency with constant weight window
echo "Evaluating Infinite Consistency with constant weight window..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency --weight-window-fn constant

echo "Calculating Real FID..."
python3.11 ./terrain_diffusion/training/sweeps/calc_real_kid.py --config ./configs/diffusion_base/diffusion_192-3.cfg --metric fid --max-images 50000
