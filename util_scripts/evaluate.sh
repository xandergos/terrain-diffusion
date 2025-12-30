#!/bin/bash
set -e

# Evaluation script for Terrain Diffusion models (one-step variants)
# All models are assumed to be in checkpoints/models/

# 1. Evaluate Consistency Decoder FID
echo "Evaluating Consistency Decoder FID..."
accelerate launch -m terrain_diffusion.evaluation.decoder_consistency --intermediate-sigma -1

# 2. Evaluate Base Diffusion
echo "Evaluating Base Diffusion..."
accelerate launch -m terrain_diffusion.evaluation.base_diffusion --inter-t -1

# 3. Evaluate Base Consistency
echo "Evaluating Base Consistency..."
accelerate launch -m terrain_diffusion.evaluation.base_consistency --decoder-inter-t -1

# 4. Evaluate Infinite Consistency
echo "Evaluating Infinite Consistency..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency --decoder-inter-t -1

# 5. Evaluate Consistency Diffusion without laplacian denoising
echo "Evaluating Base Consistency without laplacian denoising..."
accelerate launch -m terrain_diffusion.evaluation.base_consistency --decoder-inter-t -1 --disable-laplacian-denoising

# 6. Evaluate Infinite Consistency with constant weight window
echo "Evaluating Infinite Consistency with constant weight window..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency --decoder-inter-t -1 --weight-window-fn constant

# 7. Evaluate Base Diffusion without laplacian denoising
echo "Evaluating Base Diffusion without laplacian denoising..."
accelerate launch -m terrain_diffusion.evaluation.base_diffusion --inter-t -1 --disable-laplacian-denoising
