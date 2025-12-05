#!/bin/bash
set -e

# Evaluation script for Terrain Diffusion models
# All models are assumed to be in checkpoints/models/

# 1. Evaluate Diffusion Decoder
echo "Evaluating Diffusion Decoder..."
accelerate launch -m terrain_diffusion.evaluation.decoder_diffusion

# 2. Evaluate Consistency Decoder
echo "Evaluating Consistency Decoder..."
accelerate launch -m terrain_diffusion.evaluation.decoder_consistency

## 3. Evaluate Diffusion Base
echo "Evaluating Diffusion Base..."
accelerate launch -m terrain_diffusion.evaluation.base_diffusion

# 4. Evaluate Infinite Consistency Base
echo "Evaluating Infinite Consistency Base..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency

# 4. Evaluate Consistency Base
echo "Evaluating Consistency Base..."
accelerate launch -m terrain_diffusion.evaluation.base_consistency

echo "Calculating Real FID..."
python3.11 ./terrain_diffusion/training/sweeps/calc_real_kid.py --config ./configs/diffusion_base/diffusion_192-3.cfg --metric fid --max-images 50000