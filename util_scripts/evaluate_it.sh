#!/bin/bash
set -e

# Evaluation script for Terrain Diffusion models
# All models are assumed to be in checkpoints/models/

# 1. Evaluate Infinite Consistency Naive
echo "Evaluating Infinite Consistency Naive..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency_naive --inter-t -1

# 2. Evaluate Infinite Consistency Smarter
echo "Evaluating Infinite Consistency Smarter..."
accelerate launch -m terrain_diffusion.evaluation.infinite_consistency_smarter --inter-t -1