# Terrain Diffusion

Terrain Diffusion is an AI-powered terrain generation framework designed to replace traditional procedural noise functions (like Perlin noise) with a fast, high-fidelity, and infinitely tileable generative model. Built on cutting-edge diffusion techniques, it can generate elevation maps that span land and ocean, produce consistent terrain on an infinite grid, and support climate and water generation.

## Features

- **Hyper-realistic terrain generation**: Trained on real-world elevation data
- **Fast inference**: Uses a 1- or 2-step continuous-time consistency model
- **Infinitely tileable**: Seamless stitching on an infinite 2D grid backed by [xandergos/infinite-tensor](https://github.com/xandergos/infinite-tensor)
- **Physically accurate scale**: Outputs elevation in meters

## How It Works

### 1. **Model Training**
- An autoencoder learns a latent representation for terrain features
- A diffusion (EDM2) decoder learns to convert latent vectors to terrain with improved fidelity
- Another diffusion model learns to generate latent terrain representations directly
- Coarse and fine details generated separately for improved accuracy and robustness
- A tileable GAN generates the most large-scale features, ensuring coherence on continental scales

### 2. **Acceleration via sCM**
- Diffusion models are distilled following the process in "Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models".
- Distillation code modified from [xandergos/sCM-mnist](https://github.com/xandergos/sCM-mnist)
- Achieves near real-time generation speed with just 1–2 denoising steps

### 3. **Tileability & Infinite Generation**
- Final map is generated in tiles, and stitched together seamlessly
- Final pipeline: `GAN → Base Consistency Model → Consistency Decoder`
- Uses [xandergos/infinite-tensor](https://github.com/xandergos/infinite-tensor) for efficient on-demand generation

## Example

<img width="1920" height="920" alt="image" src="https://github.com/user-attachments/assets/f3c581a8-c9b8-4965-8158-2bf63b6155d5" />

## License

MIT License
