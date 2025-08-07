# Terrain Diffusion (WIP - HIGHLY EXPERIMENTAL)

Terrain Diffusion is an AI-powered terrain generation framework designed to replace traditional procedural noise functions (like Perlin noise) with a fast, high-fidelity, and infinitely tileable generative model. Built on cutting-edge diffusion techniques, it can generate elevation maps that span land and ocean, produce consistent terrain on an infinite grid, and support climate and water generation.

## 🚀 Features

- **Hyper-realistic terrain generation**: Trained on real-world elevation data
- **Fast inference**: Uses a 1- or 2-step continuous-time consistency model (CTCM)
- **Infinitely tileable**: Seamless stitching on an infinite 2D grid
- **Physically accurate scale**: Outputs elevation in meters (not normalized)
- **Modular generation**: Supports generation of water and climate maps
- **Infinite-space support**: Backed by [xandergos/infinite-tensors](https://github.com/xandergos/infinite-tensors)

## 🧠 How It Works

### 1. **Autoencoder Training**
- Trained on elevation data using a custom Laplacian-based encoder to separate low and high frequencies
- Solves the issue of small model errors being massively magnified due to wide ranges in elevation (from -10,000m to +10,000m)

### 2. **Diffusion Training**
- Trained a diffusion (EDM2) decoder on latent terrain representations
- Trained a diffusion (EDM2) model to generate latent terrain representations
- Low and high frequencies generated separately for improved accuracy and robustness

### 3. **Acceleration via sCM**
- Diffusion models are distilled following the process in "Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models".
- Distillation code modified from [xandergos/sCM-mnist](https://github.com/xandergos/sCM-mnist)
- Achieves near real-time generation speed with just 1–2 denoising steps, FID increases by just ~10%.

### 4. **Tileability & Infinite Generation**
- Condition diffusion on low-frequency features (e.g., mean elevation) for tile alignment
- Generate low-freq map via GAN (translation invariant, no padding)
- Final pipeline: `GAN → Base Consistency Model → Consistency Decoder`
- Uses [xandergos/infinite-tensor](https://github.com/xandergos/infinite-tensor) for efficient on-demand generation

## 🧪 Demo

<img width="1920" height="920" alt="image" src="https://github.com/user-attachments/assets/f3c581a8-c9b8-4965-8158-2bf63b6155d5" />

## 📜 License

MIT License
