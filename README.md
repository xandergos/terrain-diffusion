# Terrain Diffusion

This is a repository that aims to utilize diffusion models to generate realistic terrain data. It creates high-quality, diverse terrain samples for various applications such as video game development, environmental simulations, and geographical studies.

Key features of Terrain Diffusion include:

1. Diffusion-based terrain generation: Using diffusion models to create detailed and natural-looking terrain. For this task, I used the EDM2 architecture, with some modifications to support (almost) zero-SNR.

2. Multi-scale approach: Generating terrain at different resolutions (64x64, 256x256, 1024x1024 and beyond). This is performed with diffusion super-resolution. To ensure that terrain remains realistic, and to avoid artifacts, the terrain is encoded into a latent space with a laplacian pyramid, which ensures that generated terrain contains both large scale features like mountain ranges, and fine details like small hills, rivers, and accurate erosion patterns.

3. Tileable terrain: Optionally, the terrain can be generated in a tileable fashion. This allows terrain to be generated in real-time and infinitely, useful for real-time applications such as video games. This is done with a tile-stiching method like Multi-Diffusion.

5. Conditional generation: Ability to generate terrain based on specific conditions or labels, allowing for more controlled output.

5. Flexible configuration: Nearly every part of the training process can be configured, including model architecture, training data, and training parameters.

6. Integration with wandb: Tracking experiments and visualizing results using Weights & Biases for better model monitoring and analysis.

