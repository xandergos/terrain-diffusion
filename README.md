# Terrain Diffusion

This is a repository that aims to utilize diffusion models to generate realistic terrain data. It creates high-quality, diverse terrain samples for various applications such as video game development, environmental simulations, and geographical studies.

Key features of Terrain Diffusion include:

1. Diffusion-based terrain generation: Using diffusion models to create detailed and natural-looking terrain. For this task, I used the EDM2 architecture, with some modifications to support (almost) zero-SNR.

2. Multi-scale approach: Generating terrain at different resolutions (64x64, 256x256, 1024x1024 and beyond). This is performed with diffusion super-resolution. To ensure that terrain remains realistic, and to avoid artifacts, the terrain is encoded into a latent space with a laplacian pyramid, which ensures that generated terrain contains both large scale features like mountain ranges, and fine details like small hills, rivers, and accurate erosion patterns.

3. Tileable terrain: Optionally, the terrain can be generated in a tileable fashion. This allows terrain to be generated in real-time and infinitely, useful for real-time applications such as video games. This is done with a tile-stiching method like Multi-Diffusion.

5. Conditional generation: Ability to generate terrain based on specific conditions or labels, allowing for more controlled output.

5. Flexible configuration: Nearly every part of the training process can be configured, including model architecture, training data, and training parameters.

6. Integration with wandb: Tracking experiments and visualizing results using Weights & Biases for better model monitoring and analysis.

# Tiled Generation
This project implements an infinitely tileable, high-order diffusion sampler for generating infinite terrain. To my knowledge, it is the only implementation of such an algorithm. It builds on the idea of merging diffusion model outputs, similar to previous work like [Multi Diffusion](https://arxiv.org/abs/2302.08113). The key challenge lies in ensuring that tiles align seamlessly across different noise levels and timesteps. When denoising one tile, you also need data from surrounding tiles, which have to be processed at the same timestep. This results in a pyramid structure, where tiles must be denoised at decreasing levels. This introduces overhead on the order of $O(t^3)$, where $t$ is the number of sampling timesteps. Thankfully, this overhead largely disappears when many tiles are sampled near each other, since most of these computations can be reused. It also means that the overhead is minimal when $t$ is very small, which these days can be accomplished with [various](https://arxiv.org/abs/2303.01469) [methods](https://arxiv.org/abs/2202.00512). Keep in mind, however, that as $t$ approaches 1, the models lose the ability to "communicate," and the tiling becomes less seamless. Nonetheless, I have found that, at least on toy datasts, the issue is negligible with $t \geq 5$.

Here is the output of the base model (generates 64x64 images where 1 pixel is roughly 4km) on a 1024x1024 map. In real terms, the map has about 1/10th the area of all land on earth. Details are lacking because this image generates very large scale features, but it is a good sign that the image contains both deep seas (black) and mountain ranges (white).
Generating this image on an infinite tiling map takes about 12500 model evaluations, significantly more than the 4410 that would be required for a bounded map. However, on an RTX 3090 Ti this map can still be created in less than a minute, which is not too bad considering the scale of the map.
Note: Pretrained models are WIP but coming soon.
![1024x1024_example](https://github.com/user-attachments/assets/8ea21283-0aee-471d-b470-037ad2b8bd92)

To demonstrate how tiled generation works, I used a toy diffusion model that generates an image sampled from a normal distribution with mean 0 and standard deviation 0.5. In theory, an ideal output would have the entire image being a constant (STD = 0). In practice, adjacent tiles becomes more correlated but do not match exactly, which is exactly what is desired for diverse yet seamless outputs. Indeed, as shown below, adding overlap massively increases the correlation between adjacent tiles (more than I expected, really), and makes the output far smoother. Note that these examples use a bounded sampler, which reduces the overhead around the borders. In fact, in the case where the entire boundary is sampled, it removes the overhead entirely. Infinite (unbounded) generation should look almost identical, except for minor differences at the boundary where outputs are not merged.

Without tiled generation (overlap = 0/64):
Each tile is independent. In this case, each tile takes a normal distribution with STD 0.5
![No Tiled Generation](https://github.com/user-attachments/assets/d305428e-8a70-455d-86cc-8eb68e33254e)

With some tile overlap (overlap = 16/64):
Adjacent tiles become vastly more correlated. STD drops significantly. The image is "seamless" in the sense there is no sharp boundary between tiles, but you can still tell where there tiles are. This should be less of an issue with models that have smoother outputs (as is the case with terrain).
![overlap16](https://github.com/user-attachments/assets/fdc03bee-3e6f-42ea-9d60-1549350a0779)

With full tile overlap (overlap = 32/64):
The image is almost perfectly smooth, but there is not a significant change in STD, which tells us this may be overkill for many applications. Note how the image looks pretty similar to perlin noise! A good property to have when we are generating terrain.
![overlap32](https://github.com/user-attachments/assets/6eeef120-7af4-442b-a740-84008a22a9fb)
