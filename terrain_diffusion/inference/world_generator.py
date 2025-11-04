import torch
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_world(hdf5_file: str, seed: int, coarse_window: int = 64, device: str | None = None, **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    world = WorldPipeline(hdf5_file, device=device, seed=seed, **kwargs)
    
    ci0, ci1 = -coarse_window, coarse_window
    cj0, cj1 = -coarse_window, coarse_window

    pbar = tqdm(total=(ci1-ci0)*(cj1-cj0), desc="Generating world")
    for i in range(ci0, ci1):
        for j in range(cj0, cj1):
            world.latents[:, i*32:i*32+64, j*32:j*32+64]
            pbar.update(1)
            
    lowfreq = world.latents[4, ci0*32:ci1*32, cj0*32:cj1*32]
    plt.imshow(lowfreq.detach().cpu().numpy(), cmap='terrain')
    plt.show()

if __name__ == "__main__":
    generate_world('temp.h5', 854, device='cuda', coarse_window=64,
               frequency_mult=[1.0, 1.0, 1.0, 1.0, 1.0],
               cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0])