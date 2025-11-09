import torch
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_world(hdf5_file: str, seed: int, coarse_window: int = 64, device: str | None = None, **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with WorldPipeline(hdf5_file, device=device, seed=seed, **kwargs) as world:
        ci0, ci1 = -coarse_window, coarse_window
        cj0, cj1 = -coarse_window, coarse_window

        pbar = tqdm(total=(ci1-ci0)*(cj1-cj0), desc="Generating world")
        for i in range(ci0, ci1):
            for j in range(cj0, cj1):
                world.latents[:, i*32:i*32+64, j*32:j*32+64]
                pbar.update(1)
        
        print("Generating residual")
        
        pbar = tqdm(total=(ci1-ci0)*(cj1-cj0), desc="Generating world")
        for i in range(ci0, ci1):
            for j in range(cj0, cj1):
                world.residual_90[:, i*256:i*256+512, j*256:j*256+512]
                pbar.update(1)
            
    
    

if __name__ == "__main__":
    generate_world('world_big.h5', 1, device='cuda', coarse_window=76,
                    drop_water_pct=0.0,
                    frequency_mult=[0.7, 0.7, 0.7, 0.7, 0.7],
                    cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0],)