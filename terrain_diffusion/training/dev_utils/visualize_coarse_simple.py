import rasterio
import torch
from confection import Config, registry
from terrain_diffusion.inference.synthetic_map import make_synthetic_map_factory
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.models.edm_unet import EDMUnet2D
import matplotlib.pyplot as plt
import random
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from terrain_diffusion.inference.perlin_transform import build_quantiles, transform_perlin
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from terrain_diffusion.training.evaluation.sample_coarse import sample_coarse_tiled

def build_perlin_noise(height, width):
    factory = make_synthetic_map_factory()
    return factory(0, 0, height, width)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, channels, height, width = 1, 6, 128, 128
    steps = 20
    dtype = torch.float32
    generator = torch.Generator(device=device)

    # Load config and registry-resolved scheduler
    build_registry()
    cfg_path = "configs/diffusion_coarse/diffusion_coarse.cfg"
    cfg = Config().from_disk(cfg_path)
    resolved = registry.resolve(cfg, validate=False)
    scheduler = resolved["scheduler"]
    train_dset = resolved["train_dataset"]

    # Load model weights (expects default latest checkpoint layout)
    ckpt_dir = "checkpoints/diffusion_coarse/latest_checkpoint/saved_model"
    model = EDMUnet2D.from_pretrained(ckpt_dir)
    model = model.to(device)
    model.eval()

    # Run sampling
    synthetic_map = build_perlin_noise(height, width)
    synthetic_map = (synthetic_map - train_dset.means[[0, 2, 3, 4, 5], None, None]) / train_dset.stds[[0, 2, 3, 4, 5], None, None]
    synthetic_map = synthetic_map.to(device)
    while True:
        samples = sample_coarse_tiled(
            model=model,
            scheduler=scheduler,
            cond_img=synthetic_map.view(1, 5, height, width),
            cond_snr=torch.tensor([0.1, 0.5, 0.5, 0.5, 0.5]).view(1, 5)
        )

        # Simple confirmation with channel slider
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.15)
        
        # Append synthetic map channels to samples for viewing
        # synthetic_map is normalized with means/stds for indices [0,2,3,4,5]
        samples = torch.cat([samples, synthetic_map.unsqueeze(0)], dim=1)
        
        # Build per-channel stats for display (first 6 from dataset, last 5 map to [0,2,3,4,5])
        ds_means = train_dset.means
        ds_stds = train_dset.stds
        synth_idx = [0, 2, 3, 4, 5]
        display_means = np.concatenate([ds_means[:6], ds_means[synth_idx]])
        display_stds = np.concatenate([ds_stds[:6], ds_stds[synth_idx]])
        channels_total = samples.shape[1]
        
        # Initial channel
        current_channel = [0]
        im = [None]  # Store image object for colorbar updates
        cbar = [None]  # Store colorbar object
        
        def update_display(channel_idx):
            ax.clear()
            if channel_idx != 1:
                data = samples[0, channel_idx, :, :].detach().cpu().numpy()
                data = data * display_stds[channel_idx] + display_means[channel_idx]
            else:
                data0 = samples[0, 0, :, :].detach().cpu().numpy()
                data0 = data0 * display_stds[0] + display_means[0]
                data1 = samples[0, 1, :, :].detach().cpu().numpy()
                data1 = data1 * display_stds[1] + display_means[1]
                data = data0 - data1
            im[0] = ax.imshow(data)
            ax.set_title(f'Channel {channel_idx}')
            
            # Update or create colorbar (avoid removing axes to prevent KeyError)
            if cbar[0] is None:
                cbar[0] = plt.colorbar(im[0], ax=ax)
            else:
                cbar[0].update_normal(im[0])
            
            plt.draw()
        
        # Create slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = plt.Slider(ax_slider, 'Channel', 0, channels_total - 1, valinit=0, valstep=1)
        
        def on_slider_change(val):
            current_channel[0] = int(val)
            update_display(current_channel[0])
        
        slider.on_changed(on_slider_change)
        
        # Initial display
        update_display(current_channel[0])
        plt.show()


