import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from confection import Config, registry
from ema_pytorch import PostHocEMA

from terrain_diffusion.training.registry import build_registry
from pyperlin import FractalPerlin2D

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hardcoded paths/settings
    config_path = "configs/gan/injection_gan.cfg"
    ema_sigma_rel = 0.05
    ema_step = None
    batch_size = 1  
    channels = 6
    height = 256
    width = 256
    

    shape = (1,height,width) #for batch size = 1 and noises' shape = (1024,1024)
    g_cuda = torch.Generator(device='cuda') #for GPU acceleration

    base_resolutions = [(2**i,2**i) for i in range(3,9)] #for lacunarity = 2.0
    base_factors = [.5**i for i in range(2, 8)] #for persistence = 0.5
    base = FractalPerlin2D(shape, base_resolutions, base_factors, generator=g_cuda)() / 0.06

    # Build registry and resolve minimal config to get the generator
    build_registry()
    cfg = Config().from_disk(config_path)
    resolved = registry.resolve(cfg, validate=False)

    generator = resolved["generator"].to(device)
    generator.eval()

    # Apply EMA weights from checkpoint folder
    resolved["ema"]["checkpoint_folder"] = os.path.join(resolved["logging"]["save_dir"], "phema")
    ema = PostHocEMA(generator, **resolved["ema"]).to(device)
    ema_ckpt = os.path.join(resolved["logging"]["save_dir"], "latest_checkpoint", "phema.pt")
    if os.path.exists(ema_ckpt):
        ema.load_state_dict(torch.load(ema_ckpt, map_location="cpu", weights_only=False))
        ema.synthesize_ema_model(sigma_rel=ema_sigma_rel, step=ema_step).copy_params_from_ema_to_model()

    # Pure noise input and t = pi/2
    latents = torch.randn(batch_size, 32, height // 2 + 6, width // 2 + 6, device=device)
    z = torch.randn(batch_size, channels, height, width, device=device)
    t = torch.full((batch_size, channels), np.arctan(160), device=device)
    t[0] = 0.1
    z[:, 0] = torch.cos(t[0, 0]) * base[0] + torch.sin(t[0, 0]) * z[:, 0]
    imgs, _ = generator(latents, z, t)  # [B, C, H, W]
    means = [-2633.639460630914, 189.935819107797, 7.339399687579161, 205.22559684753017, 457.59783059226277, 25.97546207333763]
    stds = [2428.9742483056802, 234.03388226663458, 9.7756394724636, 360.7426708787508, 703.1262351033647, 36.3729825977694]
    imgs = (imgs * torch.tensor(stds, device=device).reshape(1, 6, 1, 1) + torch.tensor(means, device=device).reshape(1, 6, 1, 1))
    #imgs[:, 1] = imgs[:, 0] - imgs[:, 1]
    print("% of 1st channel < 0:", (imgs[:, 0] < 0).float().mean().item())
    print("% of 2nd channel < 0:", (imgs[:, 1] < 0).float().mean().item())
    ocean_mask = (imgs[:, :1] < 0).repeat(1, 6, 1, 1)
    imgs[ocean_mask] = 0
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    im = ax.imshow(imgs[0, 0].detach().cpu().numpy(), cmap="terrain", interpolation="nearest")
    ax.axis("off")
    fig.suptitle("GAN-Exp sample (channel 0), t=pi/2, noise input")

    # Slider to change channels
    from matplotlib.widgets import Slider
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.2, 0.06, 0.6, 0.03])
    slider = Slider(ax_slider, 'Channel', 0, channels - 1, valinit=0, valfmt='%d')

    def update(_):
        ch = int(slider.val)
        data = imgs[0, ch].detach().cpu().numpy()
        im.set_data(data)
        im.set_clim(data.min(), data.max())
        fig.suptitle(f"GAN-Exp sample (channel {ch}), t=pi/2, noise input")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()


