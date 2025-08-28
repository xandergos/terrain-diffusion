import os
import json
import click
import torch
import matplotlib.pyplot as plt

from terrain_diffusion.common.model_utils import get_model
from terrain_diffusion.training.gan.generator import MPGenerator


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--ckpt", "ckpt_dir", type=click.Path(exists=True, file_okay=False), required=True, help="Checkpoint directory containing model_config")
@click.option("--sigma-rel", type=float, default=0.2, show_default=True, help="EMA sigma_rel for synthesis")
@click.option("--latent-size", type=int, default=None, help="Latent spatial size; defaults to training config value or 21")
@click.option("--seed", type=int, default=None, help="Random seed")
@click.option("--device", default="cuda", show_default=True, help="Device string: cuda or cpu")
@torch.no_grad()
def main(ckpt_dir, sigma_rel, latent_size, seed, device):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Load generator with EMA weights copied in
    generator = get_model(MPGenerator, ckpt_dir, sigma_rel=sigma_rel, device=device)
    generator.eval()

    # Determine latent shape
    z_channels = int(generator.config["latent_channels"])

    # Seed
    if seed is not None:
        torch.manual_seed(seed)

    # Single sample
    z = torch.randn(1, z_channels, latent_size, latent_size, device=device)
    img = generator(z)  # [1, C, H, W]

    # Prepare normalized channels
    elev = img[0, 0] * 2435.7434 + -2607.8887
    elev = torch.sign(elev) * torch.sqrt(torch.abs(elev))
    others = img[0, 1:]

    # Interactive visualization with slider
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    # Initial display (elevation channel)
    data0 = elev.detach().cpu().numpy()
    im = ax_img.imshow(data0, cmap="terrain", interpolation="nearest")
    ax_img.set_title(f"Channel 0 (Elevation)")
    ax_img.axis("off")

    # Initial histogram
    ax_hist.hist(data0.ravel(), bins=50, color="gray", alpha=0.8)
    ax_hist.set_title("Histogram of values")
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Count")

    # Slider
    from matplotlib.widgets import Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Channel', 0, img.shape[1] - 1, valinit=0, valfmt='%d')

    def update(val):
        channel = int(slider.val)
        if channel == 0:
            # Elevation channel
            data = elev.detach().cpu().numpy()
            im.set_data(data)
            im.set_cmap("terrain")
            ax_img.set_title(f"Channel {channel} (Elevation)")
        else:
            # Other channels
            data = others[channel - 1].detach().cpu().numpy()
            im.set_data(data)
            im.set_cmap("viridis")
            ax_img.set_title(f"Channel {channel}")
        
        im.set_clim(data.min(), data.max())
        ax_hist.cla()
        ax_hist.hist(data.ravel(), bins=50, color="gray", alpha=0.8)
        ax_hist.set_title("Histogram of values")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Count")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()