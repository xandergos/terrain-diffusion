from tempfile import NamedTemporaryFile
import click
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from terrain_diffusion.inference.world_pipeline import WorldPipeline, normalize_tensor, resolve_hdf5_path
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.common.cli_helpers import parse_kwargs

BIOME_LEGEND = {
    0: "Unknown",
    1: "Af  Tropical, rainforest",
    2: "Am  Tropical, monsoon",
    3: "Aw  Tropical, savannah",
    4: "BWh Arid, desert, hot",
    5: "BWk Arid, desert, cold",
    6: "BSh Arid, steppe, hot",
    7: "BSk Arid, steppe, cold",
    8: "Csa Temperate, dry summer, hot summer",
    9: "Csb Temperate, dry summer, warm summer",
    10: "Csc Temperate, dry summer, cold summer",
    11: "Cwa Temperate, dry winter, hot summer",
    12: "Cwb Temperate, dry winter, warm summer",
    13: "Cwc Temperate, dry winter, cold summer",
    14: "Cfa Temperate, no dry season, hot summer",
    15: "Cfb Temperate, no dry season, warm summer",
    16: "Cfc Temperate, no dry season, cold summer",
    17: "Dsa Cold, dry summer, hot summer",
    18: "Dsb Cold, dry summer, warm summer",
    19: "Dsc Cold, dry summer, cold summer",
    20: "Dsd Cold, dry summer, very cold winter",
    21: "Dwa Cold, dry winter, hot summer",
    22: "Dwb Cold, dry winter, warm summer",
    23: "Dwc Cold, dry winter, cold summer",
    24: "Dwd Cold, dry winter, very cold winter",
    25: "Dfa Cold, no dry season, hot summer",
    26: "Dfb Cold, no dry season, warm summer",
    27: "Dfc Cold, no dry season, cold summer",
    28: "Dfd Cold, no dry season, very cold winter",
    29: "ET  Polar, tundra",
    30: "EF  Polar, frost",
}


def start_explorer(model_path: str, hdf5_file: str | None = None, seed: int | None = None, coarse_window: int = 64, coarse_offset_i: int = 0, coarse_offset_j: int = 0, detail_size: int = 1024, device: str | None = None, caching_strategy: str = 'indirect', **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Warning: Using CPU (CUDA not available).")

    world = WorldPipeline.from_pretrained(model_path, seed=seed, caching_strategy=caching_strategy, **kwargs)
    world.to(device)
    if caching_strategy == 'direct':
        world.bind(hdf5_file=hdf5_file)
    else:
        world.bind(hdf5_file or 'TEMP')
    with world:
        print(f"World seed: {world.seed}")
        ci0, ci1 = coarse_offset_i - coarse_window, coarse_offset_i + coarse_window
        cj0, cj1 = coarse_offset_j - coarse_window, coarse_offset_j + coarse_window

        # Coarse elevation (signed-sqrt -> meters)
        channel_names = ['Elev', 'p5', 'Temp', 'T std', 'Precip', 'Precip CV']

        coarse_elev_ss = normalize_tensor(world.coarse[:, ci0:ci1, cj0:cj1], dim=0)[0]
        coarse_elev_m = torch.sign(coarse_elev_ss) * torch.square(coarse_elev_ss)
        coarse_np = coarse_elev_m.detach().cpu().numpy()

        fig, (ax_coarse, ax_relief) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Seed: {world.seed}', fontsize=10)
        fig.subplots_adjust(bottom=0.2, top=0.92)
        im = ax_coarse.imshow(
            coarse_np,
            origin='lower',
            interpolation='nearest',
            extent=[cj0, cj1, ci0, ci1],  # x: j, y: i in coarse tile space
            cmap='viridis',
        )
        ax_coarse.set_title('Coarse Elevation (click a tile)')

        # Placeholder for relief map panel (RGB placeholder to match later updates)
        im_relief = ax_relief.imshow(
            np.zeros((2, 2, 3), dtype=np.float32),
            origin='lower',
            interpolation='nearest',
            extent=(0, 2, 0, 2),
        )
        ax_relief.set_title('Relief map')
        ax_relief.axis('off')

        # Keep latest elevation/biome to report values on hover over relief map
        last_elev = None
        last_biome = None
        last_climate = None
        last_climate0 = None
        last_ci = None
        last_cj = None
        right_mode = 'relief'  # 'relief' or 'temperature'

        def relief_format_coord(x, y):
            nonlocal last_elev, last_biome
            if last_elev is None:
                return f"x={x:.1f}, y={y:.1f}"
            ix = int(np.round(x))
            iy = int(np.round(y))
            H, W = last_elev.shape
            if 0 <= ix < W and 0 <= iy < H:
                z = float(last_elev[iy, ix])
                biome_txt = ""
                if last_biome is not None:
                    try:
                        b = int(last_biome[iy, ix])
                        label = BIOME_LEGEND.get(b, "Unknown")
                        biome_txt = f", biome={label}"
                    except Exception:
                        biome_txt = ""
                if np.isfinite(z):
                    return f"x={ix}, y={iy}, elev={z:.1f} m{biome_txt}"
            return f"x={x:.1f}, y={y:.1f}"

        ax_relief.format_coord = relief_format_coord

        # Format coord for coarse map showing all channel values
        coarse_ss_all = world.coarse[:, ci0:ci1, cj0:cj1]
        coarse_ss_all = coarse_ss_all[:-1] / coarse_ss_all[-1:]
        coarse_ss_all_np = coarse_ss_all.detach().cpu().numpy()

        def coarse_format_coord(x, y):
            ix = int(np.floor(x)) - cj0
            iy = int(np.floor(y)) - ci0
            C, H, W = coarse_ss_all_np.shape
            if 0 <= ix < W and 0 <= iy < H:
                vals = [f"{channel_names[c] if c < len(channel_names) else f'C{c}'}={coarse_ss_all_np[c, iy, ix]:.3f}" for c in range(C)]
                return f"i={iy+ci0}, j={ix+cj0} | {', '.join(vals)}"
            return f"x={x:.1f}, y={y:.1f}"

        ax_coarse.format_coord = coarse_format_coord

        def show_right_panel():
            nonlocal last_elev, last_biome, last_climate, last_climate0, right_mode, last_ci, last_cj
            if last_elev is None:
                return
            H, W = last_elev.shape
            if right_mode == 'temperature' and last_climate0 is not None:
                im_relief.set_data(last_climate0)
                im_relief.set_cmap('viridis')
                im_relief.set_extent((0, W, 0, H))
                im_relief.set_clim(np.min(last_climate0), np.max(last_climate0))
                ax_relief.set_xlim(0, W)
                ax_relief.set_ylim(0, H)
                ax_relief.set_aspect('equal', adjustable='box')
                ax_relief.set_title(f'Temperature (ci={last_ci}, cj={last_cj})')
                ax_relief.axis('off')
                fig.canvas.draw_idle()
                return
            
            relief_rgb = get_relief_map(last_elev, None, last_biome, None)
            plt.imsave('latest_relief.png', relief_rgb)
            im_relief.set_data(relief_rgb)
            im_relief.set_cmap(None)
            im_relief.set_extent((0, W, 0, H))
            ax_relief.set_xlim(0, W)
            ax_relief.set_ylim(0, H)
            ax_relief.set_aspect('equal', adjustable='box')
            ax_relief.set_title(f'Relief map (ci={last_ci}, cj={last_cj})')
            ax_relief.axis('off')
            fig.canvas.draw_idle()

        def onclick(event):
            if event.inaxes is not ax_coarse:
                return
            if event.xdata is None or event.ydata is None:
                return

            cj = int(np.floor(event.xdata))
            ci = int(np.floor(event.ydata))

            # Map coarse (i, j) to native resolution grid
            center_i = ci * 256
            center_j = cj * 256
            half = detail_size // 2
            i1, i2 = center_i - half, center_i + half
            j1, j2 = center_j - half, center_j + half

            # Fetch elevation at native resolution
            region_dict = world.get(i1, j1, i2, j2)
            elev = region_dict['elev']
            biome = None
            climate = region_dict['climate']
            elev = elev.cpu().numpy()
            #elev[elev == 0.0] = np.nan

            # Print coordinates at different resolutions
            res = world.native_resolution
            print(f"Generated coarse tile: ci={ci}, cj={cj}")
            print(f"  Note x/z coordinates are flipped.")
            print(f"  1x/{res:.0f}m (i/z, j/x): ({ci * 256}, {cj * 256})")
            print(f"  2x/{res/2:.0f}m (i/z, j/x): ({ci * 512}, {cj * 512})")
            print(f"  4x/{res/4:.1f}m (i/z, j/x): ({ci * 1024}, {cj * 1024})")
            print(f"  8x/{res/8:.2f}m (i/z, j/x): ({ci * 2048}, {cj * 2048})")

            # Store latest data and update right panel based on mode
            nonlocal last_elev, last_biome, last_climate, last_climate0, last_ci, last_cj
            last_elev = elev
            last_biome = biome
            # Store full climate tensor for colorization; show climate0 in dedicated mode
            last_climate = climate.detach().cpu().numpy()
            last_climate0 = last_climate[0]
            last_ci, last_cj = ci, cj
            show_right_panel()

        # Channel selection buttons (0..5)
        selected_channel = 0

        def set_channel(ch: int) -> None:
            nonlocal selected_channel
            selected_channel = int(ch)
            # Update coarse panel with selected channel using same transform
            coarse_ss = world.coarse[:, ci0:ci1, cj0:cj1]
            coarse_ss = coarse_ss[:-1] / coarse_ss[-1:]
            coarse_ss = coarse_ss[selected_channel]
            
            if ch <= 1:
                coarse_ss = torch.sign(coarse_ss) * torch.square(coarse_ss)
            if ch == 4:
                coarse_ss = torch.log(1 + coarse_ss)
            
            data = coarse_ss.detach().cpu().numpy()
            im.set_data(data)
            # Rescale color limits to new data range
            vmin = float(np.nanmin(data)) if np.isfinite(np.nanmin(data)) else 0.0
            vmax = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 1.0
            im.set_clim(vmin, vmax)
            channel_label = channel_names[selected_channel] if selected_channel < len(channel_names) else f'C{selected_channel}'
            ax_coarse.set_title(f'Coarse {channel_label} (click a tile)')
            fig.canvas.draw_idle()

        # Create six buttons along the bottom
        btn_axes = []
        btns = []
        start_x, width, height, bottom = 0.08, 0.12, 0.06, 0.06
        for idx in range(6):
            ax_btn = fig.add_axes([start_x + idx * (width + 0.01), bottom, width, height])
            btn_label = channel_names[idx] if idx < len(channel_names) else f'C{idx}'
            btn = Button(ax_btn, btn_label)
            btn.on_clicked(lambda _evt, ch=idx: set_channel(ch))
            btn_axes.append(ax_btn)
            btns.append(btn)

        # Right panel mode buttons (Relief / Climate0) on a second row to avoid overlap
        def set_mode(mode: str) -> None:
            nonlocal right_mode
            right_mode = mode
            show_right_panel()

        mode_bottom = 0.13
        mode_width = 0.10
        mode_height = 0.06
        ax_mode_relief = fig.add_axes([0.76, mode_bottom, mode_width, mode_height])
        ax_mode_clim0 = fig.add_axes([0.87, mode_bottom, mode_width, mode_height])
        btn_mode_relief = Button(ax_mode_relief, 'Relief')
        btn_mode_clim0 = Button(ax_mode_clim0, 'Temperature')
        btn_mode_relief.on_clicked(lambda _evt: set_mode('relief'))
        btn_mode_clim0.on_clicked(lambda _evt: set_mode('temperature'))

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)


@click.command()
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.option("--caching-strategy", type=click.Choice(["indirect", "direct"]), default="direct", help="Caching strategy: 'indirect' uses HDF5, 'direct' uses in-memory LRU cache")
@click.option("--hdf5-file", default=None, help="HDF5 file path (required for indirect caching, optional for direct)")
@click.option("--max-cache-size", type=int, default=None, help="Max cache size in bytes (for direct caching)")
@click.option("--seed", type=int, default=None, help="Random seed (default: random)")
@click.option("--device", default=None, help="Device (cuda/cpu, default: auto)")
@click.option("--batch-size", type=str, default="1,4", help="Batch size(s) for latent generation (e.g. '4' or '1,2,4,8')")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose", help="Logging mode")
@click.option("--compile/--no-compile", "torch_compile", default=True, help="Use torch.compile for faster inference")
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default=None, help="Model dtype (default: fp32)")
@click.option("--coarse-window", type=int, default=50, help="Coarse window size")
@click.option("--coarse-offset-i", type=int, default=0, help="Coarse window offset in i direction")
@click.option("--coarse-offset-j", type=int, default=0, help="Coarse window offset in j direction")
@click.option("--detail-size", type=int, default=1024, help="Full resolution detail window size")
@click.option("--kwarg", "extra_kwargs", multiple=True, help="Additional key=value kwargs (e.g. --kwarg native_resolution=30)")
def main(model_path, hdf5_file, caching_strategy, max_cache_size, seed, device, batch_size, log_mode, torch_compile, dtype, coarse_window, coarse_offset_i, coarse_offset_j, detail_size, extra_kwargs):
    """Explore a generated world interactively"""
    if caching_strategy == 'indirect' and hdf5_file is None:
        hdf5_file = 'TEMP'
    if hdf5_file is not None:
        hdf5_file = resolve_hdf5_path(hdf5_file)
    # Parse batch size(s)
    if ',' in batch_size:
        batch_sizes = [int(x.strip()) for x in batch_size.split(',')]
    else:
        batch_sizes = int(batch_size)
    # Normalize dtype
    if dtype == 'fp32':
        dtype = None
    start_explorer(
        model_path,
        hdf5_file,
        seed=seed,
        coarse_window=coarse_window,
        coarse_offset_i=coarse_offset_i,
        coarse_offset_j=coarse_offset_j,
        detail_size=detail_size,
        device=device,
        latents_batch_size=batch_sizes,
        log_mode=log_mode,
        torch_compile=torch_compile,
        dtype=dtype,
        caching_strategy=caching_strategy,
        cache_limit=max_cache_size,
        **parse_kwargs(extra_kwargs),
    )


if __name__ == '__main__':
    main()
