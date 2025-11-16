import argparse
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_diffusion.inference.relief_map import get_relief_map


def _to_numpy_f32(x) -> np.ndarray:
    """Convert torch.Tensor or array-like to numpy float32 on CPU without copying unnecessarily."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).numpy()
    return np.asarray(x, dtype=np.float32)


def start_explorer(hdf5_file: str, seed: int, coarse_window: int = 64, device: str | None = None, **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with WorldPipeline(hdf5_file, device=device, seed=seed, **kwargs) as world:
        ci0, ci1 = -coarse_window, coarse_window
        cj0, cj1 = -coarse_window, coarse_window

        # Coarse elevation (signed-sqrt -> meters)
        coarse_elev_ss = world.coarse[0, ci0:ci1, cj0:cj1]
        coarse_elev_m = torch.sign(coarse_elev_ss) * torch.square(coarse_elev_ss)
        coarse_np = coarse_elev_m.detach().cpu().numpy()

        fig, (ax_coarse, ax_relief) = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(bottom=0.2)
        im = ax_coarse.imshow(
            coarse_np,
            origin='lower',
            interpolation='nearest',
            extent=[cj0, cj1, ci0, ci1],  # x: j, y: i in coarse tile space
            cmap='viridis',
        )
        ax_coarse.set_title('Coarse channel 0 (click a tile)')

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
        last_ci = None
        last_cj = None

        def relief_format_coord(x, y):
            nonlocal last_elev
            if last_elev is None:
                return f"x={x:.1f}, y={y:.1f}"
            ix = int(np.round(x))
            iy = int(np.round(y))
            H, W = last_elev.shape
            if 0 <= ix < W and 0 <= iy < H:
                z = float(last_elev[iy, ix])
                biome_txt = ""
                if np.isfinite(z):
                    return f"x={ix}, y={iy}, elev={z:.1f} m{biome_txt}"
            return f"x={x:.1f}, y={y:.1f}"

        ax_relief.format_coord = relief_format_coord

        def show_right_panel():
            nonlocal last_elev, last_ci, last_cj
            if last_elev is None:
                return
            H, W = last_elev.shape
            # Export a tiff of last_elev
            try:
                import tifffile
                # If you want to change the filename dynamically, modify the string below
                output_path = f"elev_export.tif"
                data_to_save = _to_numpy_f32(last_elev)
                tifffile.imwrite(output_path, data_to_save.astype(np.float32))
                print(f"Exported last_elev as {output_path}")
            except ImportError:
                print("tifffile package is required for TIFF export. Please install it with 'pip install tifffile'.")
            except Exception as e:
                print(f"Error exporting TIFF: {e}")
                
            # Also export a Unity-compatible RAW heightmap (16-bit little-endian, normalized 0..1)
            try:
                raw_path = "elev_export.raw"
                data_np = _to_numpy_f32(last_elev)
                mask = np.isfinite(data_np)
                if np.any(mask):
                    vmin = float(np.min(data_np[mask]))
                    vmax = float(np.max(data_np[mask]))
                    if vmax > vmin:
                        norm = (data_np - vmin) / (vmax - vmin)
                    else:
                        norm = np.zeros_like(data_np, dtype=np.float32)
                else:
                    norm = np.zeros_like(data_np, dtype=np.float32)
                norm[~mask] = 0.0
                raw_u16_le = (np.clip(norm, 0.0, 1.0) * 65535.0 + 0.5).astype('<u2')
                raw_u16_le.tofile(raw_path)
                print(f"Exported RAW (16-bit little-endian) as {raw_path} with shape {raw_u16_le.shape}. vmin={vmin}, vmax={vmax}")
            except Exception as e:
                print(f"Error exporting RAW: {e}")
                
            relief_rgb = get_relief_map(last_elev, None, None, None, resolution=11.25)
            relief_rgb[last_elev <= 0] = np.nan
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

            # Map coarse (i, j) to 11 m grid
            center_i_90 = ci * 256 * 8
            center_j_90 = cj * 256 * 8
            half = 1024
            i1, i2 = center_i_90 - half, center_i_90 + half
            j1, j2 = center_j_90 - half, center_j_90 + half

            # Fetch elevation at 11 m resolution
            region_dict = world.get_11(i1, j1, i2, j2)
            elev = region_dict['elev']
            elev = np.sign(elev) * elev**2
            #elev[elev == 0.0] = np.nan

            # Store latest data and update right panel based on mode
            nonlocal last_elev, last_ci, last_cj
            last_elev = elev
            last_ci, last_cj = ci, cj
            show_right_panel()

        # Channel selection buttons (0..5)
        selected_channel = 0

        def set_channel(ch: int) -> None:
            nonlocal selected_channel
            selected_channel = int(ch)
            # Update coarse panel with selected channel using same transform
            coarse_ss = world.coarse[selected_channel, ci0:ci1, cj0:cj1]
            data = coarse_ss.detach().cpu().numpy()
            im.set_data(data)
            # Rescale color limits to new data range
            vmin = float(np.nanmin(data)) if np.isfinite(np.nanmin(data)) else 0.0
            vmax = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 1.0
            im.set_clim(vmin, vmax)
            ax_coarse.set_title(f'Coarse channel {selected_channel} (click a tile)')
            fig.canvas.draw_idle()

        # Create six buttons along the bottom
        btn_axes = []
        btns = []
        start_x, width, height, bottom = 0.08, 0.12, 0.06, 0.06
        for idx in range(6):
            ax_btn = fig.add_axes([start_x + idx * (width + 0.01), bottom, width, height])
            btn = Button(ax_btn, f'C{idx}')
            btn.on_clicked(lambda _evt, ch=idx: set_channel(ch))
            btn_axes.append(ax_btn)
            btns.append(btn)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)


if __name__ == '__main__':
    with NamedTemporaryFile(suffix='.h5') as tmp_file:
        start_explorer('world_mid.h5', 1, device='cuda', coarse_window=64,
                    drop_water_pct=0.5,
                    frequency_mult=[1.0, 1.0, 1.0, 1.0, 1.0],
                    cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0],
                    log_mode='debug')