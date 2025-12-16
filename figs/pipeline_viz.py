import os
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from terrain_diffusion.inference.world_pipeline import WorldPipeline, normalize_tensor
from terrain_diffusion.inference.relief_map import get_relief_map

OUTPUT_DIR = "figs/pipeline_viz"


def save_image(data: np.ndarray, path: str, vmin: float = None, vmax: float = None, cmap: str = 'viridis') -> None:
    """Save array as image with no padding, using nearest neighbor interpolation."""
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    if vmax > vmin:
        normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(data)
    normalized = np.clip(normalized, 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0)
    cm = plt.get_cmap(cmap)
    rgb = cm(normalized[::-1])[:, :, :3]
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    img.save(path, resample=Image.NEAREST)


def create_stacked_composite(images: list[np.ndarray], canvas_size: int = 2048, border: int = 2) -> np.ndarray:
    """Create a stacked composite where images fan out from bottom-left to top-right.
    
    The first image (elevation) is in foreground at bottom-left, subsequent images
    are behind it shifted up and to the right. Last image's top-right hits canvas top-right.
    """
    n = len(images)
    if n == 0:
        return np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    # Each image takes up most of the canvas
    img_size = images[0].shape[0]
    total_offset = canvas_size - img_size
    shift_per_layer = total_offset / max(1, n - 1) if n > 1 else 0
    
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    
    # Draw from back to front (last image first, elevation last)
    for idx in range(n - 1, -1, -1):
        img = images[idx]
        offset = int(idx * shift_per_layer)
        y0, x0 = total_offset - offset, offset
        
        # Add black border by first drawing a slightly larger black rect
        if border > 0:
            canvas[y0:y0 + img_size, x0:x0 + img_size] = 0
        
        # Draw image inside the border
        by0, bx0 = y0 + border, x0 + border
        by1, bx1 = y0 + img_size - border, x0 + img_size - border
        canvas[by0:by1, bx0:bx1] = img[border:img_size - border, border:img_size - border]
    
    return canvas


def visualize_pipeline(hdf5_file: str, seed: int, device: str | None = None, **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with WorldPipeline(hdf5_file, device=device, seed=seed, **kwargs) as world:
        coarse_window = 7
        offset = 0
        ci0, ci1 = offset, coarse_window + 1 + offset
        cj0, cj1 = offset, coarse_window + 1 + offset
        
        # Get raw data
        synthetic_map = world.synthetic_map_factory(ci0, cj0, ci1, cj1).numpy()
        coarse_raw = world.coarse[:, ci0:ci1, cj0:cj1]
        coarse_normalized = normalize_tensor(coarse_raw, dim=0).detach().cpu().numpy()
        
        # Channel mapping: (synth_idx, coarse_idx, name, transform)
        # Transform: 'elev' = undo signed-sqrt, 'log' = log1p, None = identity
        channels = [
            (0, 0, 'elev', 'elev'),       # synthetic elev ↔ coarse elev_mean
            (1, 2, 'temp', None),          # synthetic temp ↔ coarse temp
            (2, 3, 'temp_std', None),      # synthetic temp_std ↔ coarse temp_std
            (3, 4, 'precip', 'log'),       # synthetic precip ↔ coarse precip
            (4, 5, 'precip_std', None),    # synthetic precip_std ↔ coarse precip_std
        ]
        
        for synth_idx, coarse_idx, name, transform in channels:
            synth_data = synthetic_map[synth_idx].copy()
            coarse_data = coarse_normalized[coarse_idx].copy()
            
            if transform == 'elev':
                synth_data = np.sign(synth_data) * np.square(synth_data)
                coarse_data = np.sign(coarse_data) * np.square(coarse_data)
            elif transform == 'log':
                synth_data = np.log1p(np.maximum(synth_data, 0))
                coarse_data = np.log1p(np.maximum(coarse_data, 0))
            
            # Shared normalization
            vmin = min(np.nanmin(synth_data), np.nanmin(coarse_data))
            vmax = max(np.nanmax(synth_data), np.nanmax(coarse_data))
            
            save_image(synth_data, f"{OUTPUT_DIR}/synthetic_{name}.png", vmin, vmax)
            save_image(coarse_data, f"{OUTPUT_DIR}/coarse_{name}.png", vmin, vmax)
        
        # Extra coarse channel: elev_p5
        coarse_elev_p5 = np.sign(coarse_normalized[1]) * np.square(coarse_normalized[1])
        save_image(coarse_elev_p5, f"{OUTPUT_DIR}/coarse_elev_p5.png")
        
        print(f"Saved synthetic_*.png and coarse_*.png")
        
        # ====== 3. Relief Map and Climate (2048x2048 at 90m resolution) ======
        size = 256
        i1, i2 = size * offset, size * (coarse_window + 1 + offset)
        j1, j2 = size * offset, size * (coarse_window + 1 + offset)
        
        region_dict = world.get(i1, j1, i2, j2, with_climate=True)
        elev = region_dict['elev']
        if hasattr(elev, 'detach'):
            elev = elev.detach().cpu().numpy()
        elev_m = elev
        
        relief_rgb = get_relief_map(elev_m, None, None, None)
        relief_rgb[elev_m <= 0] = np.nan
        relief_rgb = np.nan_to_num(relief_rgb, nan=0.0)
        img = Image.fromarray((relief_rgb[::-1] * 255).astype(np.uint8))
        img.save(f"{OUTPUT_DIR}/relief_map.png", resample=Image.NEAREST)
        print(f"Saved relief_map.png")
        
        # Save climate channels
        climate = region_dict['climate']
        if climate is not None:
            if hasattr(climate, 'detach'):
                climate = climate.detach().cpu().numpy()
            climate_names = ['temp', 'temp_std', 'precip', 'precip_std', 'beta']
            climate_transforms = [None, None, 'log', None, None]
            for i, (name, transform) in enumerate(zip(climate_names, climate_transforms)):
                data = climate[i].copy()
                if transform == 'log':
                    data = np.log1p(np.maximum(data, 0))
                save_image(data, f"{OUTPUT_DIR}/climate_{name}.png")
            print(f"Saved climate_*.png")
        
        # ====== 4. Stacked Composites ======
        # Resize individual channel images to fit in stacked composite
        composite_size = 2048
        img_ratio = 0.88  # Each image takes 88% of canvas, leaving room for stacking
        img_size = int(composite_size * img_ratio)
        
        def resize_for_composite(data: np.ndarray, cmap: str = 'viridis', nearest: bool = False) -> np.ndarray:
            """Convert data to RGB and resize for composite."""
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            if vmax > vmin:
                normalized = (data - vmin) / (vmax - vmin)
            else:
                normalized = np.zeros_like(data)
            normalized = np.clip(normalized, 0, 1)
            normalized = np.nan_to_num(normalized, nan=0.0)
            cm = plt.get_cmap(cmap)
            rgb = cm(normalized[::-1])[:, :, :3]
            pil_img = Image.fromarray((rgb * 255).astype(np.uint8))
            resample = Image.NEAREST if nearest else Image.BILINEAR
            pil_img = pil_img.resize((img_size, img_size), resample)
            return np.array(pil_img)
        
        # Synthetic composite: elev, temp, temp_std, precip, precip_std
        synth_layers = []
        for synth_idx, _, name, transform in channels:
            data = synthetic_map[synth_idx].copy()
            if transform == 'elev':
                data = np.sign(data) * np.square(data)
            elif transform == 'log':
                data = np.log1p(np.maximum(data, 0))
            synth_layers.append(resize_for_composite(data, nearest=True))
        synth_composite = create_stacked_composite(synth_layers, composite_size)
        Image.fromarray(synth_composite).save(f"{OUTPUT_DIR}/composite_synthetic.png")
        
        # Coarse composite: elev_mean, elev_p5, temp, temp_std, precip, precip_std
        coarse_layers = []
        coarse_channel_info = [
            (0, 'elev_mean', 'elev'),
            (1, 'elev_p5', 'elev'),
            (2, 'temp', None),
            (3, 'temp_std', None),
            (4, 'precip', 'log'),
            (5, 'precip_std', None),
        ]
        for coarse_idx, name, transform in coarse_channel_info:
            data = coarse_normalized[coarse_idx].copy()
            if transform == 'elev':
                data = np.sign(data) * np.square(data)
            elif transform == 'log':
                data = np.log1p(np.maximum(data, 0))
            coarse_layers.append(resize_for_composite(data, nearest=True))
        coarse_composite = create_stacked_composite(coarse_layers, composite_size)
        Image.fromarray(coarse_composite).save(f"{OUTPUT_DIR}/composite_coarse.png")
        
        # Elevation composite: elev + climate channels
        elev_layers = [resize_for_composite(elev_m)]
        if climate is not None:
            for i, (name, transform) in enumerate(zip(climate_names, climate_transforms)):
                data = climate[i].copy()
                if transform == 'log':
                    data = np.log1p(np.maximum(data, 0))
                elev_layers.append(resize_for_composite(data))
        elev_composite = create_stacked_composite(elev_layers, composite_size)
        Image.fromarray(elev_composite).save(f"{OUTPUT_DIR}/composite_elevation.png")
        
        # Elevation composite with shaded relief
        def resize_rgb_for_composite(rgb: np.ndarray) -> np.ndarray:
            """Resize RGB array for composite."""
            pil_img = Image.fromarray((rgb[::-1] * 255).astype(np.uint8))
            pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
            return np.array(pil_img)
        
        elev_layers_relief = [resize_rgb_for_composite(relief_rgb)]
        if climate is not None:
            for i, (name, transform) in enumerate(zip(climate_names, climate_transforms)):
                data = climate[i].copy()
                if transform == 'log':
                    data = np.log1p(np.maximum(data, 0))
                elev_layers_relief.append(resize_for_composite(data))
        elev_composite_relief = create_stacked_composite(elev_layers_relief, composite_size)
        Image.fromarray(elev_composite_relief).save(f"{OUTPUT_DIR}/composite_elevation_relief.png")
        
        print(f"Saved composite_*.png")


if __name__ == '__main__':
    with NamedTemporaryFile(suffix='.h5') as tmp_file:
        visualize_pipeline(
            tmp_file.name, device='cuda', seed=1, log_mode='debug',
            drop_water_pct=0.5,
            frequency_mult=[1.0, 1.0, 1.0, 1.0, 1.0],
            cond_snr=[0.5, 0.5, 0.5, 0.5, 0.5],
            histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.5],
            mode="a",
            latents_batch_size=32,
        )

