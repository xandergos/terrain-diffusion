import argparse
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_diffusion.inference.relief_map import get_relief_map

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


def view_world(hdf5_file: str, seed: int, i0: int, j0: int, i1: int, j1: int,
               device: str | None = None, stride=8, resolution_div=2, save_dir=None, relief=1.0, **kwargs) -> np.ndarray:
    from PIL import Image
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Warning: Using CPU (CUDA not available).")

    with WorldPipeline(hdf5_file, device=device, seed=seed, mode='r', **kwargs) as world:
        H_out = (i1 - i0) // stride
        W_out = (j1 - j0) // stride
        full_elev = np.zeros((H_out, W_out), dtype=np.float32)

        TILE = 2048
        OUT_TILE = max(1, TILE // max(1, stride))
        pbar = tqdm(total=(i1-i0)*(j1-j0), desc="Viewing world")
        for oi in range(0, H_out, OUT_TILE):
            h_out = min(OUT_TILE, H_out - oi)
            is_ = i0 + oi * stride
            h = h_out * stride
            for oj in range(0, W_out, OUT_TILE):
                w_out = min(OUT_TILE, W_out - oj)
                js = j0 + oj * stride
                w = w_out * stride

                region = world.get(is_, js, is_ + h, js + w, with_climate=False)
                elev = region['elev'].detach().cpu().numpy()

                if stride > 1:
                    elev_t = torch.from_numpy(elev).to(torch.float32)[None, None]
                    elev_ds = torch.nn.functional.avg_pool2d(elev_t, kernel_size=stride, stride=stride)[0, 0].numpy()
                else:
                    elev_ds = elev

                full_elev[oi:oi + h_out, oj:oj + w_out] = elev_ds[:h_out, :w_out]
                pbar.update(h * w)

        land_pixels = np.sum(full_elev >= 0)
        ocean_pixels = np.sum(full_elev < 0)
        total_pixels = full_elev.size
        print(f"Land: {land_pixels:,} ({100*land_pixels/total_pixels:.1f}%), Ocean: {ocean_pixels:,} ({100*ocean_pixels/total_pixels:.1f}%)")

        img = get_relief_map(full_elev, None, None, None, resolution=90*stride/resolution_div, relief=relief)
        
        # Flip vertically to match origin='lower'
        out_img = np.clip(img[::-1], 0.0, 1.0)
        out_u8 = (out_img * 255.0 + 0.5).astype(np.uint8)
        if save_dir:
            Image.fromarray(out_u8).save(save_dir)
        return out_u8


def draw_box(img: np.ndarray, top: int, left: int, bottom: int, right: int, color=(255, 0, 0), thickness=3):
    """Draw a rectangle on the image (in-place)."""
    H, W = img.shape[:2]
    top, bottom = max(0, top), min(H, bottom)
    left, right = max(0, left), min(W, right)
    # Top edge
    img[top:top+thickness, left:right] = color
    # Bottom edge
    img[bottom-thickness:bottom, left:right] = color
    # Left edge
    img[top:bottom, left:left+thickness] = color
    # Right edge
    img[top:bottom, right-thickness:right] = color


def generate_zoom_sequence(hdf5_file: str, seed: int, views: list, output_path: str, device: str | None = None, size: int = 1024):
    """Generate images at multiple zoom levels and concatenate with red boxes showing zoom regions."""
    from PIL import Image
    
    coords = []
    
    # Generate and save each image
    for idx, view in enumerate(views):
        i0, j0, i1, j1 = view['i0'], view['j0'], view['i1'], view['j1']
        stride = view.get('stride', 1)
        kwargs = {k: v for k, v in view.items() if k not in ('i0', 'j0', 'i1', 'j1')}
        view_world(hdf5_file, seed, i0, j0, i1, j1, device=device, save_dir=f'world_{idx}.png', **kwargs)
        coords.append((i0, j0, i1, j1, stride))
    
    # Load images, resize to 1024x1024
    images = []
    for idx in range(len(views)):
        img = Image.open(f'world_{idx}.png')
        img = img.resize((size, size), Image.LANCZOS)
        images.append(np.array(img))
    
    # Draw red boxes on each image (except the last) showing where the next zoom region is
    for idx in range(len(images) - 1):
        cur_i0, cur_j0, cur_i1, cur_j1, cur_stride = coords[idx]
        next_i0, next_j0, next_i1, next_j1, _ = coords[idx + 1]
        
        cur_H = (cur_i1 - cur_i0) // cur_stride
        cur_W = (cur_j1 - cur_j0) // cur_stride
        
        # Convert next region coords to current image pixel coords (accounting for vertical flip)
        box_left = (next_j0 - cur_j0) / cur_stride * size / cur_W
        box_right = (next_j1 - cur_j0) / cur_stride * size / cur_W
        box_bottom = (cur_H - (next_i0 - cur_i0) / cur_stride) * size / cur_H
        box_top = (cur_H - (next_i1 - cur_i0) / cur_stride) * size / cur_H
        
        draw_box(images[idx], int(box_top), int(box_left), int(box_bottom), int(box_right), thickness=max(2, size // 200))
    
    # Concatenate horizontally with white separator lines
    separator = np.full((size, 15, 3), 255, dtype=np.uint8)
    parts = []
    for i, img in enumerate(images):
        if i > 0:
            parts.append(separator)
        parts.append(img)
    combined = np.concatenate(parts, axis=1)
    Image.fromarray(combined).save(output_path)

if __name__ == '__main__':
    common_kwargs = dict(
        drop_water_pct=0.5,
        frequency_mult=[1.0, 1.0, 1.0, 1.0, 1.0],
        cond_snr=[0.5, 0.5, 0.5, 0.5, 0.5],
        histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.5],
    )
    views = [
        dict(i0=-12800, j0=-12800, i1=12800, j1=12800, stride=8, resolution_div=2, relief=0.6, **common_kwargs),
        dict(i0=0, j0=-3072, i1=8533, j1=-3072+8533, stride=2, resolution_div=1, relief=0.8, **common_kwargs),
        dict(i0=0, j0=768, i1=2844, j1=768+2844, stride=2, resolution_div=1, relief=1.0, **common_kwargs),
        dict(i0=0, j0=1024, i1=948, j1=1024+948, stride=1, resolution_div=1, relief=1.0, **common_kwargs),
    ]
    generate_zoom_sequence('world.h5', 1, views, 'world_combined.png', device='cuda')