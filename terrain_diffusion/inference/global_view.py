import argparse
from tempfile import NamedTemporaryFile
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
from terrain_diffusion.inference.world_pipeline import WorldPipeline, plot_channels_slider
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


def view_world(hdf5_file: str, seed: int, coarse_i0: int, coarse_j0: int, coarse_i1: int, coarse_j1: int,
               device: str | None = None, stride=8, resolution_div=2, interpolation='bicubic', save_dir=None, **kwargs) -> None:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with WorldPipeline(hdf5_file, device=device, seed=seed, mode='r', **kwargs) as world:
        ci0, ci1 = coarse_i0, coarse_i1
        cj0, cj1 = coarse_j0, coarse_j1
        
        i0 = ci0 * 256
        i1 = ci1 * 256
        j0 = cj0 * 256
        j1 = cj1 * 256
        
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

                region = world.get_90(is_, js, is_ + h, js + w, with_climate=False)
                elev = region['elev'].detach().cpu().numpy()
                elev = np.sign(elev) * elev**2

                if stride > 1:
                    elev_t = torch.from_numpy(elev).to(torch.float32)[None, None]
                    elev_ds = torch.nn.functional.avg_pool2d(elev_t, kernel_size=stride, stride=stride)[0, 0].numpy()
                else:
                    elev_ds = elev

                full_elev[oi:oi + h_out, oj:oj + w_out] = elev_ds[:h_out, :w_out]
                pbar.update(h * w)

        img = get_relief_map(full_elev, None, None, None, resolution=90*stride/resolution_div)
        
        # Save at native array resolution; flip vertically to match origin='lower'
        from PIL import Image
        out_img = np.clip(img, 0.0, 1.0)
        out_u8 = (out_img[::-1] * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out_u8).save(save_dir or 'world.png')

if __name__ == '__main__':
    # 
    # view_world('temp.h5', 854, device='cuda', coarse_window=64, stride=16,
    #                   frequency_mult=[1.0, 1.0, 1.0, 1.0, 1.0],
    #                   cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0],
    #                   log_mode='debug')
    view_world('world_big.h5', 1, -75, -75, 75, 75, device='cuda', stride=8, resolution_div=2,
               save_dir='world_0.png',
                drop_water_pct=0.0,
                frequency_mult=[0.7, 0.7, 0.7, 0.7, 0.7],
                cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0],)
    view_world('world_big.h5', 1, 0, -20, 50, 30, device='cuda', stride=2, resolution_div=1,
               save_dir='world_1.png',
               interpolation='nearest',
               drop_water_pct=0.0,
               frequency_mult=[0.7, 0.7, 0.7, 0.7, 0.7],
               cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0],)
    view_world('world_big.h5', 1, 0, 3, 17, 20, device='cuda', stride=2, resolution_div=1,
           save_dir='world_2.png',
           interpolation='nearest',
           drop_water_pct=0.0,
           frequency_mult=[0.7, 0.7, 0.7, 0.7, 0.7],
           cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0],)
    view_world('world_big.h5', 1, 0, 3, 6, 9, device='cuda', stride=1, resolution_div=1,
           save_dir='world_3.png',
           interpolation='nearest',
           drop_water_pct=0.0,
           frequency_mult=[0.7, 0.7, 0.7, 0.7, 0.7],
           cond_snr=[1.0, 1.0, 1.0, 1.0, 1.0],)