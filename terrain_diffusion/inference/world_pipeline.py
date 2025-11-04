import random
from tempfile import NamedTemporaryFile
import time

import torch
from infinite_tensor import HDF5TileStore, TensorWindow, MemoryTileStore
from synthetic_map import make_synthetic_map_factory
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.models.edm_unet import EDMUnet2D
import numpy as np
from typing import Union
from terrain_diffusion.models.mp_layers import mp_concat
from terrain_diffusion.models.perceptron import Perceptron
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
import matplotlib.pyplot as plt
import skimage
from terrain_diffusion.inference.postprocessing import *
    
def gaussian_noise_patch(
    base_seed: int,
    y0: int,
    x0: int,
    h: int,
    w: int,
    channels: int = 1,
    tile_h: int = 256,
    tile_w: int = 256,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Returns a (C, H, W) patch from an infinite, tile-seeded Gaussian noise field.
    Tiles are seeded deterministically from (base_seed, tile_y, tile_x).
    Coordinates are integer pixel indices; supports negative coordinates.
    """
    out = np.empty((channels, h, w), dtype=dtype)

    ty0 = (y0) // tile_h
    ty1 = (y0 + h - 1) // tile_h
    tx0 = (x0) // tile_w
    tx1 = (x0 + w - 1) // tile_w

    for ty in range(ty0, ty1 + 1):
        tile_y0 = ty * tile_h
        for tx in range(tx0, tx1 + 1):
            tile_x0 = tx * tile_w

            oy0 = max(y0, tile_y0)
            oy1 = min(y0 + h, tile_y0 + tile_h)
            ox0 = max(x0, tile_x0)
            ox1 = min(x0 + w, tile_x0 + tile_w)

            out_y0 = oy0 - y0
            out_y1 = oy1 - y0
            out_x0 = ox0 - x0
            out_x1 = ox1 - x0

            tile_y_off0 = oy0 - tile_y0
            tile_y_off1 = oy1 - tile_y0
            tile_x_off0 = ox0 - tile_x0
            tile_x_off1 = ox1 - tile_x0

            # Deterministic RNG per tile from (base_seed, ty, tx)
            words = np.array([base_seed, ty, tx]).astype(np.uint32)
            ss = np.random.SeedSequence(words)            
            rng = np.random.Generator(np.random.PCG64DXSM(ss))

            # Generate full tile (C, tile_h, tile_w) to keep tile content invariant across requests
            tile = rng.standard_normal(size=(channels, tile_h, tile_w), dtype=dtype)

            out[:, out_y0:out_y1, out_x0:out_x1] = tile[:, tile_y_off0:tile_y_off1, tile_x_off0:tile_x_off1]

    return out    

def linear_weight_window(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    s = size
    mid = (s - 1) / 2
    y, x = torch.meshgrid(torch.arange(s, device=device), torch.arange(s, device=device), indexing='ij')
    eps = 1e-3
    wy = 1 - (1 - eps) * torch.clamp(torch.abs(y - mid).to(dtype) / mid, 0, 1)
    wx = 1 - (1 - eps) * torch.clamp(torch.abs(x - mid).to(dtype) / mid, 0, 1)
    return (wy * wx)

def normalize_infinite_tensor(tile_store, tensor_id, tensor, tile_size, dim=0):
    tensor_shape_list = list(tensor.shape)
    tensor_shape_list[dim] -= 1
    tensor_shape = tuple(tensor_shape_list)
    
    
    out_window_size = tuple(x if x is not None else tile_size for x in tensor_shape)
    tensor_out_window = TensorWindow(size=out_window_size, stride=out_window_size)
    
    in_window_size = tuple(x if x is not None else tile_size for x in tensor.shape)
    tensor_in_window = TensorWindow(size=in_window_size, stride=in_window_size)

    def _normalize_fn(ctx, t):
        idx_num = [slice(None)] * t.ndim
        idx_den = [slice(None)] * t.ndim
        idx_num[dim] = slice(None, -1)
        idx_den[dim] = slice(-1, None)
        return t[tuple(idx_num)] / t[tuple(idx_den)]

    return tile_store.get_or_create(tensor_id,
                                    shape=tensor_shape,
                                    f=_normalize_fn,
                                    output_window=tensor_out_window,
                                    args=(tensor,),
                                    args_windows=(tensor_in_window,))
    
def plot_channels_slider(data, cmap=None, channel_names=None):
    """
    Plot either:
      - a single multi-channel tensor/array shaped (C, H, W) with a channel slider, or
      - a list of tensors/arrays where each item may be:
          (H, W), (C, H, W), or RGB in either (3, H, W) or (H, W, 3).

    For a list input, a single index slider is provided to flip through views:
      - grayscale items (H, W) are single views
      - RGB items are single views
      - non-RGB multi-channel items produce one view per channel

    Returns (fig, ax, slider).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.array(x)

    # Backward-compatible behavior for a single (C, H, W) tensor/array
    if not isinstance(data, (list, tuple)):
        arr = to_numpy(data)
        if arr.ndim != 3:
            raise ValueError("Expected data of shape (C, H, W) for single input.")

        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.25)
        curr_chan = 0
        im = ax.imshow(arr[curr_chan], cmap=cmap)
        cbar = plt.colorbar(im, ax=ax)

        def _title_for(chan: int) -> str:
            if channel_names and 0 <= chan < len(channel_names):
                return f"{channel_names[chan]} (channel {chan})"
            return f"Channel {chan}"

        ax.set_title(_title_for(curr_chan))

        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
        slider = Slider(ax=ax_slider, label='Channel', valmin=0, valmax=arr.shape[0] - 1, valinit=curr_chan, valstep=1)

        def update(_):
            chan = int(slider.val)
            im.set_data(arr[chan])
            ax.set_title(_title_for(chan))
            im.autoscale()
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
        return fig, ax, slider

    # List/tuple input: build a flat list of views to browse
    views = []          # each element is either (H, W) or (H, W, 3)
    view_titles = []    # title per view

    for idx, item in enumerate(data):
        arr = to_numpy(item)

        # Normalize shapes and expand into views
        if arr.ndim == 2:
            views.append(arr)
            view_titles.append(f"Item {idx}")
        elif arr.ndim == 3:
            # Detect HWC RGB
            if arr.shape[-1] == 3 and arr.shape[0] != 3:
                views.append(arr)
                view_titles.append(f"Item {idx} (RGB)")
            # Detect CHW
            elif arr.shape[0] == 3 and arr.shape[-1] != 3:
                views.append(np.moveaxis(arr, 0, -1))
                view_titles.append(f"Item {idx} (RGB)")
            else:
                # Treat as (C, H, W) multi-channel (grayscale per channel)
                # Fall back to HWC multi-channel if that seems more plausible
                if arr.shape[0] <= arr.shape[-1]:
                    # assume CHW
                    for c in range(arr.shape[0]):
                        views.append(arr[c])
                        view_titles.append(f"Item {idx} ch {c}")
                else:
                    # assume HWC non-RGB: split channels
                    for c in range(arr.shape[-1]):
                        views.append(arr[..., c])
                        view_titles.append(f"Item {idx} ch {c}")
        else:
            raise ValueError("Unsupported item dimensionality; expected 2D or 3D arrays/tensors.")

    if not views:
        raise ValueError("No views to display from the provided list.")

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    curr_idx = 0

    # Initial draw
    first = views[curr_idx]
    is_rgb = (first.ndim == 3 and first.shape[-1] == 3)
    im = ax.imshow(first, cmap=None if is_rgb else cmap)
    cbar = None
    if not is_rgb:
        cbar = plt.colorbar(im, ax=ax)

    def _title_for(idx: int) -> str:
        if channel_names and 0 <= idx < len(channel_names):
            return channel_names[idx]
        return view_titles[idx] if 0 <= idx < len(view_titles) else f"Index {idx}"

    ax.set_title(_title_for(curr_idx))

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax=ax_slider, label='Index', valmin=0, valmax=len(views) - 1, valinit=curr_idx, valstep=1)

    def update(_):
        nonlocal cbar
        idx = int(slider.val)
        img = views[idx]
        is_rgb_local = (img.ndim == 3 and img.shape[-1] == 3)
        im.set_data(img)
        if is_rgb_local:
            im.set_cmap(None)
            if cbar is not None:
                cbar.remove()
                cbar = None
        else:
            im.set_cmap(cmap)
            if cbar is None:
                cbar = plt.colorbar(im, ax=ax)
            im.autoscale()
        ax.set_title(_title_for(idx))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return fig, ax, slider
    
class WorldPipeline:
    def __init__(self, 
                 hdf5_file, 
                 mode='a', 
                 compression: str | None = "gzip", 
                 compression_opts: int | None = 4,
                 seed=None,
                 device='cpu',
                 coarse_model="checkpoints/models/diffusion_coarse",
                 base_model="checkpoints/models/consistency_base-192x3",
                 decoder_model="checkpoints/models/consistency_decoder-64x3",
                 biome_classifier_model="checkpoints/models/biome_classifier",
                 **kwargs):
        self.kwargs = kwargs
        self.device = device
        self.tile_store = HDF5TileStore(hdf5_file, mode=mode, compression=compression, compression_opts=compression_opts)
        self.seed = seed or random.randint(0, 2**31-1)
        
        self.log_mode = kwargs.get('log_mode', 'info')
        
        self.synthetic_map_factory = make_synthetic_map_factory(seed=self.seed, frequency_mult=self.kwargs.get('frequency_mult', [1.5, 3, 3, 3, 3]), drop_water_pct=0.5)
        self.biome_classifier = Perceptron.from_pretrained(biome_classifier_model)
        
        # Create coarse perlin
        self.coarse_model = EDMUnet2D.from_pretrained(coarse_model).to(device)
        self.coarse = self._get_coarse_map()
        
        self.base_model = EDMUnet2D.from_pretrained(base_model).to(device)
        self.latents = self._get_latent_map()
        
        self.decoder_model = EDMUnet2D.from_pretrained(decoder_model).to(device)
        self.residual_90 = self._get_residual_90_map()
        
    def _get_coarse_map(self):
        TILE_SIZE = 64
        TILE_STRIDE = TILE_SIZE//2
        MODEL_MEANS = torch.tensor([-37.67916460232751, 2.22578822145657, 18.030293275011356, 333.8442390481231, 1350.1259248456176, 52.444339366764396])
        MODEL_STDS = torch.tensor([39.68515115440358, 3.0981253981231522, 8.940333096712806, 322.25238547630295, 856.3430083394657, 30.982620765341043])
        COND_SNR = torch.tensor(self.kwargs.get('cond_snr', [0.3, 0.1, 1.0, 0.1, 1.0]))
        
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
        weight_window = linear_weight_window(TILE_SIZE, 'cpu', torch.float32)
        
        t_cond = torch.atan(COND_SNR).to(self.device)
        vals = torch.log(torch.tan(t_cond) / 8.0)
        cond_inputs = [v.detach().view(-1) for v in vals]
        
        def f(ctx):
            _, i, j = ctx
            i1, j1 = i * TILE_STRIDE, j * TILE_STRIDE
            i2, j2 = i1 + TILE_SIZE, j1 + TILE_SIZE
            synthetic_map = torch.as_tensor(self.synthetic_map_factory(j1, i1, j2, i2))
            synthetic_map = (synthetic_map - MODEL_MEANS[[0, 2, 3, 4, 5], None, None]) / MODEL_STDS[[0, 2, 3, 4, 5], None, None]
            synthetic_map = synthetic_map.to(self.device)[None]
            
            cond_img = torch.cos(t_cond.view(1, -1, 1, 1)) * synthetic_map + torch.sin(t_cond.view(1, -1, 1, 1)) * torch.randn_like(synthetic_map)
            
            scheduler.set_timesteps(20)
            sample = torch.randn(1, 6, TILE_SIZE, TILE_SIZE, device=self.device) * scheduler.sigmas[0]
            for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                t = t.to(self.device)
                sigma = sigma.to(self.device)
                scaled_in = scheduler.precondition_inputs(sample, sigma)
                cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1))

                x_in = torch.cat([scaled_in, cond_img], dim=1)
                model_out = self.coarse_model(x_in, noise_labels=cnoise, conditional_inputs=cond_inputs)
                sample = scheduler.step(model_out, t, sample).prev_sample
                
            sample = sample.cpu() / scheduler.config.sigma_data
            sample = (sample * MODEL_STDS.view(1, -1, 1, 1)) + MODEL_MEANS.view(1, -1, 1, 1)
            sample[0, 1] = sample[0, 0] - sample[0, 1]
            output = torch.cat([sample[0] * weight_window[None], weight_window[None]], dim=0)
            return output
                    
        output_window = TensorWindow(size=(7, TILE_SIZE, TILE_SIZE), stride=(7, TILE_STRIDE, TILE_STRIDE))
        tensor = self.tile_store.get_or_create("base_coarse_map",
                                               shape=(7, None, None),
                                               f=f,
                                               output_window=output_window)
        
        norm_tensor = normalize_infinite_tensor(self.tile_store, "coarse_map", tensor, TILE_SIZE)
        tensor.mark_for_cleanup()
        return norm_tensor
        
    def _get_latent_map(self):
        TILE_SIZE = 64
        TILE_STRIDE = TILE_SIZE//2
        COND_INPUT_MEAN = torch.tensor([14.99, 11.65, 15.87, 619.26, 833.12, 69.40, 0.66])
        COND_INPUT_STD = torch.tensor([21.72, 21.78, 10.40, 452.29, 738.09, 34.59, 0.47])
        HISTOGRAM_RAW = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        COND_MAX_NOISE = 0.1
        NOISE_LEVEL = 0.0
        
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
        weight_window = linear_weight_window(TILE_SIZE, 'cpu', torch.float32)
        
        def _process_cond_img(
            cond_img: torch.Tensor, 
            histogram_raw: torch.Tensor, 
            cond_means: torch.Tensor, 
            cond_stds: torch.Tensor,
            noise_level: torch.Tensor = torch.tensor(0.0)
        ) -> torch.Tensor:
            """
            Process the conditioning image and compute the final conditioning tensor.

            Args:
                cond_img (torch.Tensor): Conditioning image tensor of shape (B, C, 4, 4). Unnormalized.
                    Channels are: means (signed-sqrt), p5 (signed-sqrt), temp mean (C), temp std (C), precip mean (mm/yr), precip std (coeff of var)
                histogram_raw (torch.Tensor): Raw histogram (pre-softmax values) features to include in the conditioning vector. Length equal to the number of subsets trained on.
                cond_means (torch.Tensor): Array or tensor with means for normalization.
                cond_stds (torch.Tensor): Array or tensor with stds for normalization.
                noise_level (float): Noise level (0-1) to apply to the conditioning tensor.

            Returns:
                torch.Tensor: Processed conditioning tensor to be passed into the model.
            """
            cond_img[0:1] = cond_img[0:1] + torch.randn_like(cond_img[:, 0:1]) * noise_level * COND_MAX_NOISE
            cond_img[1:2] = cond_img[1:2] + torch.randn_like(cond_img[:, 1:2]) * noise_level * COND_MAX_NOISE
            cond_img = (cond_img - torch.tensor(cond_means, device=cond_img.device).view(1, -1, 1, 1)) / torch.tensor(cond_stds, device=cond_img.device).view(1, -1, 1, 1)
            
            cond_img[0:1] = cond_img[0:1].nan_to_num(cond_means[0])
            cond_img[1:2] = cond_img[1:2].nan_to_num(cond_means[1])
            
            means_crop = cond_img[:, 0:1]
            p5_crop = cond_img[:, 1:2]
            climate_means_crop = cond_img[:, 2:6, 1:3, 1:3].mean(dim=(2, 3))
            mask_crop = cond_img[:, 6:7]
            
            climate_means_crop[torch.isnan(climate_means_crop)] = torch.randn_like(climate_means_crop[torch.isnan(climate_means_crop)])
            
            noise_level = (noise_level - 0.5) * np.sqrt(12)
            return mp_concat([means_crop.flatten(1), p5_crop.flatten(1), climate_means_crop.flatten(1), mask_crop.flatten(1), histogram_raw, noise_level.view(-1, 1)], dim=1).float()

        def f(ctx, sample, cond_img, t, seed_offset=0):
            if self.log_mode == 'debug':
                print(f"Latent f at {ctx}")
            if sample is None:
                sample = torch.zeros((1, 5, TILE_SIZE, TILE_SIZE), device=self.device)
            else:
                sample = torch.as_tensor(sample, device=self.device)
                sample = sample[:-1] / sample[-1:] * scheduler.config.sigma_data
            
            mask = torch.ones((1, 4, 4))
            cond_img = torch.cat([cond_img, mask], dim=0)[None]
            
            t = torch.as_tensor(t, device=self.device)
            cond_inputs = _process_cond_img(cond_img, HISTOGRAM_RAW, COND_INPUT_MEAN, COND_INPUT_STD, noise_level=torch.tensor(NOISE_LEVEL)).to(self.device)
            
            noise = torch.as_tensor(gaussian_noise_patch(self.seed + seed_offset, ctx[1]*TILE_STRIDE, ctx[2]*TILE_STRIDE, TILE_SIZE, TILE_SIZE, 
                                                         channels=5, tile_h=TILE_SIZE, tile_w=TILE_SIZE))[None]
            
            t = t.view(1, 1, 1, 1).to(self.device)
            z = noise.to(self.device) * scheduler.config.sigma_data
            x_t = torch.cos(t) * sample + torch.sin(t) * z
            model_in = (x_t / scheduler.config.sigma_data).to(self.device)
            pred = -self.base_model(model_in, noise_labels=t.flatten(), conditional_inputs=[cond_inputs])
            sample = torch.cos(t) * x_t - torch.sin(t) * scheduler.config.sigma_data * pred
            sample = sample.cpu() / scheduler.config.sigma_data
            return torch.cat([sample[0] * weight_window[None], weight_window[None]], dim=0)

        t_init = torch.atan(scheduler.sigmas[0] / scheduler.config.sigma_data)
        output_window = TensorWindow(size=(6, TILE_SIZE, TILE_SIZE), stride=(6, TILE_STRIDE, TILE_STRIDE))
        coarse_window = TensorWindow(size=(6, 4, 4), stride=(6, 1, 1))
        tensor = self.tile_store.get_or_create("init_latent_map",
                                               shape=(6, None, None),
                                               f=lambda ctx, cond: f(ctx, None, cond, t_init, seed_offset=5819),
                                               output_window=output_window,
                                               args=(self.coarse,),
                                               args_windows=(coarse_window,))
        
        inter_t = [torch.arctan(torch.tensor(0.06) / 0.5)]
        #inter_t = []
        for i, t in enumerate(inter_t):
            next_tensor = self.tile_store.get_or_create("step_latent_map_{}".format(i),
                                                shape=(6, None, None),
                                                f=lambda ctx, sample, cond: f(ctx, sample, cond, t, seed_offset=5820+i),
                                                output_window=output_window,
                                                args=(tensor, self.coarse,),
                                                args_windows=(output_window, coarse_window,))
            tensor.mark_for_cleanup()
            tensor = next_tensor
        
        norm_tensor = normalize_infinite_tensor(self.tile_store, "latent_map", tensor, TILE_SIZE)
        tensor.mark_for_cleanup()
        return norm_tensor
    
    def _get_residual_90_map(self):
        TILE_SIZE = 512
        TILE_STRIDE = TILE_SIZE//2
        
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
        weight_window = linear_weight_window(TILE_SIZE, 'cpu', torch.float32)
        
        t_list = [torch.atan(scheduler.sigmas[0] / scheduler.config.sigma_data)]
        #t_list += [torch.arctan(torch.tensor(0.075) / 0.5)]
        def f(ctx, latents):
            if self.log_mode == 'debug':
                print(f"Residual f at {ctx}")
            sample = torch.zeros((1, 1, TILE_SIZE, TILE_SIZE), device=self.device)
            latents = latents[:4].view(1, 4, TILE_SIZE//8, TILE_SIZE//8)
            upsampled_latents = torch.nn.functional.interpolate(latents, size=(TILE_SIZE, TILE_SIZE), mode='nearest').to(self.device)
            
            for i, t in enumerate(t_list):
                noise = torch.as_tensor(gaussian_noise_patch(self.seed + 5819 + i, ctx[1]*TILE_STRIDE, ctx[2]*TILE_STRIDE, TILE_SIZE, TILE_SIZE, 
                                                            channels=1, tile_h=TILE_SIZE, tile_w=TILE_SIZE))[None]
                t = t.view(1, 1, 1, 1).to(self.device)
                z = noise.to(self.device) * scheduler.config.sigma_data
                x_t = torch.cos(t) * sample + torch.sin(t) * z
                model_in = (x_t / scheduler.config.sigma_data).to(self.device)
                model_in = torch.cat([model_in, upsampled_latents], dim=1)
                pred = -self.decoder_model(model_in, noise_labels=t.flatten(), conditional_inputs=[])
                sample = torch.cos(t) * x_t - torch.sin(t) * scheduler.config.sigma_data * pred
                
            sample = sample.cpu() / scheduler.config.sigma_data
            return torch.cat([sample[0] * weight_window[None], weight_window[None]], dim=0)
        
        output_window = TensorWindow(size=(2, TILE_SIZE, TILE_SIZE), stride=(2, TILE_STRIDE, TILE_STRIDE))
        input_window = TensorWindow(size=(5, TILE_SIZE//8, TILE_SIZE//8), stride=(5, TILE_STRIDE//8, TILE_STRIDE//8))
        
        tensor = self.tile_store.get_or_create("init_residual_90_map",
                                               shape=(2, None, None),
                                               f=f,
                                               output_window=output_window,
                                               args=(self.latents,),
                                               args_windows=(input_window,))
            
        norm_tensor = normalize_infinite_tensor(self.tile_store, "residual_90_map", tensor, TILE_SIZE)
        tensor.mark_for_cleanup()
        return norm_tensor
    
    def get_90(self, i1, j1, i2, j2):
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        RESIDUAL_MEAN = 0.0
        RESIDUAL_STD = 1.1678
        
        # Elevation at 90m
        residual = self.residual_90[0, i1:i2, j1:j2] * RESIDUAL_STD + RESIDUAL_MEAN
        lowfreq = self.latents[4, i1//8:i2//8, j1//8:j2//8] * LOWFREQ_STD + LOWFREQ_MEAN
        residual, lowfreq = laplacian_denoise(residual, lowfreq, sigma=5)
        elev = laplacian_decode(residual, lowfreq)

        # Climate from coarse map (channels 2:6 are: temp_mean, temp_std, precip_mean, precip_std)
        # Coarse resolution is 256 x 90m per pixel. To support partial pixels, we
        # load an expanded coarse crop, upsample by 256x (nearest), then crop.
        def ceil_div(a: int, b: int) -> int:
            return -((-a) // b)

        ci1 = i1 // 256
        cj1 = j1 // 256
        ci2 = ceil_div(i2, 256)
        cj2 = ceil_div(j2, 256)

        # Slice climate channels at coarse resolution
        coarse_window_size = 15
        coarse_padding = (coarse_window_size - 1) // 2 + 1
        # elev, p5, temp_mean, temp_std, precip_mean, precip_std, mask
        coarse_map = self.coarse[:, ci1-coarse_padding:ci2+coarse_padding, cj1-coarse_padding:cj2+coarse_padding]
        coarse_elev_denorm = torch.sign(coarse_map[0]) * torch.square(torch.maximum(torch.zeros_like(coarse_map[0]), coarse_map[0]))
        temp_baseline, beta = local_baseline_temperature_torch(coarse_map[2], coarse_elev_denorm, win=coarse_window_size, fallback_threshold=0.02)
        
        # Upsample to 90m grid
        coarse_up = torch.nn.functional.interpolate(coarse_map[None], scale_factor=256, mode='bicubic')[0]
        temp_baseline_up = torch.nn.functional.interpolate(temp_baseline[None], scale_factor=256, mode='bicubic')[0]
        beta_up = torch.nn.functional.interpolate(beta[None], scale_factor=256, mode='bicubic')[0]
        
        temp_baseline_up = temp_baseline_up[:, 256:-256, 256:-256]
        beta_up = beta_up[:, 256:-256, 256:-256]

        # Crop to requested region (account for coarse padding in upsampled grid)
        oi1 = i1 - (ci1 - coarse_padding) * 256
        oj1 = j1 - (cj1 - coarse_padding) * 256
        oi2 = oi1 + (i2 - i1)
        oj2 = oj1 + (j2 - j1)
        coarse_up = coarse_up[:, oi1:oi2, oj1:oj2]
        
        temp_realistic = temp_baseline_up[0] + beta_up[0] * torch.square(torch.maximum(elev, torch.zeros_like(elev)))
        climate = torch.stack([temp_realistic, coarse_up[3], coarse_up[4], coarse_up[5], beta_up[0]])
        
        biome_in_mean = torch.tensor([687.1005, 14.652394, 688.7267, 795.926, 66.63564])
        biome_in_std = torch.tensor([875.3516, 10.713884, 461.52255, 730.29474, 35.284668])
        biome_in_img = torch.cat([elev[None], climate[0:4]], dim=0).permute(1, 2, 0).view(-1, 5)
        biome_map = self.biome_classifier((biome_in_img - biome_in_mean.view(1, 5)) / biome_in_std.view(1, 5))
        #biome_map = torch.argmax(biome_map, dim=1).reshape(i2 - i1, j2 - j1)
        
        temp = 0.3
        idx = torch.distributions.Categorical(logits=biome_map / temp).sample()
        biome_map = idx.reshape(i2 - i1, j2 - j1)

        return {
            'elev': elev,
            'climate': climate,
            'biome': biome_map,
        }
    
#if __name__ == '__main__':
#    
#    
#    with NamedTemporaryFile(suffix='.h5') as tmp_file:
#        # prev = 
#        x = WorldPipeline('temp.h5', device='cuda', seed=854)
#    
#        start_time = time.time()
#        elev, climate = x.get_90(-128*24, -128*0, -128*8, 128*16)
#        adj = np.maximum(elev.numpy(), 0)
#        adj[adj == 0] = np.nan
#        end_time = time.time()
#        print(f"Time taken: {end_time - start_time} seconds")
#        adj = adj**2
#        
#        #adj = smooth_river_bumps(adj)
#        adj_new = fill_depressions_priority_flood(adj, epsilon=0.03, max_raise=2000)
#        #adj_new = adj
#        adj_diff = adj_new - adj
#        
#        flow = plot_flow_indicator(adj_new, max_pool_kernel=4)
#        
#        full_map = torch.cat([torch.as_tensor(adj_new)[None], climate], dim=0)
#        
#        pred_temp = (climate[0] + climate[4] * torch.as_tensor(adj_new))[None]
#        full_map = torch.cat([torch.as_tensor(adj_new)[None], pred_temp, climate[1:]], dim=0)
#        
#        biome_in_mean = [687.1005, 14.652394, 688.7267, 795.926, 66.63564]
#        biome_in_std = [875.3516, 10.713884, 461.52255, 730.29474, 35.284668]
#        
#        orig_size = full_map.shape[1:]
#        biome_map = biome_class((full_map.permute(1, 2, 0)[:, :, :5].view(-1, 5) - torch.tensor(biome_in_mean)[None]) / torch.tensor(biome_in_std)[None])
#        biome_map = torch.argmax(biome_map, dim=1).reshape(*orig_size)
#        
#        # INSERT_YOUR_CODE
#        # Convert biome_map to RGB
#        biome_rgb_lut = np.array([
#            [0, 0, 0],         # index 0 (unused, possibly "background" or "no class")
#            [0, 0, 255],       # 1:  Af   Tropical, rainforest
#            [0, 120, 255],     # 2:  Am   Tropical, monsoon
#            [70, 170, 250],    # 3:  Aw   Tropical, savannah
#            [255, 0, 0],       # 4:  BWh  Arid, desert, hot
#            [255, 150, 150],   # 5:  BWk  Arid, desert, cold
#            [245, 165, 0],     # 6:  BSh  Arid, steppe, hot
#            [255, 220, 100],   # 7:  BSk  Arid, steppe, cold
#            [255, 255, 0],     # 8:  Csa  Temperate, dry summer, hot summer
#            [200, 200, 0],     # 9:  Csb  Temperate, dry summer, warm summer
#            [150, 150, 0],     #10:  Csc  Temperate, dry summer, cold summer
#            [150, 255, 150],   #11:  Cwa  Temperate, dry winter, hot summer
#            [100, 200, 100],   #12:  Cwb  Temperate, dry winter, warm summer
#            [50, 150, 50],     #13:  Cwc  Temperate, dry winter, cold summer
#            [200, 255, 80],    #14:  Cfa  Temperate, no dry season, hot summer
#            [100, 255, 80],    #15:  Cfb  Temperate, no dry season, warm summer
#            [50, 200, 0],      #16:  Cfc  Temperate, no dry season, cold summer
#            [255, 0, 255],     #17:  Dsa  Cold, dry summer, hot summer
#            [200, 0, 200],     #18:  Dsb  Cold, dry summer, warm summer
#            [150, 50, 150],    #19:  Dsc  Cold, dry summer, cold summer
#            [150, 100, 150],   #20:  Dsd  Cold, dry summer, very cold winter
#            [170, 175, 255],   #21:  Dwa  Cold, dry winter, hot summer
#            [90, 120, 220],    #22:  Dwb  Cold, dry winter, warm summer
#            [75, 80, 180],     #23:  Dwc  Cold, dry winter, cold summer
#            [50, 0, 135],      #24:  Dwd  Cold, dry winter, very cold winter
#            [0, 255, 255],     #25:  Dfa  Cold, no dry season, hot summer
#            [55, 200, 255],    #26:  Dfb  Cold, no dry season, warm summer
#            [0, 125, 125],     #27:  Dfc  Cold, no dry season, cold summer
#            [0, 70, 95],       #28:  Dfd  Cold, no dry season, very cold winter
#            [178, 178, 178],   #29:  ET   Polar, tundra
#            [102, 102, 102],   #30:  EF   Polar, frost
#        ], dtype=np.uint8)
#        
#        # biome_map values are indices 0-30, map to RGB
#        biome_map_np = biome_map.cpu().numpy()
#        biome_rgb = biome_rgb_lut[biome_map_np]
#        
#        basic_climate = torch.cat([pred_temp, climate[1:4]], dim=0)
#        flow_up = skimage.transform.resize(flow, (elev.shape[0], elev.shape[1]), order=0)
#        rm = get_relief_map(elev, basic_climate, biome_map, flow_up, flow_threshold=10)
#        
#        # Plot with a slider to switch channels
#        plot_channels_slider([flow] + [x for x in full_map] + [biome_rgb, adj_diff, rm])
#        