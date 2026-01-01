import random
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
import json
import os
import atexit

import h5py
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from infinite_tensor import HDF5TileStore, TensorWindow, MemoryTileStore
from terrain_diffusion.inference.synthetic_map import make_synthetic_map_factory
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.common.model_utils import resolve_model_path, MODEL_PATHS
import numpy as np
from typing import Union
from terrain_diffusion.models.mp_layers import mp_concat
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
import matplotlib.pyplot as plt
import skimage
from terrain_diffusion.inference.postprocessing import *

_TEMP_FILES = set()


def resolve_hdf5_path(hdf5_file: str) -> str:
    """Resolve 'TEMP' to a temporary file path, otherwise return the path as-is."""
    if hdf5_file.upper() == 'TEMP':
        temp_file = NamedTemporaryFile(delete=False, suffix='.h5', prefix='terrain_')
        temp_path = temp_file.name
        temp_file.close()
        _TEMP_FILES.add(temp_path)
        return temp_path
    return hdf5_file


def cleanup_temp_file(temp_path: str):
    """Clean up a specific temporary file."""
    if temp_path in _TEMP_FILES:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            _TEMP_FILES.discard(temp_path)
        except Exception:
            pass


def _cleanup_all_temp_files():
    """Clean up all temporary files created via resolve_hdf5_path."""
    for temp_path in list(_TEMP_FILES):
        cleanup_temp_file(temp_path)


atexit.register(_cleanup_all_temp_files)
    
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
    
MOCK = False

def normalize_tensor(tensor, dim=0):
    idx_num = [slice(None)] * tensor.ndim
    idx_den = [slice(None)] * tensor.ndim
    idx_num[dim] = slice(None, -1)
    idx_den[dim] = slice(-1, None)
    return tensor[tuple(idx_num)] / tensor[tuple(idx_den)]


class WorldPipeline(ConfigMixin):
    """Multi-scale terrain generation pipeline."""
    
    config_name = "config.json"
    
    # Subfolders for each model when saving/loading
    COARSE_MODEL_FOLDER = "coarse_model"
    BASE_MODEL_FOLDER = "base_model"
    DECODER_MODEL_FOLDER = "decoder_model"
    
    ignore_for_config = ["seed", "latents_batch_size", "log_mode", "cache_limit", "caching_strategy", "torch_compile", "dtype"]
    
    @register_to_config
    def __init__(
        self,
        seed: int | None = None,
        latents_batch_size: int | list[int] = 1,
        native_resolution: float = 90.0,
        *,
        log_mode: str = 'info',
        torch_compile: bool = False,
        dtype: str | None = None,
        latent_compression: int = 8,
        frequency_mult: list = None,
        drop_water_pct: float = 0.5,
        cond_snr: list = None,
        coarse_pooling: int = 1,
        elev_coarse_pool_mode: str = 'avg',
        p5_coarse_pool_mode: str = 'avg',
        residual_mean: float = 0.0,
        residual_std: float = 1.1678,
        coarse_means: list = None,
        coarse_stds: list = None,
        caching_strategy: str = 'indirect',
        cache_limit: int | None = 100 * 1024 * 1024,
        onestep_latent: bool = False,
        decoder_tile_size: int = 512,
        decoder_tile_stride: int = 384,
        **deprecated_kwargs,
    ):
        super().__init__()
        
        # Resolve seed
        self.seed = seed if seed is not None else random.randint(0, 2**31-1)
        # Normalize batch sizes to sorted list
        if isinstance(latents_batch_size, int):
            self._batch_sizes = [latents_batch_size]
        else:
            self._batch_sizes = sorted(latents_batch_size)
        self.latents_batch_size = self._batch_sizes[-1]  # max batch size for tile_store
        self.native_resolution = native_resolution
        self.latent_compression = latent_compression
        self.log_mode = log_mode
        self.torch_compile = torch_compile
        
        if self.torch_compile and os.name == 'nt':
            print("WARNING: torch.compile is not currently supported on Windows. This will affect performance.")
            self.torch_compile = False
            
        self.caching_strategy = caching_strategy
        self.cache_limit = cache_limit
        self.onestep_latent = onestep_latent
        self.decoder_tile_size = decoder_tile_size
        self.decoder_tile_stride = decoder_tile_stride
        self.kwargs = {
            'latent_compression': latent_compression,
            'log_mode': log_mode,
            'frequency_mult': frequency_mult if frequency_mult is not None else [1.5, 3, 3, 3, 3],
            'drop_water_pct': drop_water_pct,
            'cond_snr': cond_snr if cond_snr is not None else [0.3, 0.1, 1.0, 0.1, 1.0],
            'coarse_pooling': coarse_pooling,
            'elev_coarse_pool_mode': elev_coarse_pool_mode,
            'p5_coarse_pool_mode': p5_coarse_pool_mode,
            'histogram_raw': deprecated_kwargs.get('histogram_raw', [0.0, 0.0, 0.0, 0.0, 0.0]),
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'coarse_means': coarse_means if coarse_means is not None else [-37.67916460232751, 2.22578822145657, 18.030293275011356, 333.8442390481231, 1350.1259248456176, 52.444339366764396],
            'coarse_stds': coarse_stds if coarse_stds is not None else [39.68515115440358, 3.0981253981231522, 8.940333096712806, 322.25238547630295, 856.3430083394657, 30.982620765341043],
        }
        
        # Convert dtype string to torch dtype
        if dtype == 'bf16':
            self._dtype = torch.bfloat16
        elif dtype == 'fp16':
            self._dtype = torch.float16
        else:
            self._dtype = None
        
        # Models as submodules - set via from_local_models() or from_pretrained()
        self.coarse_model: EDMUnet2D | None = None
        self.base_model: EDMUnet2D | None = None
        self.decoder_model: EDMUnet2D | None = None
        
        # Runtime state - initialized via bind()
        self.tile_store = None
        self._hdf5_file_path = None
        self._is_temp_file = False
        self.synthetic_map_factory = None
        self.coarse = None
        self.latents = None
        self.residual = None
    
    def _get_padded_batch_size(self, actual_size: int) -> int:
        """Get the next highest legal batch size for padding."""
        for bs in self._batch_sizes:
            if bs >= actual_size:
                return bs
        return self._batch_sizes[-1]  # fallback to max
    
    def _common_tile_kwargs(self) -> dict:
        """Get common kwargs for tile_store.get_or_create()."""
        if self.caching_strategy != 'direct':
            return {}
        kwargs = {'cache_method': 'direct'}
        if self.cache_limit is not None:
            kwargs['cache_limit'] = self.cache_limit
        return kwargs
    
    def _apply_dtype_and_compile(self):
        """Apply eval mode, dtype conversion and torch.compile to models."""
        models = [
            ('coarse_model', self.coarse_model),
            ('base_model', self.base_model),
            ('decoder_model', self.decoder_model),
        ]
        
        # Set eval mode
        for name, model in models:
            if model is not None:
                model.eval()
        
        # Apply dtype conversion
        if self._dtype is not None:
            dtype_name = {torch.bfloat16: 'bf16', torch.float16: 'fp16'}[self._dtype]
            print(f"Converting models to {dtype_name}...")
            for name, model in models:
                if model is not None:
                    model.to(self._dtype)
        
        # Apply torch.compile
        if self.torch_compile:
            print(f"Compiling models with torch.compile...")
            if self.coarse_model is not None:
                self.coarse_model = torch.compile(self.coarse_model)
            if self.base_model is not None:
                self.base_model = torch.compile(self.base_model)
            if self.decoder_model is not None:
                self.decoder_model = torch.compile(self.decoder_model)
            print("Model compilation complete.")
    
    def _warmup_compiled_models(self):
        """Warmup compiled models with all batch sizes."""
        if not self.torch_compile:
            return
        
        warmup_batch_sizes = self._batch_sizes
        print(f"Warming up compiled models with batch sizes: {warmup_batch_sizes}...")
        device = self.device
        dtype = self._dtype or torch.float32
        
        # Warmup coarse model (batch size 1)
        if self.coarse_model is not None:
            print("  Warming up coarse_model...")
            dummy_in = torch.randn(1, 11, 64, 64, device=device, dtype=dtype)
            dummy_noise = torch.randn(1, device=device, dtype=dtype)
            with torch.no_grad():
                self.coarse_model(dummy_in, noise_labels=dummy_noise, conditional_inputs=[torch.zeros(1, device=device, dtype=dtype) for _ in range(5)])
        
        # Warmup base model (latent stage) with each batch size
        # Conditioning size: 16 (means) + 16 (p5) + 4 (climate) + 16 (mask) + 5 (histogram) + 1 (noise_level) = 58
        if self.base_model is not None:
            for bs in warmup_batch_sizes:
                print(f"  Warming up base_model with batch size {bs}...")
                dummy_in = torch.randn(bs, 5, 64, 64, device=device, dtype=dtype)
                dummy_cond = torch.randn(bs, 58, device=device, dtype=dtype)
                dummy_noise = torch.randn(bs, device=device, dtype=dtype)
                with torch.no_grad():
                    self.base_model(dummy_in, noise_labels=dummy_noise, conditional_inputs=[dummy_cond])
        
        # Warmup decoder model (batch size 1)
        if self.decoder_model is not None:
            print("  Warming up decoder_model...")
            dummy_in = torch.randn(1, 5, 512, 512, device=device, dtype=dtype)
            dummy_noise = torch.randn(1, device=device, dtype=dtype)
            with torch.no_grad():
                self.decoder_model(dummy_in, noise_labels=dummy_noise, conditional_inputs=[])
        
        print("Warmup complete.")
    
    @classmethod
    def from_local_models(
        cls,
        coarse_model_path: str | None = None,
        base_model_path: str | None = None,
        decoder_model_path: str | None = None,
        **kwargs,
    ) -> "WorldPipeline":
        """Create pipeline loading models from local paths.
        
        Args:
            coarse_model_path: Path to coarse model, or None for default.
            base_model_path: Path to base model, or None for default.
            decoder_model_path: Path to decoder model, or None for default.
            **kwargs: Additional arguments passed to __init__.
        """
        pipeline = cls(**kwargs)
        pipeline.coarse_model = EDMUnet2D.from_pretrained(
            resolve_model_path(coarse_model_path, *MODEL_PATHS["coarse"])
        )
        pipeline.base_model = EDMUnet2D.from_pretrained(
            resolve_model_path(base_model_path, *MODEL_PATHS["base"])
        )
        pipeline.decoder_model = EDMUnet2D.from_pretrained(
            resolve_model_path(decoder_model_path, *MODEL_PATHS["decoder"])
        )
        pipeline._apply_dtype_and_compile()
        return pipeline
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save pipeline config and all submodels to a directory.
        
        Args:
            save_directory: Directory to save to.
            **kwargs: Additional arguments passed to submodel save_pretrained.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save pipeline config only (no weights for the pipeline itself)
        self.save_config(save_directory)
        
        # Save each submodel in its own subfolder
        if self.coarse_model is not None:
            self.coarse_model.save_pretrained(os.path.join(save_directory, self.COARSE_MODEL_FOLDER), **kwargs)
        if self.base_model is not None:
            self.base_model.save_pretrained(os.path.join(save_directory, self.BASE_MODEL_FOLDER), **kwargs)
        if self.decoder_model is not None:
            self.decoder_model.save_pretrained(os.path.join(save_directory, self.DECODER_MODEL_FOLDER), **kwargs)
    
    def push_to_hub(self, repo_id: str, commit_message: str = "Initial commit", private: bool = False, token: str = None, **kwargs):
        """Push pipeline and all submodels to HuggingFace Hub.
        
        Args:
            repo_id: HF Hub repo ID (e.g. "username/model-name").
            commit_message: Commit message for the upload.
            private: Whether to create a private repo.
            token: HF token for authentication (uses cached token if None).
            **kwargs: Additional arguments passed to HfApi.upload_folder.
            
        Returns:
            URL of the uploaded model on HuggingFace Hub.
        """
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, exist_ok=True, private=private)
        with TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)
            api.upload_folder(repo_id=repo_id, folder_path=tmpdir, commit_message=commit_message, **kwargs)
        return f"https://huggingface.co/{repo_id}"
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, token: str = None, **kwargs) -> "WorldPipeline":
        """Load pipeline and all submodels from a directory or HuggingFace Hub.
        
        Args:
            pretrained_model_name_or_path: Directory or HF Hub model ID.
            token: HF token for private repos (uses cached token if None).
            **kwargs: Additional arguments passed to __init__.
        """
        # Load pipeline config only (no weights for the pipeline itself)
        config = {**cls.load_config(pretrained_model_name_or_path, token=token), **kwargs}
        pipeline = cls(**config)
        
        pipeline.coarse_model = EDMUnet2D.from_pretrained(
            pretrained_model_name_or_path, subfolder=cls.COARSE_MODEL_FOLDER, token=token
        )
        pipeline.base_model = EDMUnet2D.from_pretrained(
            pretrained_model_name_or_path, subfolder=cls.BASE_MODEL_FOLDER, token=token
        )
        pipeline.decoder_model = EDMUnet2D.from_pretrained(
            pretrained_model_name_or_path, subfolder=cls.DECODER_MODEL_FOLDER, token=token
        )
        pipeline._apply_dtype_and_compile()
        
        return pipeline
    
    @property
    def device(self):
        """Infer device from submodel parameters."""
        for model in (self.coarse_model, self.base_model, self.decoder_model):
            if model is not None:
                return next(model.parameters()).device
        return torch.device('cpu')
    
    def to(self, device):
        """Move all submodels to the specified device."""
        if self.coarse_model is not None:
            self.coarse_model = self.coarse_model.to(device)
        if self.base_model is not None:
            self.base_model = self.base_model.to(device)
        if self.decoder_model is not None:
            self.decoder_model = self.decoder_model.to(device)
        self._warmup_compiled_models()
        return self
    
    def bind(
        self,
        hdf5_file: str | None = None,
        mode: str = 'a',
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
    ):
        """Bind the pipeline for generation.
        
        Args:
            hdf5_file: Path to HDF5 file, or 'TEMP' for a temporary file. Required when caching_strategy='indirect', ignored when 'direct'.
            mode: File mode for HDF5 (only used with indirect caching).
            compression: Compression algorithm (only used with indirect caching).
            compression_opts: Compression level (only used with indirect caching).
            
        Returns:
            self for chaining.
        """
        if self.caching_strategy == 'direct':
            self._init_tile_store(None, None, None, None)
        else:
            if hdf5_file is None:
                raise ValueError("hdf5_file is required when caching_strategy='indirect'")
            original_hdf5_file = hdf5_file
            hdf5_file = resolve_hdf5_path(hdf5_file)
            self._is_temp_file = original_hdf5_file.upper() == 'TEMP'
            self._hdf5_file_path = hdf5_file
            
            self._reconcile_params_with_file(hdf5_file)
            self._init_tile_store(hdf5_file, mode, compression, compression_opts)
        
        self._init_conditioning()
        self._build_hierarchy()
        return self
    
    def _reconcile_params_with_file(self, hdf5_file: str):
        """Check stored params vs current, use stored if they exist (with overwrite prompt on mismatch)."""
        ATTR_KEY = 'WORLD_PIPELINE_PARAMS'
        
        def make_current():
            return {
                'seed': self.seed,
                'kwargs': self.kwargs,
            }
        
        if not os.path.exists(hdf5_file):
            with h5py.File(hdf5_file, 'w') as f:
                f.attrs[ATTR_KEY] = json.dumps(make_current(), sort_keys=True)
            return
        
        with h5py.File(hdf5_file, 'a') as f:
            if ATTR_KEY not in f.attrs:
                f.attrs[ATTR_KEY] = json.dumps(make_current(), sort_keys=True)
                return
            
            stored = json.loads(f.attrs[ATTR_KEY])
            current = make_current()
            mismatches = [(k, stored[k], current[k]) for k in current if k in stored and stored[k] != current[k]]
            
            if not mismatches:
                return
            
            print("\n=== Parameter mismatch with stored world file ===")
            for key, stored_val, current_val in mismatches:
                print(f"  {key}:")
                print(f"    stored:  {stored_val}")
                print(f"    current: {current_val}")
            
            choice = input("\nOverwrite stored params with current? (This WILL cause artifacts unless a model was simply moved) [y/N]: ").strip().lower()
            
            if choice == 'y':
                f.attrs[ATTR_KEY] = json.dumps(current, sort_keys=True)
            else:
                self.seed = stored['seed']
                self.kwargs = stored['kwargs']
    
    def _init_tile_store(self, hdf5_file, mode, compression, compression_opts):
        """Initialize the tile store."""
        if self.caching_strategy == 'direct':
            self.tile_store = MemoryTileStore()
        else:
            self.tile_store = HDF5TileStore(
                hdf5_file, mode=mode, compression=compression, 
                compression_opts=compression_opts, tile_cache_size=100
            )
    
    def _init_conditioning(self):
        """Initialize conditioning (synthetic maps, etc)."""
        self.synthetic_map_factory = make_synthetic_map_factory(
            seed=self.seed, 
            frequency_mult=self.kwargs['frequency_mult'], 
            drop_water_pct=self.kwargs['drop_water_pct']
        )
    
    def _build_hierarchy(self):
        """Build the generation hierarchy."""
        self.coarse = self._build_coarse_stage()
        self.latents = self._build_latent_stage()
        self.residual = self._build_decoder_stage()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def empty_cache(self):
        """Clear the cache for all tensors in the pipeline."""
        if self.tile_store is None:
            return
        
        tensor_ids = ["base_coarse_map", "init_latent_map", "init_residual_map"]
        # Add step_latent_map tensors (there's typically one: step_latent_map_0)
        inter_t = [torch.arctan(torch.tensor(0.35) / 0.5)]
        for i in range(len(inter_t)):
            tensor_ids.append(f"step_latent_map_{i}")
        
        for tensor_id in tensor_ids:
            self.tile_store.evict_cache_for(tensor_id, 0)
    
    def close(self):
        """Release resources associated with this pipeline."""
        if self.tile_store is not None and hasattr(self.tile_store, 'close'):
            self.tile_store.close()
        # Clean up temporary file if this pipeline created one
        if self._is_temp_file:
            cleanup_temp_file(self._hdf5_file_path)

    # =========================================================================
    # Coarse Stage
    # =========================================================================
    
    def _coarse_inference(self, ctx, scheduler, weight_window, t_cond, cond_inputs, pool_size=1):
        """Run inference for one coarse tile."""
        TILE_SIZE = 64
        TILE_STRIDE = TILE_SIZE - 16
        MODEL_MEANS = torch.tensor(self.kwargs['coarse_means'])
        MODEL_STDS = torch.tensor(self.kwargs['coarse_stds'])
        
        model_dtype = self._dtype or torch.float32
        _, i, j = ctx
        i1, j1 = i * (TILE_STRIDE // pool_size), j * (TILE_STRIDE // pool_size)
        i1, j1 = i1 * pool_size, j1 * pool_size
        i2, j2 = i1 + TILE_SIZE, j1 + TILE_SIZE
        
        synthetic_map = torch.as_tensor(self.synthetic_map_factory(j1, i1, j2, i2))
        synthetic_map[1] = torch.where(synthetic_map[1] > 20, synthetic_map[1], (synthetic_map[1] - 20) * 1.25 + 20)
        synthetic_map = (synthetic_map - MODEL_MEANS[[0, 2, 3, 4, 5], None, None]) / MODEL_STDS[[0, 2, 3, 4, 5], None, None]
        synthetic_map = synthetic_map.to(self.device, dtype=model_dtype)[None]
        
        cond_noise = torch.as_tensor(gaussian_noise_patch(
            self.seed, i1, j1, TILE_SIZE, TILE_SIZE, 
            channels=5, tile_h=TILE_SIZE, tile_w=TILE_SIZE
        ), device=self.device, dtype=model_dtype)[None]
        cond_img = torch.cos(t_cond.view(1, -1, 1, 1)) * synthetic_map + torch.sin(t_cond.view(1, -1, 1, 1)) * cond_noise
        
        scheduler.set_timesteps(20)
        sample_noise = torch.as_tensor(gaussian_noise_patch(
            self.seed + 1, i1, j1, TILE_SIZE, TILE_SIZE,
            channels=6, tile_h=TILE_SIZE, tile_w=TILE_SIZE
        ), device=self.device, dtype=model_dtype)[None]
        sample = sample_noise * scheduler.sigmas[0]
        
        for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
            t = t.to(self.device)
            sigma = sigma.to(self.device)
            scaled_in = scheduler.precondition_inputs(sample, sigma)
            cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1))

            x_in = torch.cat([scaled_in, cond_img], dim=1)
            model_out = self.coarse_model(x_in, noise_labels=torch.tensor([cnoise.item()], device=self.device, dtype=model_dtype), conditional_inputs=cond_inputs)
            sample = scheduler.step(model_out, t, sample).prev_sample
        
        sample = sample.cpu().float() / scheduler.config.sigma_data
        sample = (sample * MODEL_STDS.view(1, -1, 1, 1)) + MODEL_MEANS.view(1, -1, 1, 1)
        sample[0, 1] = sample[0, 0] - sample[0, 1]
        
        if pool_size > 1:
            sample = self._pool_coarse_conditioning(sample[0], pool_size)[None]
        
        output = torch.cat([sample[0] * weight_window[None], weight_window[None]], dim=0)
        return output
    
    def _build_coarse_stage(self):
        """Build and register the coarse stage."""
        TILE_SIZE = 64
        TILE_STRIDE = TILE_SIZE - 16
        COND_SNR = torch.tensor(self.kwargs['cond_snr'])
        
        coarse_pool = self.kwargs['coarse_pooling']
        assert TILE_SIZE % coarse_pool == 0, f"coarse_pooling {coarse_pool} must divide TILE_SIZE {TILE_SIZE}"
        assert TILE_STRIDE % coarse_pool == 0, f"coarse_pooling {coarse_pool} must divide TILE_STRIDE {TILE_STRIDE}"
        
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
        weight_window = linear_weight_window(TILE_SIZE // coarse_pool, 'cpu', torch.float32)
        
        t_cond = torch.atan(COND_SNR).to(self.device)
        vals = torch.log(torch.tan(t_cond) / 8.0)
        cond_inputs = [v.detach().view(-1) for v in vals]
        
        def f(ctx):
            return self._coarse_inference(ctx, scheduler, weight_window, t_cond, cond_inputs, pool_size=coarse_pool)
                    
        output_window = TensorWindow(
            size=(7, TILE_SIZE // coarse_pool, TILE_SIZE // coarse_pool), 
            stride=(7, TILE_STRIDE // coarse_pool, TILE_STRIDE // coarse_pool)
        )
        return self.tile_store.get_or_create(
            "base_coarse_map",
            shape=(7, None, None),
            f=f,
            output_window=output_window,
            **self._common_tile_kwargs()
        )

    # =========================================================================
    # Latent Stage
    # =========================================================================
    
    def _pool_channel(self, x, pool_size, mode):
        """Pool a single channel with the specified mode (max/avg/min)."""
        x = x.unsqueeze(0)
        if mode == 'max':
            return torch.nn.functional.max_pool2d(x, kernel_size=pool_size, stride=pool_size).squeeze(0)
        elif mode == 'min':
            return -torch.nn.functional.max_pool2d(-x, kernel_size=pool_size, stride=pool_size).squeeze(0)
        else:  # avg
            return torch.nn.functional.avg_pool2d(x, kernel_size=pool_size, stride=pool_size).squeeze(0)

    def _pool_coarse_conditioning(self, cond_img, pool_size):
        """Pool coarse conditioning from (C, H*n, W*n) to (C, H, W)."""
        if pool_size == 1:
            return cond_img
        n = pool_size
        ch0_pooled = self._pool_channel(cond_img[0:1], n, self.kwargs['elev_coarse_pool_mode'])
        ch1_pooled = self._pool_channel(cond_img[1:2], n, self.kwargs['p5_coarse_pool_mode'])
        ch_rest_pooled = torch.nn.functional.avg_pool2d(cond_img[2:].unsqueeze(0), kernel_size=n, stride=n).squeeze(0)
        return torch.cat([ch0_pooled, ch1_pooled, ch_rest_pooled], dim=0)

    def _process_latent_conditioning(self, cond_img, histogram_raw, cond_means, cond_stds, noise_level):
        """Process conditioning for latent stage."""
        COND_MAX_NOISE = 0.0
        
        cond_img[0:1] = cond_img[0:1] + torch.randn_like(cond_img[:, 0:1]) * noise_level * COND_MAX_NOISE
        cond_img[1:2] = cond_img[1:2] + torch.randn_like(cond_img[:, 1:2]) * noise_level * COND_MAX_NOISE
        cond_img = (cond_img - cond_means.to(cond_img.device).view(1, -1, 1, 1)) / cond_stds.to(cond_img.device).view(1, -1, 1, 1)
        
        cond_img[0:1] = cond_img[0:1].nan_to_num(cond_means[0])
        cond_img[1:2] = cond_img[1:2].nan_to_num(cond_means[1])
        
        means_crop = cond_img[:, 0:1]
        p5_crop = cond_img[:, 1:2]
        climate_means_crop = cond_img[:, 2:6, 1:3, 1:3].mean(dim=(2, 3))
        mask_crop = cond_img[:, 6:7]
        
        climate_means_crop[torch.isnan(climate_means_crop)] = torch.randn_like(climate_means_crop[torch.isnan(climate_means_crop)])
        
        noise_level_norm = (noise_level - 0.5) * np.sqrt(12)
        return mp_concat([
            means_crop.flatten(1), p5_crop.flatten(1), climate_means_crop.flatten(1), 
            mask_crop.flatten(1), histogram_raw, noise_level_norm.view(-1, 1)
        ], dim=1).float()

    def _latent_inference(self, ctxs, samples, cond_imgs, t, scheduler, weight_window, histogram_raw, cond_means, cond_stds, seed_offset=0):
        """Run inference for latent tiles."""
        TILE_SIZE = 64
        TILE_STRIDE = TILE_SIZE // 2
        NOISE_LEVEL = 0.0
        
        if self.log_mode == 'verbose':
            print(f"Latent f batch size {len(ctxs)} at {ctxs}")
        if MOCK:
            return [torch.ones((6, TILE_SIZE, TILE_SIZE)) for _ in ctxs]
        
        if samples is None:
            samples = [None] * len(ctxs)
        
        model_in_list = []
        cond_inputs_list = []
        samples_processed = []
        
        t_tensor = torch.as_tensor(t, device=self.device)
        model_dtype = self._dtype or torch.float32
        
        for ctx, sample, cond_img in zip(ctxs, samples, cond_imgs):
            if sample is None:
                sample = torch.zeros((1, 5, TILE_SIZE, TILE_SIZE), device=self.device, dtype=model_dtype)
            else:
                sample = torch.as_tensor(sample, device=self.device, dtype=model_dtype)
                sample = sample[:-1] / sample[-1:] * scheduler.config.sigma_data
            
            cond_img = cond_img[:-1] / cond_img[-1:]
            
            mask = torch.ones((1, 4, 4))
            cond_img = torch.cat([cond_img, mask], dim=0)[None]
            
            cond_inputs = self._process_latent_conditioning(
                cond_img, histogram_raw, cond_means, cond_stds, torch.tensor(NOISE_LEVEL)
            ).to(self.device, dtype=model_dtype)
            
            noise = torch.as_tensor(gaussian_noise_patch(
                self.seed + seed_offset, ctx[1]*TILE_STRIDE, ctx[2]*TILE_STRIDE, 
                TILE_SIZE, TILE_SIZE, channels=5, tile_h=TILE_SIZE, tile_w=TILE_SIZE
            ))[None]
            
            z = noise.to(self.device, dtype=model_dtype) * scheduler.config.sigma_data
            t_view = t_tensor.view(1, 1, 1, 1).to(self.device)
            x_t = torch.cos(t_view) * sample + torch.sin(t_view) * z
            model_in = (x_t / scheduler.config.sigma_data).to(self.device)
            
            model_in_list.append(model_in)
            cond_inputs_list.append(cond_inputs)
            samples_processed.append((sample, x_t))

        if not model_in_list:
            return []

        actual_batch_size = len(model_in_list)
        padded_batch_size = self._get_padded_batch_size(actual_batch_size) if self.torch_compile else actual_batch_size
        
        model_in_batch = torch.cat(model_in_list, dim=0)
        cond_inputs_batch = torch.cat(cond_inputs_list, dim=0)
        noise_labels_batch = torch.full((padded_batch_size,), t_tensor.item(), device=self.device, dtype=model_dtype)
        
        # Pad to next legal batch size for torch.compile efficiency
        if padded_batch_size > actual_batch_size:
            pad_count = padded_batch_size - actual_batch_size
            model_in_batch = torch.cat([model_in_batch, model_in_batch[:1].repeat(pad_count, 1, 1, 1)], dim=0)
            cond_inputs_batch = torch.cat([cond_inputs_batch, cond_inputs_batch[:1].repeat(pad_count, 1)], dim=0)
        
        pred_batch = -self.base_model(model_in_batch, noise_labels=noise_labels_batch, conditional_inputs=[cond_inputs_batch])
        
        outputs = []
        for i in range(actual_batch_size):
            pred = pred_batch[i]
            sample, x_t = samples_processed[i]
            pred = pred.unsqueeze(0)
            t_view = t_tensor.view(1, 1, 1, 1).to(self.device)
            sample = torch.cos(t_view) * x_t - torch.sin(t_view) * scheduler.config.sigma_data * pred
            sample = sample.cpu().float() / scheduler.config.sigma_data
            outputs.append(torch.cat([sample[0] * weight_window[None], weight_window[None]], dim=0))
        return outputs
    
    def _build_latent_stage(self):
        """Build and register the latent stage."""
        TILE_SIZE = 64
        TILE_STRIDE = TILE_SIZE // 2
        COND_INPUT_MEAN = torch.tensor([14.99, 11.65, 15.87, 619.26, 833.12, 69.40, 0.66])
        COND_INPUT_STD = torch.tensor([21.72, 21.78, 10.40, 452.29, 738.09, 34.59, 0.47])
        HISTOGRAM_RAW = torch.tensor([self.kwargs['histogram_raw']])
        
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
        weight_window = linear_weight_window(TILE_SIZE, 'cpu', torch.float32)

        t_init = torch.atan(scheduler.sigmas[0] / scheduler.config.sigma_data)
        output_window = TensorWindow(size=(6, TILE_SIZE, TILE_SIZE), stride=(6, TILE_STRIDE, TILE_STRIDE))
        coarse_window = TensorWindow(size=(7, 4, 4), stride=(7, 1, 1), offset=(0, -1, -1))
        
        tensor = self.tile_store.get_or_create(
            "init_latent_map",
            shape=(6, None, None),
            f=lambda ctxs, conds: self._latent_inference(
                ctxs, None, conds, t_init, scheduler, weight_window, HISTOGRAM_RAW, COND_INPUT_MEAN, COND_INPUT_STD, seed_offset=5819
            ),
            output_window=output_window,
            args=(self.coarse,),
            args_windows=(coarse_window,),
            batch_size=self.latents_batch_size,
            **self._common_tile_kwargs()
        )
        
        # Skip intermediate refinement steps for 1-step latent model
        if not self.onestep_latent:
            inter_t = [torch.arctan(torch.tensor(0.35) / 0.5)]
            for i, t in enumerate(inter_t):
                tensor = self.tile_store.get_or_create(
                    f"step_latent_map_{i}",
                    shape=(6, None, None),
                    f=lambda ctxs, samples, conds, t=t, i=i: self._latent_inference(
                        ctxs, samples, conds, t, scheduler, weight_window, HISTOGRAM_RAW, COND_INPUT_MEAN, COND_INPUT_STD, seed_offset=5820+i
                    ),
                    output_window=output_window,
                    args=(tensor, self.coarse,),
                    args_windows=(output_window, coarse_window,),
                    batch_size=self.latents_batch_size,
                    **self._common_tile_kwargs()
                )
        
        return tensor
    
    # =========================================================================
    # Decoder Stage
    # =========================================================================
    
    def _decoder_inference(self, ctx, latents, scheduler, weight_window, t_list, tile_size, tile_stride):
        """Run inference for one decoder tile."""
        TILE_SIZE = tile_size
        TILE_STRIDE = tile_stride
        
        if self.log_mode == 'verbose':
            print(f"Residual f at {ctx}")
        if MOCK:
            return torch.ones((2, TILE_SIZE, TILE_SIZE))
        
        model_dtype = self._dtype or torch.float32
        
        sample = torch.zeros((1, 1, TILE_SIZE, TILE_SIZE), device=self.device, dtype=model_dtype)
        lc = self.latent_compression
        latents = (latents[:-1] / latents[-1:])[:4].view(1, 4, TILE_SIZE//lc, TILE_SIZE//lc)
        upsampled_latents = torch.nn.functional.interpolate(
            latents, size=(TILE_SIZE, TILE_SIZE), mode='nearest'
        ).to(self.device, dtype=model_dtype)
        
        for i, t in enumerate(t_list):
            noise = torch.as_tensor(gaussian_noise_patch(
                self.seed + 5819 + i, ctx[1]*TILE_STRIDE, ctx[2]*TILE_STRIDE, 
                TILE_SIZE, TILE_SIZE, channels=1, tile_h=TILE_SIZE, tile_w=TILE_SIZE
            ))[None]
            t = t.view(1, 1, 1, 1).to(self.device)
            z = noise.to(self.device, dtype=model_dtype) * scheduler.config.sigma_data
            x_t = torch.cos(t) * sample + torch.sin(t) * z
            model_in = (x_t / scheduler.config.sigma_data).to(self.device)
            model_in = torch.cat([model_in, upsampled_latents], dim=1)
            pred = -self.decoder_model(model_in, noise_labels=torch.tensor([t.item()], device=self.device, dtype=model_dtype), conditional_inputs=[])
            sample = torch.cos(t) * x_t - torch.sin(t) * scheduler.config.sigma_data * pred
            
        sample = sample.cpu().float() / scheduler.config.sigma_data
        return torch.cat([sample[0] * weight_window[None], weight_window[None]], dim=0)
    
    def _build_decoder_stage(self):
        """Build and register the decoder stage."""
        TILE_SIZE = self.decoder_tile_size
        TILE_STRIDE = self.decoder_tile_stride
        
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
        weight_window = linear_weight_window(TILE_SIZE, 'cpu', torch.float32)
        
        t_list = [torch.atan(scheduler.sigmas[0] / scheduler.config.sigma_data)]
        #t_list += [torch.arctan(torch.tensor(0.065) / 0.5)]
        
        def f(ctx, latents):
            return self._decoder_inference(ctx, latents, scheduler, weight_window, t_list, TILE_SIZE, TILE_STRIDE)
        
        lc = self.latent_compression
        output_window = TensorWindow(size=(2, TILE_SIZE, TILE_SIZE), stride=(2, TILE_STRIDE, TILE_STRIDE))
        input_window = TensorWindow(size=(6, TILE_SIZE//lc, TILE_SIZE//lc), stride=(6, TILE_STRIDE//lc, TILE_STRIDE//lc))
        
        return self.tile_store.get_or_create(
            "init_residual_map",
            shape=(2, None, None),
            f=f,
            output_window=output_window,
            args=(self.latents,),
            args_windows=(input_window,),
            **self._common_tile_kwargs()
        )

    # =========================================================================
    # Output Methods
    # =========================================================================

    @torch.no_grad()
    def _compute_elev(self, i1, j1, i2, j2, residual_map, scale: int):
        """Compute elevation from residual map."""
        LOWFREQ_MEAN = -31.4
        LOWFREQ_STD = 38.6
        RESIDUAL_MEAN = self.kwargs['residual_mean']
        RESIDUAL_STD = self.kwargs['residual_std']
        
        sigma = 5
        kernel_size = (int(sigma * 2) // 2) * 2 + 1
        pad_lr = kernel_size // 2 + 1
        pad_hr = pad_lr * scale

        def ceil_div(a: int, b: int) -> int:
            return -((-a) // b)

        # Add padding
        pi1_raw, pj1_raw = i1 - pad_hr, j1 - pad_hr
        pi2_raw, pj2_raw = i2 + pad_hr, j2 + pad_hr
        pi1 = (pi1_raw // scale) * scale
        pj1 = (pj1_raw // scale) * scale
        pi2 = ceil_div(pi2_raw, scale) * scale
        pj2 = ceil_div(pj2_raw, scale) * scale

        residual_init = residual_map[:, pi1:pi2, pj1:pj2]
        residual_p = (residual_init[0] / residual_init[1]) * RESIDUAL_STD + RESIDUAL_MEAN
        latents_init = self.latents[:, pi1//scale:pi2//scale, pj1//scale:pj2//scale]
        latents_norm = latents_init[:-1] / latents_init[-1:]
        lowfreq_p = latents_norm[4] * LOWFREQ_STD + LOWFREQ_MEAN

        residual_p, lowfreq_p = laplacian_denoise(residual_p, lowfreq_p, sigma=sigma)
        elev_p = laplacian_decode(residual_p, lowfreq_p)

        oi, oj = i1 - pi1, j1 - pj1
        h, w = i2 - i1, j2 - j1
        elev_sqrt = elev_p[oi:oi + h, oj:oj + w]
        elev = torch.sign(elev_sqrt) * torch.square(elev_sqrt)
        return elev
    
    def _compute_climate(self, i1: int, j1: int, i2: int, j2: int, elev: torch.Tensor, scale: int) -> torch.Tensor | None:
        """Compute climate from coarse map."""
        def ceil_div(a: int, b: int) -> int:
            return -((-a) // b)

        ci1 = i1 // (32 * scale)
        cj1 = j1 // (32 * scale)
        ci2 = ceil_div(i2, 32 * scale)
        cj2 = ceil_div(j2, 32 * scale)

        coarse_window_size = 15
        coarse_padding = (coarse_window_size - 1) // 2 + 1
        coarse_init = self.coarse[:, ci1-coarse_padding:ci2+coarse_padding, cj1-coarse_padding:cj2+coarse_padding]
        coarse_map = coarse_init[:-1] / coarse_init[-1:]
        coarse_elev_denorm = torch.sign(coarse_map[0]) * torch.square(torch.maximum(torch.zeros_like(coarse_map[0]), coarse_map[0]))
        temp_baseline, beta = local_baseline_temperature_torch(coarse_map[2], coarse_elev_denorm, win=coarse_window_size, fallback_threshold=0.02)
        central_coarse = coarse_map[:, coarse_window_size//2:-(coarse_window_size//2), coarse_window_size//2:-(coarse_window_size//2)]

        S = 32 * scale
        H_src, W_src = temp_baseline.shape[-2:]
        
        device = elev.device
        grid_i = torch.arange(i1, i2, device=device)
        grid_j = torch.arange(j1, j2, device=device)
        ii, jj = torch.meshgrid(grid_i, grid_j, indexing='ij')
        
        u = (ii + 0.5) / S - ci1 + 0.5
        v = (jj + 0.5) / S - cj1 + 0.5
        
        grid_y = (u + 0.5) * 2 / H_src - 1
        grid_x = (v + 0.5) * 2 / W_src - 1
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        if temp_baseline.ndim == 2:
            temp_baseline = temp_baseline.unsqueeze(0)
        if beta.ndim == 2:
            beta = beta.unsqueeze(0)
            
        features = torch.cat([temp_baseline, beta, central_coarse], dim=0).unsqueeze(0)
        
        features_up = torch.nn.functional.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=False)
        features_up = features_up.squeeze(0)
        
        temp_baseline_up = features_up[0:1]
        beta_up = features_up[1:2]
        coarse_up = features_up[2:]
        
        temp_realistic = temp_baseline_up[0] + beta_up[0] * torch.maximum(elev, torch.zeros_like(elev))
        climate = torch.stack([temp_realistic, coarse_up[3], coarse_up[4], coarse_up[5], beta_up[0]])
        return climate
    
    def get(self, i1, j1, i2, j2, with_climate=True):
        """
        Get terrain data for the given bounding box.
        
        Args:
            i1, j1, i2, j2: Bounding box coordinates
            with_climate: Whether to compute climate data
            
        Returns:
            dict with 'elev' (H, W) in meters and optionally 'climate' (5, H, W)
        """
        elev = self._compute_elev(i1, j1, i2, j2, self.residual, scale=self.latent_compression)
        climate = self._compute_climate(i1, j1, i2, j2, elev, scale=self.latent_compression) if with_climate else None
        
        return {
            'elev': elev,
            'climate': climate,
        }
