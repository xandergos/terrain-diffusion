import random

import torch
from infinite_tensor import HDF5TileStore, TensorWindow
from synthetic_map import make_synthetic_map_factory
from terrain_diffusion.models.edm_unet import EDMUnet2D
import numpy as np
from typing import Union
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
import matplotlib.pyplot as plt
    
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
    
class WorldPipeline:
    def __init__(self, 
                 hdf5_file, 
                 mode='a', 
                 compression: str | None = "gzip", 
                 compression_opts: int | None = 4,
                 seed=None,
                 device='cpu',
                 coarse_model="checkpoints/models/diffusion_coarse"):
        self.device = device
        self.tile_store = HDF5TileStore(hdf5_file, mode=mode, compression=compression, compression_opts=compression_opts)
        self.seed = seed or random.randint(0, 2**31-1)
        
        self.synthetic_map_factory = make_synthetic_map_factory(seed=self.seed)
        
        # Create coarse perlin
        self.coarse_model = EDMUnet2D.from_pretrained(coarse_model).to(device)
        self.coarse = self._get_coarse_map()
        
    def _get_coarse_map(self):
        TILE_SIZE = 64
        TILE_STRIDE = 32
        MODEL_MEANS = torch.tensor([-37.67916460232751, 2.22578822145657, 18.030293275011356, 333.8442390481231, 1350.1259248456176, 52.444339366764396])
        MODEL_STDS = torch.tensor([39.68515115440358, 3.0981253981231522, 8.940333096712806, 322.25238547630295, 856.3430083394657, 30.982620765341043])
        COND_SNR = torch.tensor([0.1, 0.5, 0.5, 0.5, 0.5])
        
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
        weight_window = linear_weight_window(TILE_SIZE, 'cpu', torch.float32)
        
        t_cond = torch.atan(COND_SNR).to(self.device)
        vals = torch.log(torch.tan(t_cond) / 8.0)
        cond_inputs = [v.detach().view(-1) for v in vals]
        
        def f(ctx):
            _, i, j = ctx
            i1, j1 = i * TILE_STRIDE, j * TILE_STRIDE
            i2, j2 = i1 + TILE_SIZE, j1 + TILE_SIZE
            synthetic_map = torch.as_tensor(self.synthetic_map_factory(i1, j1, i2, j2))
            synthetic_map = (synthetic_map - MODEL_MEANS[[0, 2, 3, 4, 5], None, None]) / MODEL_STDS[[0, 2, 3, 4, 5], None, None]
            synthetic_map = synthetic_map.to(self.device)[None]
            
            cond_img = torch.cos(t_cond.view(1, -1, 1, 1)) * synthetic_map + torch.sin(t_cond.view(1, -1, 1, 1)) * torch.randn_like(synthetic_map)
            
            scheduler.set_timesteps(15)
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
        
        
    
if __name__ == '__main__':
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix='.h5') as tmp_file:
        x = WorldPipeline(tmp_file.name, device='cuda')
        c = x.coarse[:, 0:128, 0:128]
        plt.imshow(c[0].numpy())
        plt.show()
