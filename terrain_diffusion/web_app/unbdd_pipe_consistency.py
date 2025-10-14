import os
import uuid

import torch
from tqdm import tqdm
from terrain_diffusion.common.model_utils import get_model
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.datasets import H5LatentsDataset
from terrain_diffusion.models.mp_generator import MPGenerator
from ema_pytorch import PostHocEMA
from safetensors.torch import load_model
import matplotlib.pyplot as plt
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.inference.scheduler.functional_dpmsolver import multistep_dpm_solver_second_order_update, dpm_solver_first_order_update, precondition_outputs
from matplotlib.widgets import Slider
from infinite_tensor import MemoryTileStore, TensorWindow

def create_unbounded_pipe(sigmas: list[int] = None, cond_input_scaling: float = 1.0, use_consistency_decoder: bool = False):
    if sigmas is None:
        sigmas = [80, 10, 1]
    
    tile_store = MemoryTileStore()
    
    def get_weights(size):
        s = size
        mid = (s - 1) / 2
        y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
        epsilon = 1e-3
        distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y - mid).float() / mid, 0, 1)
        distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x - mid).float() / mid, 0, 1)
        return (distance_y * distance_x)[None, None, :, :]

    weights64 = get_weights(64)

    def latent_generator(ctx):
        """Generate infinite latent noise for GAN input"""
        seed = (ctx[-1] * 16777619 + ctx[-2] * 373587943) & 0x7fffffff
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(1, 128, 128, 128, generator=generator)  # Return on CPU

    cond_inputs_size = round(8 * 8 * cond_input_scaling)
    def cond_inputs_generator(ctx, latent_tensor):
        """Generate conditional inputs from latent using GAN"""
        # Get the latent window and move to device for processing
        latent_window = latent_tensor.to(device)
        with torch.no_grad():
            cond_result = gan.raw_forward(latent_window)[:, :, 1:-1, 1:-1]
            
        # Pad cond_result with 5 additional zero channels (For simple gan)
        # cond_result = torch.cat([cond_result, torch.zeros_like(cond_result[:, :1]).repeat(1, 5, 1, 1)], dim=1)
        
        # Interpolate to window size
        cond_interpolated = torch.nn.functional.interpolate(cond_result, size=(cond_inputs_size, cond_inputs_size), mode='nearest')
        return cond_interpolated[0].cpu()  # Return on CPU

    def noise_generator(ctx):
        """Generate infinite noise tensor"""
        return torch.randn(1, 10, 64, 64) * sigma_data  # Return on CPU

    def process_model_step(ctx, x_t_tensor, cond_inputs_tensor, cnoise_val):
        """Process a single model step on infinite tensors"""
        # Move inputs to device for processing
        x_t_window = x_t_tensor.to(device)
        cond_window = cond_inputs_tensor.to(device)
        
        best_pixel = torch.argmax(cond_window[0])
        
        best_elev = torch.flatten(cond_window[0])[best_pixel] * 2435 - 2607
        best_elev = (torch.sign(best_elev) * torch.sqrt(torch.abs(best_elev)) + 31.4) / 38.6
        mean_temp = torch.flatten(cond_window[1])[best_pixel]
        std_temp = torch.flatten(cond_window[2])[best_pixel]
        mean_prec = torch.flatten(cond_window[3])[best_pixel]
        std_prec = torch.flatten(cond_window[4])[best_pixel]
        all_water = (torch.flatten(cond_window[5])[best_pixel] < -3).long()
        
        # Run model
        with torch.no_grad():
            model_output = model(x_t_window / sigma_data,
                                noise_labels=cnoise_val.to(device).expand(x_t_window.shape[0]),
                                conditional_inputs=[
                                    best_elev.to(device).view(1).expand(x_t_window.shape[0]),
                                    mean_temp.to(device).view(1).expand(x_t_window.shape[0]),
                                    std_temp.to(device).view(1).expand(x_t_window.shape[0]),
                                    mean_prec.to(device).view(1).expand(x_t_window.shape[0]),
                                    std_prec.to(device).view(1).expand(x_t_window.shape[0]),
                                    all_water.to(device).view(1).expand(x_t_window.shape[0])
                                    ])
        
        # Return on CPU
        model_output = model_output.cpu() * weights64
        model_output = torch.cat([model_output, weights64], dim=1)
        return model_output

    device = 'cuda'
    torch.no_grad().__enter__()
    gan = get_model(MPGenerator, 'checkpoints/gan/latest_checkpoint', sigma_rel=0.2, device=device)
    model = get_model(EDMUnet2D, 'checkpoints/consistency_base-128x3/409kimg_checkpoint', sigma_rel=0.05, device=device)
    autoencoder = EDMAutoencoder.from_pretrained('checkpoints/models/autoencoder_x8').to(device)

    # Optional consistency decoder (reconstruct residual+water from latent cond image)
    consistency_decoder = None
    if use_consistency_decoder:
        try:
            consistency_decoder = EDMUnet2D.from_pretrained('checkpoints/models/early_decoder_ckpt').to(device)
            consistency_decoder.eval()
            print(f"Loaded consistency decoder from checkpoints/models/early_decoder_ckpt")
        except Exception as e:
            print(f"Failed to load consistency decoder at checkpoints/models/early_decoder_ckpt: {e}")
            consistency_decoder = None

    dataset = H5LatentsDataset('data/dataset.h5', 64, [[0.1, 1.0]], [90], [1], eval_dataset=False,
                            latents_mean=[0, 0, 0, 0],
                            latents_std=[1, 1, 1, 1],
                            sigma_data=0.5,
                            split="train",
                            beauty_dist=[[1, 1, 1, 1, 1]],
                            residual_mean=0.00216,
                            residual_std=1.1678,
                            watercover_mean=0.08018,
                            watercover_std=0.26459)

    sigma_data = 0.5

    # Create infinite latent tensor
    latent_infinite = tile_store.get_or_create(
        uuid.uuid4(),
        shape=(1, 128, None, None),
        f=latent_generator,
        output_window=TensorWindow((1, 128, 128, 128))
    )

    # Create infinite conditional inputs tensor
    cond_inputs_infinite = tile_store.get_or_create(
        uuid.uuid4(),
        shape=(6, None, None),
        f=cond_inputs_generator,
        output_window=TensorWindow((6, cond_inputs_size, cond_inputs_size)),
        args=(latent_infinite,),
        args_windows=[TensorWindow((1, 128, 21, 21), stride=(1, 128, 2, 2), dimension_map=(None, 0, 1, 2))]
    )

    # Mark latent_infinite for cleanup since it's only used by cond_inputs_infinite
    latent_infinite.mark_for_cleanup()

    # Initialize infinite prediction tensor
    pred_x0_infinite = tile_store.get_or_create(
        uuid.uuid4(),
        shape=(1, 10, None, None),
        f=lambda ctx: torch.zeros(1, 10, 64, 64) * sigma_data,  # Return on CPU
        output_window=TensorWindow((1, 10, 64, 64))
    )

    # Process through diffusion steps
    for i, sigma in enumerate(sigmas):
        sigma = torch.tensor(sigma, dtype=torch.float32)
        t = torch.atan(sigma / sigma_data)
        sigma, t = sigma.to(device), t.to(device)
        cnoise = t.expand(1)
        
        # Create noise for this step
        noise_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, 10, None, None),
            f=noise_generator,
            output_window=TensorWindow((1, 10, 64, 64))
        )
        
        # Create x_t = sin(t) * noise + cos(t) * pred_x0
        def create_x_t(ctx, noise_tensor, pred_tensor):
            return torch.sin(t.cpu()) * noise_tensor + torch.cos(t.cpu()) * pred_tensor
        
        x_t_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, 10, None, None),
            f=create_x_t,
            output_window=TensorWindow((1, 10, 64, 64)),
            args=(noise_infinite, pred_x0_infinite),
            args_windows=[TensorWindow((1, 10, 64, 64)), TensorWindow((1, 10, 64, 64))]
        )
        
        # Mark noise_infinite for cleanup since it's only used by x_t_infinite
        noise_infinite.mark_for_cleanup()
        
        # Create model output tensor
        def model_step_wrapper(ctx, x_t_tensor, cond_tensor):
            return process_model_step(ctx, x_t_tensor, cond_tensor, cnoise)
        
        model_output_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, 11, None, None),
            f=model_step_wrapper,
            output_window=TensorWindow((1, 11, 64, 64), stride=(1, 11, 32, 32)),
            args=(x_t_infinite, cond_inputs_infinite),
            args_windows=[TensorWindow((1, 10, 64, 64), stride=(1, 10, 32, 32)), TensorWindow((6, 8, 8), stride=(6, 4, 4), dimension_map=(1, 2, 3))]
        )
        
        # Mark x_t_infinite for cleanup since it's only used by model_output_infinite
        x_t_infinite.mark_for_cleanup()
        
        model_output_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, 10, None, None),
            f=lambda ctx, x_t_tensor: x_t_tensor[:, :-1] / x_t_tensor[:, -1:],
            output_window=TensorWindow((1, 10, 64, 64)),
            args=(model_output_infinite,),
            args_windows=[TensorWindow((1, 11, 64, 64))]
        )
        
        # Update pred_x0 = cos(t) * x_t + sin(t) * model_output * sigma_data
        def update_pred_x0(ctx, x_t_tensor, model_output_tensor):
            return torch.cos(t.cpu()) * x_t_tensor + torch.sin(t.cpu()) * model_output_tensor * sigma_data
        
        new_pred_x0_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, 10, None, None),
            f=update_pred_x0,
            output_window=TensorWindow((1, 10, 64, 64)),
            args=(x_t_infinite, model_output_infinite),
            args_windows=[TensorWindow((1, 10, 64, 64)), TensorWindow((1, 10, 64, 64))]
        )
        
        # Mark model_output_infinite for cleanup since it's only used by new_pred_x0_infinite
        model_output_infinite.mark_for_cleanup()
        
        # If this is not the last iteration, mark the old pred_x0_infinite for cleanup
        if i < len(sigmas) - 1:
            pred_x0_infinite.mark_for_cleanup()
        
        pred_x0_infinite = new_pred_x0_infinite

    weights512 = get_weights(512)
    # Final processing and decoding
    def decode_samples(ctx, pred_tensor):
        # Move to device for processing
        samples = (pred_tensor * 2).to(device)
        latent = samples[:, :4]
        lowfreq = samples[:, 4:5]

        # Choose decoding path
        if consistency_decoder is not None:
            # Build cond_img by upsampling latent to 128x128 (as in H5DecoderTerrainDataset)
            cond_img = torch.nn.functional.interpolate(latent, size=(128, 128), mode='nearest')

            # Initialize samples for residual+water (2 channels)
            rw = torch.zeros((cond_img.shape[0], 2, 128, 128), device=device)

            # Two-step consistency update (mirrors visualize_consistency_decoder)
            sigma_data_local = sigma_data
            t_list = [torch.atan(torch.tensor(sigmas[0], dtype=torch.float32, device=device) / sigma_data_local),
                      torch.tensor(1.1, dtype=torch.float32, device=device)]
            for tval in t_list:
                t = tval.view(1, 1, 1, 1).expand(rw.shape[0], 1, 1, 1)
                z = torch.randn_like(rw) * sigma_data_local
                x_t = torch.cos(t) * rw + torch.sin(t) * z
                model_input = torch.cat([x_t / sigma_data_local, cond_img], dim=1)
                pred = -consistency_decoder(model_input, noise_labels=t.flatten(), conditional_inputs=[])
                rw = torch.cos(t) * x_t - torch.sin(t) * sigma_data_local * pred

            decoded = rw / sigma_data_local
            residual, watercover = decoded[:, :1], decoded[:, 1:2]
        else:
            with torch.no_grad():
                decoded = autoencoder.decode(latent)
            residual, watercover = decoded[:, :1], decoded[:, 1:2]

        # Denormalize and compose terrain
        watercover = torch.clip(dataset.denormalize_watercover(watercover), 0, 1)
        residual = dataset.denormalize_residual(residual)
        lowfreq = dataset.denormalize_lowfreq(lowfreq)
        residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
        decoded_terrain = laplacian_decode(residual, lowfreq)

        climate = samples[:, 5:9]
        climate = torch.nn.functional.interpolate(climate, size=decoded_terrain.shape[-2:], mode='nearest').cpu()
        climate = dataset.denormalize_climate(climate, 90)

        decoded_terrain = torch.cat([decoded_terrain.cpu() * weights512, watercover.cpu() * weights512, climate * weights512, weights512], dim=1)
        return decoded_terrain

    decoded_terrain_infinite = tile_store.get_or_create(
        uuid.uuid4(),
        shape=(1, 7, None, None),
        f=decode_samples,
        output_window=TensorWindow((1, 7, 512, 512), stride=(1, 7, 256, 256)),
        args=(pred_x0_infinite,),
        args_windows=[TensorWindow((1, 10, 64, 64), stride=(1, 10, 32, 32))]
    )

    mode = 'decoded'
    if mode == 'decoded':
        final_terrain_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(6, None, None),
            f=lambda ctx, pred_tensor: torch.squeeze(pred_tensor[:, :-1] / pred_tensor[:, -1:], dim=0),
            output_window=TensorWindow((6, 512, 512), stride=(6, 512, 512)),
            args=(decoded_terrain_infinite,),
            args_windows=[TensorWindow((1, 7, 512, 512), stride=(1, 7, 512, 512), dimension_map=(None, 0, 1, 2))]
        )
    elif mode == 'lowres':
        def decode_lowres(ctx, pred_tensor):
            lowres = torch.squeeze(pred_tensor[:, 4:5] * 2, dim=0)
            lowres = lowres * 38.6 - 31.4
            return lowres
        
        final_terrain_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, None, None),
            f=decode_lowres,
            output_window=TensorWindow((1, 512, 512), stride=(1, 512, 512)),
            args=(pred_x0_infinite,),
            args_windows=[TensorWindow((1, 5, 512, 512), stride=(1, 5, 512, 512), dimension_map=(None, 0, 1, 2))]
        )
    elif mode == 'conditional':
        def decode_cond(ctx, cond_tensor):
            elev = cond_tensor[:1] * 2435 - 2607
            elev = torch.sign(elev) * torch.sqrt(torch.abs(elev))
            return elev
        
        final_terrain_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, None, None),
            f=decode_cond,
            output_window=TensorWindow((1, 64, 64)),
            args=(cond_inputs_infinite,),
            args_windows=[TensorWindow((6, 64, 64), stride=(6, 64, 64))]
        )
    elif mode == 'latent':
        final_terrain_infinite = tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, None, None),
            f=lambda ctx, pred_tensor: torch.squeeze(pred_tensor[:, 0], dim=0),  # Take just first latent channel
            output_window=TensorWindow((1, 64, 64)),
            args=(pred_x0_infinite,),
            args_windows=[TensorWindow((1, 5, 64, 64), stride=(1, 5, 64, 64), dimension_map=(None, 0, 1, 2))]
        )

    # Mark cond_inputs_infinite for cleanup since we're done with diffusion steps
    cond_inputs_infinite.mark_for_cleanup()

    # Mark pred_x0_infinite for cleanup since it's only used by decoded_terrain_infinite
    pred_x0_infinite.mark_for_cleanup()
    
    # Mark decoded_terrain_infinite for cleanup since it's only used by final_terrain_infinite
    decoded_terrain_infinite.mark_for_cleanup()
    
    return final_terrain_infinite