import os

import torch
from tqdm import tqdm
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.datasets.datasets import H5LatentsDataset, H5LatentsSimpleDataset
from terrain_diffusion.training.gan.generator import MPGenerator
from ema_pytorch import PostHocEMA
from safetensors.torch import load_model
import matplotlib.pyplot as plt
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
from terrain_diffusion.inference.scheduler.functional_dpmsolver import multistep_dpm_solver_second_order_update, dpm_solver_first_order_update, precondition_outputs
from matplotlib.widgets import Slider
from infinite_tensors.infinite_tensors import InfiniteTensor, TensorWindow

def create_unbounded_pipe(sigmas: list[int] = None, cond_input_scaling: float = 1.0):
    if sigmas is None:
        sigmas = [80, 10, 1]
    
    def get_weights(size):
        s = size
        mid = (s - 1) / 2
        y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
        epsilon = 1e-3
        distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y - mid).float() / mid, 0, 1)
        distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x - mid).float() / mid, 0, 1)
        return (distance_y * distance_x)[None, None, :, :]

    weights64 = get_weights(64)

    def get_model(cls, checkpoint_path, sigma_rel=None, ema_step=None, device='cpu'):
        config_path = os.path.join(checkpoint_path, 'model_config')
        model = cls.from_config(cls.load_config(config_path))

        if sigma_rel is not None:
            # sigma_rels are placeholders since we dont use them
            ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], checkpoint_folder=os.path.join(checkpoint_path, '..', 'phema')).to(device)
            ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
        else:
            load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))

        return model.to(device)

    def latent_generator(ctx):
        """Generate infinite latent noise for GAN input"""
        seed = (ctx[-1] * 16777619 + ctx[-2] * 373587943) & 0x7fffffff
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(1, 128, 128, 128, generator=generator)  # Return on CPU

    cond_inputs_size = round(64 * cond_input_scaling)
    def cond_inputs_generator(ctx, latent_tensor):
        """Generate conditional inputs from latent using GAN"""
        # Get the latent window and move to device for processing
        latent_window = latent_tensor.to(device)
        with torch.no_grad():
            cond_result = gan.raw_forward(latent_window)[:, :, 2:-2, 2:-2]
        
        # Interpolate to window size (64x64 window from 14x14 latent)
        cond_interpolated = torch.nn.functional.interpolate(cond_result, size=(cond_inputs_size, cond_inputs_size), mode='nearest')
        return cond_interpolated[0].cpu()  # Return on CPU

    def noise_generator(ctx):
        """Generate infinite noise tensor"""
        return torch.randn(1, 5, 64, 64) * sigma_data  # Return on CPU

    def process_model_step(ctx, x_t_tensor, cond_inputs_tensor, cnoise_val):
        """Process a single model step on infinite tensors"""
        # Move inputs to device for processing
        x_t_window = x_t_tensor.to(device)
        cond_window = cond_inputs_tensor.to(device)
        
        # Extract conditional features from the window
        thresholded = (torch.amax(cond_window[5], dim=[-1, -2]) < -1).long()
        
        weighted_cond_inputs = torch.sum(cond_window * torch.sigmoid(cond_window[5:]), dim=[-1, -2]) \
            / (torch.sum(torch.sigmoid(cond_window[5:]), dim=[-1, -2]) + 1e-8)
        mean_elev = cond_window[0] * 2435 - 2607
        mean_elev = (torch.sign(mean_elev) * torch.sqrt(torch.abs(mean_elev)) + 31.4) / 38.6
        mean_elev = torch.quantile(torch.flatten(mean_elev), q=0.99)
        mean_temp = weighted_cond_inputs[1] * (1 - thresholded.float())
        std_temp = weighted_cond_inputs[2] * (1 - thresholded.float())
        mean_prec = weighted_cond_inputs[3] * (1 - thresholded.float())
        std_prec = weighted_cond_inputs[4] * (1 - thresholded.float())
        all_water = thresholded
        
        # Run model
        with torch.no_grad():
            model_output = model(x_t_window / sigma_data,
                                noise_labels=cnoise_val.to(device).expand(x_t_window.shape[0]),
                                conditional_inputs=[mean_elev.to(device).view(1).expand(x_t_window.shape[0])])
        
        # Return on CPU
        model_output = model_output.cpu() * weights64
        model_output = torch.cat([model_output, weights64], dim=1)
        return model_output

    device = 'cuda'
    torch.no_grad().__enter__()
    gan = get_model(MPGenerator, 'checkpoints/gan/latest_checkpoint', sigma_rel=0.05, device=device)
    model = get_model(EDMUnet2D, 'checkpoints/consistency_base-192x3/latest_checkpoint', sigma_rel=0.05, device=device)
    autoencoder = EDMAutoencoder.from_pretrained('checkpoints/models/autoencoder').to(device)

    dataset = H5LatentsDataset('dataset.h5', 64, [[0.1, 1.0]], [90], [1], eval_dataset=False,
                            latents_mean=[0, 0, 0, 0],
                            latents_std=[1, 1, 1, 1],
                            sigma_data=0.5,
                            split="train",
                            beauty_dist=[[1, 1, 1, 1, 1]])

    sigma_data = 0.5

    # Create infinite latent tensor
    latent_infinite = InfiniteTensor(
        shape=(1, 128, None, None),
        f=latent_generator,
        output_window=TensorWindow((1, 128, 128, 128))
    )

    # Create infinite conditional inputs tensor
    cond_inputs_infinite = InfiniteTensor(
        shape=(6, None, None),
        f=cond_inputs_generator,
        output_window=TensorWindow((6, cond_inputs_size, cond_inputs_size)),
        args=(latent_infinite,),
        args_windows=[TensorWindow((1, 128, 14, 14), window_stride=(1, 128, 5, 5), dimension_map=(None, 0, 1, 2))]
    )

    # Mark latent_infinite for cleanup since it's only used by cond_inputs_infinite
    latent_infinite.mark_for_cleanup()

    # Initialize infinite prediction tensor
    pred_x0_infinite = InfiniteTensor(
        shape=(1, 5, None, None),
        f=lambda ctx: torch.zeros(1, 5, 64, 64) * sigma_data,  # Return on CPU
        output_window=TensorWindow((1, 5, 64, 64))
    )

    # Process through diffusion steps
    for i, sigma in enumerate(sigmas):
        sigma = torch.tensor(sigma, dtype=torch.float32)
        t = torch.atan(sigma / sigma_data)
        sigma, t = sigma.to(device), t.to(device)
        cnoise = t.expand(1)
        
        # Create noise for this step
        noise_infinite = InfiniteTensor(
            shape=(1, 5, None, None),
            f=noise_generator,
            output_window=TensorWindow((1, 5, 64, 64))
        )
        
        # Create x_t = sin(t) * noise + cos(t) * pred_x0
        def create_x_t(ctx, noise_tensor, pred_tensor):
            return torch.sin(t.cpu()) * noise_tensor + torch.cos(t.cpu()) * pred_tensor
        
        x_t_infinite = InfiniteTensor(
            shape=(1, 5, None, None),
            f=create_x_t,
            output_window=TensorWindow((1, 5, 64, 64)),
            args=(noise_infinite, pred_x0_infinite),
            args_windows=[TensorWindow((1, 5, 64, 64)), TensorWindow((1, 5, 64, 64))]
        )
        
        # Mark noise_infinite for cleanup since it's only used by x_t_infinite
        noise_infinite.mark_for_cleanup()
        
        # Create model output tensor
        def model_step_wrapper(ctx, x_t_tensor, cond_tensor):
            return process_model_step(ctx, x_t_tensor, cond_tensor, cnoise)
        
        model_output_infinite = InfiniteTensor(
            shape=(1, 6, None, None),
            f=model_step_wrapper,
            output_window=TensorWindow((1, 6, 64, 64), window_stride=(1, 6, 32, 32)),
            args=(x_t_infinite, cond_inputs_infinite),
            args_windows=[TensorWindow((1, 5, 64, 64), window_stride=(1, 5, 32, 32)), TensorWindow((6, 64, 64), window_stride=(6, 32, 32), dimension_map=(1, 2, 3))]
        )
        
        # Mark x_t_infinite for cleanup since it's only used by model_output_infinite
        x_t_infinite.mark_for_cleanup()
        
        model_output_infinite = InfiniteTensor(
            shape=(1, 5, None, None),
            f=lambda ctx, x_t_tensor: x_t_tensor[:, :-1] / x_t_tensor[:, -1:],
            output_window=TensorWindow((1, 5, 64, 64)),
            args=(model_output_infinite,),
            args_windows=[TensorWindow((1, 6, 64, 64))]
        )
        
        # Update pred_x0 = cos(t) * x_t + sin(t) * model_output * sigma_data
        def update_pred_x0(ctx, x_t_tensor, model_output_tensor):
            return torch.cos(t.cpu()) * x_t_tensor + torch.sin(t.cpu()) * model_output_tensor * sigma_data
        
        new_pred_x0_infinite = InfiniteTensor(
            shape=(1, 5, None, None),
            f=update_pred_x0,
            output_window=TensorWindow((1, 5, 64, 64)),
            args=(x_t_infinite, model_output_infinite),
            args_windows=[TensorWindow((1, 5, 64, 64)), TensorWindow((1, 5, 64, 64))]
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
        with torch.no_grad():
            decoded = autoencoder.decode(latent)
        residual, watercover = decoded[:, :1], decoded[:, 1:2]
        watercover = torch.sigmoid(watercover)
        residual = dataset.denormalize_residual(residual, 90)
        lowfreq = dataset.denormalize_lowfreq(lowfreq, 90)
        residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
        decoded_terrain = laplacian_decode(residual, lowfreq)
        decoded_terrain = torch.cat([decoded_terrain.cpu() * weights512, weights512], dim=1)
        return decoded_terrain

    decoded_terrain_infinite = InfiniteTensor(
        shape=(1, 2, None, None),
        f=decode_samples,
        output_window=TensorWindow((1, 2, 512, 512), window_stride=(1, 2, 256, 256)),
        args=(pred_x0_infinite,),
        args_windows=[TensorWindow((1, 5, 64, 64), window_stride=(1, 5, 32, 32))]
    )


    mode = 'decoded'
    if mode == 'decoded':
        final_terrain_infinite = InfiniteTensor(
            shape=(None, None),
            f=lambda ctx, pred_tensor: torch.squeeze(pred_tensor[:, 0:1] / pred_tensor[:, 1:2], dim=(0, 1)),
            output_window=TensorWindow((512, 512), window_stride=(512, 512)),
            args=(decoded_terrain_infinite,),
            args_windows=[TensorWindow((1, 2, 512, 512), window_stride=(1, 2, 512, 512), dimension_map=(None, None, 0, 1))]
        )
    elif mode == 'lowres':
        def decode_lowres(ctx, pred_tensor):
            lowres = torch.squeeze(pred_tensor[:, 4:5] * 2, dim=(0, 1))
            lowres = lowres * 38.6 - 31.4
            return lowres
        
        final_terrain_infinite = InfiniteTensor(
            shape=(None, None),
            f=decode_lowres,
            output_window=TensorWindow((512, 512), window_stride=(512, 512)),
            args=(pred_x0_infinite,),
            args_windows=[TensorWindow((1, 5, 512, 512), window_stride=(1, 5, 512, 512), dimension_map=(None, None, 0, 1))]
        )
    elif mode == 'conditional':
        def decode_cond(ctx, cond_tensor):
            elev = cond_tensor[0] * 2435 - 2607
            elev = torch.sign(elev) * torch.sqrt(torch.abs(elev))
            return elev
        
        final_terrain_infinite = InfiniteTensor(
            shape=(None, None),
            f=decode_cond,
            output_window=TensorWindow((64, 64)),
            args=(cond_inputs_infinite,),
            args_windows=[TensorWindow((6, 64, 64), window_stride=(6, 64, 64), dimension_map=(None, 0, 1))]
        )
    elif mode == 'latent':
        final_terrain_infinite = InfiniteTensor(
            shape=(None, None),
            f=lambda ctx, pred_tensor: torch.squeeze(pred_tensor[:, 0], dim=0),  # Take just first latent channel
            output_window=TensorWindow((64, 64)),
            args=(pred_x0_infinite,),
            args_windows=[TensorWindow((1, 5, 64, 64), window_stride=(1, 5, 64, 64), dimension_map=(None, None, 0, 1))]
        )

    # Mark cond_inputs_infinite for cleanup since we're done with diffusion steps
    cond_inputs_infinite.mark_for_cleanup()

    # Mark pred_x0_infinite for cleanup since it's only used by decoded_terrain_infinite
    pred_x0_infinite.mark_for_cleanup()
    
    # Mark decoded_terrain_infinite for cleanup since it's only used by final_terrain_infinite
    decoded_terrain_infinite.mark_for_cleanup()
    
    return final_terrain_infinite