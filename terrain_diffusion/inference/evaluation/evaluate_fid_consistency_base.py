"""
Script for generating and evaluating base samples using Fréchet Inception Distance (FID).

This script provides functionality to:
1. Load pre-trained diffusion models (main and guidance models)
2. Generate synthetic samples using specified configurations
3. Calculate Fréchet Inception Distance to assess sample quality and diversity
4. Support flexible evaluation parameters like guidance scale, sampling steps, and batch processing

Typical Use:
python evaluate_sr_fid.py --main-config path/to/main_config.cfg \
                           --guide-config path/to/guide_config.cfg \
                           --guidance-scale 1.0 \
                           --max-samples 32768
"""


import os
import click
import torch
import numpy as np
from tqdm import tqdm
from confection import Config, registry
from ema_pytorch import PostHocEMA
from torch.utils.data import DataLoader
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.inference.evaluation.utils import get_dataloader
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
from terrain_diffusion.training.utils import recursive_to
from PIL import Image

def create_models_consistency(main_resolved, main_sigma_rel=0.05, save_models=False):
    """
    Create and initialize the main and guidance models from config files.
    
    Args:
        main_resolved (str): Path to main model config file
        guide_resolved (str, optional): Path to guidance model config file. Defaults to None.
        main_sigma_rel (float, optional): EMA sigma_rel for main model. Defaults to 0.05.
        guide_sigma_rel (float, optional): EMA sigma_rel for guidance model. Defaults to 0.05.
        guide_ema_step (int, optional): EMA step for guidance model. Defaults to None.
        save_models (bool, optional): Save the models after loading. Defaults to False.
    
    Returns:
        tuple: (main_model, guide_model)
    """
    # Initialize models
    model_m = EDMUnet2D.from_pretrained(main_resolved['model']['main_path'])
    
    # Apply EMA for main model
    phema_m_dir = f"{main_resolved['logging']['save_dir']}/phema"
    assert os.path.exists(phema_m_dir), f"Error: The phema directory {phema_m_dir} does not exist."
    main_resolved['ema']['checkpoint_folder'] = phema_m_dir
    ema_m = PostHocEMA(model_m, **main_resolved['ema'])
    ema_m.load_state_dict(torch.load(f"{main_resolved['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
    ema_m.synthesize_ema_model(sigma_rel=main_sigma_rel).copy_params_from_ema_to_model()
    
    if save_models:
        checkpoint_path = main_resolved['logging']['save_dir']
        save_path = os.path.join(checkpoint_path, 'saved_model')
        model_m.save_pretrained(save_path)
        print(f'Saved main model to {save_path}.')
    
    # Move models to device and compile
    model_m = torch.compile(model_m)
    
    return model_m

def evaluate_models_fid_consistency_base(base_model,
                                         ae_model, 
                                         dataloader, 
                                         intermediate_timesteps=[1.1],
                                         scale_timesteps=True,
                                         num_samples=2048,
                                         save_samples_dir=None, 
                                         log_samples=None, 
                                         dtype=torch.float32):
    """     
    Evaluate consistency models using Fréchet Inception Distance (FID).
    
    Args:
        model: Consistency model
        dataloader: Data loader for validation dataset
        intermediate_timesteps (list, optional): List of intermediate timesteps for consistency sampling. Defaults to [1.1].
        scale_timesteps (bool, optional): Whether to scale timesteps to match input distribution. Defaults to True.
        num_samples (int, optional): Maximum number of samples to evaluate. Defaults to 2048.
        save_samples_dir (str, optional): Directory to save generated samples. Defaults to None.
        log_samples (int, optional): Number of samples between logging. Defaults to num_samples.
        dtype (torch.dtype, optional): Data type for inputs. Defaults to torch.float32.
    
    Returns:
        float: Final FID score
    """
    if log_samples is None:
        log_samples = num_samples
    
    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature=2048)
    fid_metric = fid_metric.to(base_model.device)
    
    sigma_data = 0.5  # Standard deviation of data distribution
    
    # Define timesteps for consistency sampling
    intermediate_timesteps = list(intermediate_timesteps)
    intermediate_timesteps.sort(reverse=True)
    timesteps = torch.as_tensor([np.arctan(80/0.5), *intermediate_timesteps], dtype=torch.float32, device=base_model.device)
    
    # Generate samples and calculate FID
    last_log = 0
    with torch.no_grad():
        samples_generated = 0
        data_iter = iter(dataloader)
        
        pbar = tqdm(total=num_samples, desc="Generating Samples", unit="samples")
        
        while samples_generated < num_samples:
            batch = next(data_iter)
            batch = recursive_to(batch, base_model.device)
            
            # Get conditional inputs
            cond_img = batch.get('cond_img')
            cond_inputs = batch.get('cond_inputs')
            
            # Initialize with random noise
            z = torch.randn_like(batch['image']) * sigma_data
            pred_x0 = torch.zeros_like(z)
            
            first_t = True
            # Consistency sampling loop
            for t in timesteps:
                t = t.view(1, 1, 1, 1).expand(batch['image'].shape[0], 1, 1, 1)
                if scale_timesteps and not first_t:
                    sigma = sigma_data * torch.tan(t)
                    sigma = sigma * (torch.std(batch['image'], dim=(1, 2, 3), keepdim=True) / sigma_data)
                    t = torch.atan(sigma / sigma_data)
                else:
                    first_t = False
                x_t = torch.cos(t) * pred_x0 + torch.sin(t) * z
                
                # Prepare model input
                model_input = torch.cat([x_t / sigma_data, cond_img], dim=1)
                
                # Get model predictions
                with torch.autocast(device_type="cuda", dtype=dtype):
                    pred = -base_model(model_input, noise_labels=t.flatten(), conditional_inputs=cond_inputs)
                
                pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
            
            def decode(samples):
                samples = samples * 2
                latent = samples[:, :4]
                lowfreq = samples[:, 4:5]
                decoded = ae_model.decode(latent)
                residual, watercover = decoded[:, :1], decoded[:, 1:2]
                watercover = torch.sigmoid(watercover)
                residual = dataloader.dataset.denormalize_residual(residual, 90)
                lowfreq = dataloader.dataset.denormalize_lowfreq(lowfreq, 90)
                residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
                decoded_terrain = laplacian_decode(residual, lowfreq)
                return decoded_terrain
            
            # Process and update FID metrics
            real_samples = decode(batch['image'])
            samples = decode(pred_x0)
            real_min = torch.amin(real_samples, dim=(1, 2, 3), keepdim=True)
            real_max = torch.amax(real_samples, dim=(1, 2, 3), keepdim=True)
            
            value_range = torch.maximum(real_max - real_min, torch.tensor(0.1))
            value_mid = (real_min + real_max) / 2
            
            # Normalize and process samples
            samples_norm = torch.clamp(((samples - value_mid) / value_range + 0.5) * 255, 0, 255)
            samples_norm = samples_norm.repeat(1, 3, 1, 1)
            samples_norm = samples_norm.to(torch.uint8)
            
            real_norm = torch.clamp(((real_samples - value_mid) / value_range + 0.5) * 255, 0, 255)
            real_norm = real_norm.repeat(1, 3, 1, 1)
            real_norm = real_norm.to(torch.uint8)
            
            # Update FID metric
            fid_metric.update(samples_norm, real=False)
            fid_metric.update(real_norm, real=True)
            
            if samples_generated - last_log >= log_samples:
                fid_score = fid_metric.compute().item()
                print(f"Interim FID Score ({samples_generated}/{num_samples}): {fid_score}")
                last_log += log_samples
            
            # Save samples if requested
            if save_samples_dir is not None:
                os.makedirs(save_samples_dir, exist_ok=True)
                for i, (sample, real_sample) in enumerate(zip(samples_norm, real_norm)):
                    fake_filename = os.path.join(save_samples_dir, f'{samples_generated + i:06d}_fake.png')
                    real_filename = os.path.join(save_samples_dir, f'{samples_generated + i:06d}_real.png')
                    
                    Image.fromarray(sample.squeeze().cpu().numpy().transpose(1, 2, 0)).save(fake_filename)
                    Image.fromarray(real_sample.squeeze().cpu().numpy().transpose(1, 2, 0)).save(real_filename)
            
            samples_generated += batch['image'].shape[0]
            pbar.update(batch['image'].shape[0])
        
        pbar.close()
        
        # Calculate final FID score
        final_fid_score = fid_metric.compute().item()
        print(f"Final FID Score ({samples_generated}/{num_samples}): {final_fid_score}")
        return final_fid_score

@click.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True, help="Path to model config")
@click.option("-ae", type=click.Path(exists=True), required=True, help="Path to autoencoder")
@click.option("--sigma-rel", type=float, default=0.05, help="EMA sigma_rel (default: 0.05)")
@click.option("--timestep", type=float, multiple=True, default=(1.1,), help="List of intermediate timesteps for consistency sampling (default: [1.1])")
@click.option("--scale-timesteps", is_flag=True, help="Scale timesteps to match input distribution (default: False)")
@click.option("--num-samples", type=int, default=2048, help="Number of samples to generate (Default: 2048)")
@click.option("--batch-size", type=int, default=64, help="Batch size for generation (default: 64)")
@click.option("--log-samples", type=int, default=None, help="Number of samples between logs (Default: num_samples)")
@click.option("--save-samples-dir", type=click.Path(file_okay=False, writable=True), help="Directory to save samples", default=None)
@click.option("--save-model", is_flag=True, help="Save model to disk (default: False)", default=False)
@click.option("--device", type=str, default='cuda', help="Device to run evaluation on (default: 'cuda')")
@click.option("--dtype", type=str, default='float32', help="Data type for inputs (default: 'float32')")
def evaluate_consistency_fid_cli(config,
                                 sigma_rel,
                                 timestep,
                                 scale_timesteps,
                                 num_samples,
                                 batch_size,
                                 log_samples,
                                 save_samples_dir,
                                 save_model,
                                 device,
                                 dtype):
    build_registry()
    model_cfg = Config().from_disk(config)
    
    model_resolved = registry.resolve(model_cfg, validate=False)
    
    dataloader = get_dataloader(model_resolved, batch_size)
    model = create_models_consistency(model_resolved, 
                                         sigma_rel, 
                                         save_model)
    model = model.to(device)
    
    ae = EDMAutoencoder.from_pretrained('checkpoints/models/autoencoder').to(device)
    
    # Run evaluation
    evaluate_models_fid_consistency_base(
        model,
        ae,
        dataloader,
        timestep,
        scale_timesteps,
        num_samples,
        save_samples_dir,
        log_samples,
        {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype]
    )

if __name__ == "__main__":
    evaluate_consistency_fid_cli()