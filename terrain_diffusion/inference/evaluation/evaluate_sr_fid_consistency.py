"""
Script for generating and evaluating super-resolution samples using Fréchet Inception Distance (FID).

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
from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.training.diffusion.registry import build_registry
from terrain_diffusion.training.diffusion.unet import EDMUnet2D
from terrain_diffusion.training.utils import recursive_to

@click.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to consistency distillation config")
@click.option("--sigma-rel", type=float, default=0.05, help="EMA sigma_rel for model")
@click.option("--intermediate-timestep", type=float, default=1.1, help="Intermediate timestep for consistency model")
@click.option("--log-samples", type=int, default=2048, help="Number of samples to generate between logs (minimum 2048)")
@click.option("--max-samples", type=int, default=2048*16, help="Max number of samples to generate")
@click.option("--batch-size", type=int, default=64, help="Batch size for generation")
@click.option("--cpu", is_flag=True, help="Use CPU", default=False)
@click.option("--no-wandb", is_flag=True, help="Disable wandb logging")
@click.option("--save-samples-dir", type=click.Path(file_okay=False, writable=True), help="Directory to save generated samples", default=None)
@click.option("--fp16", is_flag=True, help="Use FP16", default=False)
def evaluate_sr_fid(config, sigma_rel, intermediate_timestep,
                   log_samples, max_samples, batch_size, cpu, no_wandb, save_samples_dir, fp16):
    """Generate samples using consistency model and calculate FID score."""
    # Load configs and build registry
    from accelerate import Accelerator
    
    build_registry()
    cfg = Config().from_disk(config)
    
    # Initialize Accelerator for distributed/mixed precision support
    device = 'cuda' if not cpu else 'cpu'
    
    # Initialize wandb for compatibility with sweeps; don't need to log online
    wandb.init(project=cfg['wandb']['project'],
               mode='offline' if no_wandb else 'online',
               job_type='eval-fid',
               config={
                    'config': config,
                    'sigma_rel': sigma_rel,
                    'intermediate_timestep': intermediate_timestep,
                    'log_samples': log_samples,
                    'max_samples': max_samples,
                    'batch_size': batch_size,
                    'fp16': fp16
               })
    
    # Resolve configs to get models and dataset
    resolved = registry.resolve(cfg, validate=False)
    
    # Initialize model (removed guidance model)
    model = EDMUnet2D.from_config(EDMUnet2D.load_config(cfg['model']['path']))
    
    # Apply EMA
    phema_m_dir = f"{cfg['logging']['save_dir']}/phema"
    assert os.path.exists(phema_m_dir), f"Error: The phema directory {phema_m_dir} does not exist."
    resolved['ema']['checkpoint_folder'] = phema_m_dir
    
    ema = PostHocEMA(model, **resolved['ema'])
    ema.load_state_dict(torch.load(f"{cfg['logging']['save_dir']}/latest_checkpoint/phema.pt"))
    ema.synthesize_ema_model(sigma_rel=sigma_rel).copy_params_from_ema_to_model()
    del ema
    
    # Initialize dataset and dataloader
    val_dataset = resolved['val_dataset']
    dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), batch_size=batch_size,
                            **resolved['dataloader_kwargs'])
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048)
    fid.to(device)
    
    # Prepare models, dataloader, and FID metric with Accelerator
    model = torch.compile(model).to(device)
    
    # Generate samples and calculate FID
    last_log = 0
    with torch.no_grad():
        samples_generated = 0
        data_iter = iter(dataloader)
        
        # Create a progress bar for the entire generation process
        pbar = tqdm(total=max_samples, desc="Generating Samples", unit="samples")
        
        sigma_data = cfg['training']['sigma_data']
        while samples_generated < max_samples:
            batch = next(data_iter)
            batch = recursive_to(batch, device)
            
            cond_img = batch.get('cond_img')
            cond_inputs = batch.get('cond_inputs', [])
            
            # Consistency model sampling
            timesteps = torch.as_tensor([np.arctan(80/0.5), intermediate_timestep], device=device, dtype=torch.float32)
            
            z = torch.randn_like(batch['image']) * sigma_data
            pred_x0 = torch.zeros_like(batch['image'])
            
            for t in timesteps:
                x_t = torch.cos(t) * pred_x0 + torch.sin(t) * z
                t = t.view(1).to(device)
                model_input = torch.cat([x_t / sigma_data, cond_img], dim=1)
                with torch.autocast(device_type=device, dtype=torch.float16 if fp16 else None):
                    pred = -model(model_input, noise_labels=t.expand(batch_size).flatten(), conditional_inputs=cond_inputs)
                pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
            
            samples = pred_x0  # Final samples
            
            # Normalize samples to the range of real samples +/- 10% to allow some error without clipping
            real_samples = batch['image']
            real_min = torch.amin(real_samples, dim=(1, 2, 3), keepdim=True)
            real_max = torch.amax(real_samples, dim=(1, 2, 3), keepdim=True)
            
            # Calculate 10% margin
            margin = 0.1 * (real_max - real_min)
            norm_min = real_min - margin
            norm_max = real_max + margin
            
            # Normalize samples to the adjusted range
            samples_norm = torch.clamp((samples - norm_min) / (norm_max - norm_min) * 255, 0, 255)
            samples_norm = samples_norm.repeat(1, 3, 1, 1)  # Convert to RGB
            samples_norm = samples_norm.to(torch.uint8)
            
            # Update FID metric
            fid.update(samples_norm, real=False)
            
            # Normalize real samples to the same range
            real_norm = torch.clamp((real_samples - norm_min) / (norm_max - norm_min) * 255, 0, 255)
            real_norm = real_norm.repeat(1, 3, 1, 1)  # Convert to RGB
            real_norm = real_norm.to(torch.uint8)
            fid.update(real_norm, real=True)
            
            if samples_generated - last_log >= log_samples:
                fid_score = fid.compute().item()
                wandb.log({
                    "fid": fid_score,
                    "samples_generated": samples_generated
                }, commit=True)
                print(f"Interim FID Score ({samples_generated}/{max_samples}): {fid_score}")
                
                last_log += log_samples
            
            # Save generated samples if save_samples_dir is specified
            if save_samples_dir is not None:
                # Create directory if it doesn't exist
                os.makedirs(save_samples_dir, exist_ok=True)
                
                # Save each sample in the batch
                for i, (sample, real_sample) in enumerate(zip(samples_norm, real_norm)):
                    # Generate unique filenames with leading zeros 
                    fake_filename = os.path.join(save_samples_dir, f'{samples_generated + i:06d}_fake.png')
                    real_filename = os.path.join(save_samples_dir, f'{samples_generated + i:06d}_real.png')
                    
                    # Save as PNG images
                    from PIL import Image
                    Image.fromarray(sample.squeeze().cpu().numpy().transpose(1, 2, 0)).save(fake_filename)
                    Image.fromarray(real_sample.squeeze().cpu().numpy().transpose(1, 2, 0)).save(real_filename)
            
            
            samples_generated += batch_size
            pbar.update(batch_size)
        
        pbar.close()
        
        # Calculate final FID score
        if last_log != samples_generated:
            fid_score = fid.compute().item()
            wandb.log({
                "fid": fid_score,
                "samples_generated": samples_generated
            }, commit=True)
        
        print(f"Final FID Score ({samples_generated}/{max_samples}): {fid_score}")

if __name__ == "__main__":
    evaluate_sr_fid()