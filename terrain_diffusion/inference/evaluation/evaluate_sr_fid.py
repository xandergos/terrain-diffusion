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
from terrain_diffusion.training.utils import recursive_to

@click.command()
@click.option("--main-config", type=click.Path(exists=True), required=True, help="Path to main model config")
@click.option("--guide-config", type=click.Path(exists=True), required=False, help="Path to guidance model config", default=None)
@click.option("--main-sigma-rel", type=float, default=0.05, help="EMA sigma_rel for main model")
@click.option("--guide-sigma-rel", type=float, default=0.05, help="EMA sigma_rel for guidance model")
@click.option("--guide-ema-step", type=int, default=None, help="EMA step for guidance model")
@click.option("--guidance-scale", type=float, default=1.0, help="Guidance scale")
@click.option("--log-samples", type=int, default=2048, help="Number of samples to generate between logs (minimum 2048)")
@click.option("--max-samples", type=int, default=2048*16, help="Max number of samples to generate")
@click.option("--batch-size", type=int, default=64, help="Batch size for generation")
@click.option("--cpu", is_flag=True, help="Use CPU", default=False)
@click.option("--scheduler-steps", type=int, default=32, help="Number of steps for scheduler")
@click.option("--use-wandb", is_flag=True, help="Enable wandb logging")
@click.option("--save-samples-dir", type=click.Path(file_okay=False, writable=True), help="Directory to save generated samples", default=None)
@click.option("--fp16", is_flag=True, help="Use FP16", default=False)
@click.option("--save-models", is_flag=True, help="Save models", default=False)
def evaluate_sr_fid_cli(main_config, guide_config, main_sigma_rel, guide_sigma_rel, guide_ema_step, 
         guidance_scale, log_samples, max_samples, batch_size, cpu, scheduler_steps, use_wandb, save_samples_dir, fp16, save_models):
    """Generate samples and calculate FID score."""
    device = 'cuda' if not cpu else 'cpu'
    
    # Load configs and build registry
    build_registry()
    main_cfg = Config().from_disk(main_config)
    if guide_config:
        guide_cfg = Config().from_disk(guide_config)
    else:
        guide_cfg = None
        model_g = None
    
    # Initialize wandb for compatibility with sweeps; don't need to log online
    wandb.init(project=main_cfg['wandb']['project'],
               mode='offline' if not use_wandb else 'online',
               job_type='eval-fid',
               config={
                    'main_config': main_config,
                    'guide_config': guide_config,
                    'main_sigma_rel': main_sigma_rel,
                    'guide_sigma_rel': guide_sigma_rel,
                    'guide_ema_step': guide_ema_step,
                    'guidance_scale': guidance_scale,
                    'log_samples': log_samples,
                    'max_samples': max_samples,
                    'batch_size': batch_size,
                    'scheduler_steps': scheduler_steps,
                    'fp16': fp16
               })
    
    # Resolve configs to get models and dataset
    main_resolved = registry.resolve(main_cfg, validate=False)
    if guide_cfg:
        guide_resolved = registry.resolve(guide_cfg, validate=False)
    
    # Initialize models
    model_m = main_resolved['model']
    if guide_cfg:
        model_g = guide_resolved['model']
    
    # Apply EMA
    phema_m_dir = f"{main_cfg['logging']['save_dir']}/phema"
    assert os.path.exists(phema_m_dir), f"Error: The phema directory {phema_m_dir} does not exist. This is based on the configs logging.save_dir value."
    main_resolved['ema']['checkpoint_folder'] = phema_m_dir
    ema_m = PostHocEMA(model_m, **main_resolved['ema'])
    ema_m.load_state_dict(torch.load(f"{main_cfg['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
    ema_m.synthesize_ema_model(sigma_rel=main_sigma_rel).copy_params_from_ema_to_model()
    
    if guide_cfg:
        phema_g_dir = f"{guide_cfg['logging']['save_dir']}/phema"
        assert os.path.exists(phema_g_dir), f"Error: The phema directory {phema_g_dir} does not exist. This is based on the configs logging.save_dir value."
        guide_resolved['ema']['checkpoint_folder'] = phema_g_dir
        ema_g = PostHocEMA(model_g, **guide_resolved['ema'])
        ema_g.load_state_dict(torch.load(f"{guide_cfg['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
        ema_g.synthesize_ema_model(sigma_rel=guide_sigma_rel, step=guide_ema_step).copy_params_from_ema_to_model()
        
    if save_models:
        checkpoint_path = main_cfg['logging']['save_dir']
        model_m.save_pretrained(os.path.join(checkpoint_path, 'saved_model'))
        save_path = os.path.join(checkpoint_path, 'saved_model')
        print(f'Saved main model to {save_path}.')

        if guide_cfg:
            checkpoint_path = guide_cfg['logging']['save_dir']
            model_g.save_pretrained(os.path.join(checkpoint_path, 'saved_model'))
            save_path = os.path.join(checkpoint_path, 'saved_model')
            print(f'Saved guidance model to {save_path}.')
    
    # Initialize dataset and dataloader
    val_dataset = main_resolved['val_dataset']
    dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), batch_size=batch_size,
                            **main_resolved['dataloader_kwargs'])
    
    # Initialize scheduler
    scheduler = main_resolved['scheduler']
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048)
    fid.to(device)
    
    model_m = model_m.to(device)
    model_m = torch.compile(model_m)
    if model_g:
        model_g = model_g.to(device)
        model_g = torch.compile(model_g)
        
    # Generate samples and calculate FID
    last_log = 0
    with torch.no_grad():
        samples_generated = 0
        data_iter = iter(dataloader)
        
        # Create a progress bar for the entire generation process
        pbar = tqdm(total=max_samples, desc="Generating Samples", unit="samples")
        
        while samples_generated < max_samples:
            batch = next(data_iter)
            batch = recursive_to(batch, device)
            
            # Get conditional inputs
            cond_img = batch.get('cond_img')
            cond_inputs = batch.get('cond_inputs', [])
            
            # Generate samples
            samples = torch.randn_like(batch['image']) * scheduler.sigmas[0]
            
            # Sampling loop
            scheduler.set_timesteps(scheduler_steps)
            for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                t, sigma = t.to(samples.device), sigma.to(samples.device)
                
                scaled_input = scheduler.precondition_inputs(samples, sigma)
                cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))
                
                # Get model predictions
                x = torch.cat([scaled_input, cond_img], dim=1)
                with torch.autocast(device_type='cuda', dtype=torch.float16 if fp16 else torch.float32):
                    if not model_g or guidance_scale == 1.0:
                        model_output = model_m(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                    else:
                        model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                        model_output_g = model_g(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                        model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
                
                samples = scheduler.step(model_output, t, samples).prev_sample
            
            # Normalize samples to the range of real samples +/- 10% to allow some error without clipping
            real_samples = batch['image']
            real_min = torch.amin(real_samples, dim=(1, 2, 3), keepdim=True)
            real_max = torch.amax(real_samples, dim=(1, 2, 3), keepdim=True)
            
            value_range = torch.maximum(real_max - real_min, torch.tensor(0.1))
            value_mid = (real_min + real_max) / 2
            
            # Normalize samples to the adjusted range
            samples_norm = torch.clamp(((samples - value_mid) / value_range + 0.5) * 255, 0, 255)
            samples_norm = samples_norm.repeat(1, 3, 1, 1)  # Convert to RGB
            samples_norm = samples_norm.to(torch.uint8)
            
            # Update FID metric
            fid.update(samples_norm, real=False)
            
            # Normalize real samples to the same range
            real_norm = torch.clamp(((real_samples - value_mid) / value_range + 0.5) * 255, 0, 255)
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


def evaluate_sr_fid(model_m, scheduler, dataloader, accelerator, 
                   max_samples=2048, batch_size=64, scheduler_steps=32):
    """
    Evaluate super-resolution models using Fréchet Inception Distance (FID) without guidance.
    
    Args:
        model_m (nn.Module): Main diffusion model
        scheduler: Diffusion scheduler
        dataloader: Dataloader to evaluate on
        accelerator (Accelerator): Accelerator instance for distributed training
        max_samples (int, optional): Maximum number of samples to evaluate. Defaults to 2048.
        batch_size (int, optional): Batch size for evaluation. Defaults to 64.
        scheduler_steps (int, optional): Number of steps for scheduler. Defaults to 32.
    
    Returns:
        float: Final FID score
    """
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048)
    
    # Generate samples and calculate FID
    with torch.no_grad():
        samples_generated = 0
        data_iter = iter(dataloader)
        
        # Create a progress bar for the entire generation process
        pbar = tqdm(total=max_samples, desc="Generating Samples", unit="sample", 
                    disable=not accelerator.is_main_process)
        
        while samples_generated < max_samples:
            batch = next(data_iter)
            
            # Get conditional inputs
            cond_img = batch.get('cond_img')
            
            # Generate samples
            samples = torch.randn_like(batch['image']) * scheduler.sigmas[0]
            
            # Sampling loop
            scheduler.set_timesteps(scheduler_steps)
            for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                t, sigma = t.to(samples.device), sigma.to(samples.device)
                
                scaled_input = scheduler.precondition_inputs(samples, sigma)
                cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1))
                
                # Get model predictions
                x = torch.cat([scaled_input, cond_img], dim=1)
                with accelerator.autocast():
                    model_output = model_m(x, noise_labels=cnoise, conditional_inputs=[])
                
                samples = scheduler.step(model_output, t, samples).prev_sample
            
            # Normalize samples to the range of real samples +/- 10%
            real_samples = batch['image']
            real_min = torch.amin(real_samples, dim=(1, 2, 3), keepdim=True)
            real_max = torch.amax(real_samples, dim=(1, 2, 3), keepdim=True)
            
            margin = 0.1 * (real_max - real_min)
            norm_min = real_min - margin
            norm_max = real_max + margin
            
            # Normalize samples
            samples_norm = torch.clamp((samples - norm_min) / (norm_max - norm_min) * 255, 0, 255)
            samples_norm = samples_norm.repeat(1, 3, 1, 1)  # Convert to RGB
            samples_norm = samples_norm.to(torch.uint8)
            
            # Update FID metric
            fid.update(samples_norm, real=False)
            
            # Normalize real samples
            real_norm = torch.clamp((real_samples - norm_min) / (norm_max - norm_min) * 255, 0, 255)
            real_norm = real_norm.repeat(1, 3, 1, 1)  # Convert to RGB
            real_norm = real_norm.to(torch.uint8)
            fid.update(real_norm, real=True)
            
            samples_generated += batch_size
            pbar.update(batch_size)
        
        pbar.close()
        
        # Calculate final FID score
        fid_score = fid.compute().item()
        
        return fid_score


if __name__ == "__main__":
    evaluate_sr_fid_cli()