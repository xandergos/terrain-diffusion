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
from terrain_diffusion.inference.evaluation.utils import get_dataloader
from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to
from PIL import Image


def create_models(main_resolved, guide_resolved=None, main_sigma_rel=0.05, guide_sigma_rel=0.05, 
                  guide_ema_step=None, save_models=False):
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
    model_m = main_resolved['model']
    model_g = guide_resolved['model'] if guide_resolved else None
    
    # Apply EMA for main model
    phema_m_dir = f"{main_resolved['logging']['save_dir']}/phema"
    assert os.path.exists(phema_m_dir), f"Error: The phema directory {phema_m_dir} does not exist."
    main_resolved['ema']['checkpoint_folder'] = phema_m_dir
    ema_m = PostHocEMA(model_m, **main_resolved['ema'])
    ema_m.load_state_dict(torch.load(f"{main_resolved['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
    ema_m.synthesize_ema_model(sigma_rel=main_sigma_rel).copy_params_from_ema_to_model()
    
    # Apply EMA for guidance model if it exists
    if guide_resolved:
        phema_g_dir = f"{guide_resolved['logging']['save_dir']}/phema"
        assert os.path.exists(phema_g_dir), f"Error: The phema directory {phema_g_dir} does not exist."
        guide_resolved['ema']['checkpoint_folder'] = phema_g_dir
        ema_g = PostHocEMA(model_g, **guide_resolved['ema'])
        ema_g.load_state_dict(torch.load(f"{guide_resolved['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
        ema_g.synthesize_ema_model(sigma_rel=guide_sigma_rel, step=guide_ema_step).copy_params_from_ema_to_model()
    
    if save_models:
        checkpoint_path = main_resolved['logging']['save_dir']
        save_path = os.path.join(checkpoint_path, 'saved_model')
        model_m.save_pretrained(save_path)
        print(f'Saved main model to {save_path}.')

        if guide_resolved:
            checkpoint_path = guide_resolved['logging']['save_dir']
            save_path = os.path.join(checkpoint_path, 'saved_model')
            model_g.save_pretrained(save_path)
            print(f'Saved guidance model to {save_path}.')
    
    # Move models to device and compile
    model_m = torch.compile(model_m)
    if model_g:
        model_g = torch.compile(model_g)
    
    return model_m, model_g

def evaluate_models_fid(model_m, 
                        model_g, 
                        scheduler, 
                        dataloader, 
                        num_samples=2048,
                        guidance_scale=1.0, 
                        scheduler_steps=15, 
                        save_samples_dir=None, 
                        log_samples=None, 
                        dtype=torch.float32):
    """
    Evaluate models using Fréchet Inception Distance (FID).
    
    Args:
        model_m: Main diffusion model
        model_g: Guidance model (can be None)
        scheduler: Diffusion scheduler
        dataloader: Data loader for validation dataset
        guidance_scale (float, optional): Scale for guidance. Defaults to 1.0.
        num_samples (int, optional): Maximum number of samples to evaluate. Defaults to 2048 (minimum).
        batch_size (int, optional): Batch size for evaluation. Defaults to 64.
        scheduler_steps (int, optional): Number of steps for scheduler. Defaults to 32.
        log_samples (int, optional): Number of samples between logging (if enabled). Defaults to num_samples.
        save_samples_dir (str, optional): Directory to save generated samples. Defaults to None.
        dtype (torch.dtype, optional): Data type for inputs. Defaults to torch.float32.
    Returns:
        float: Final FID score
    """
    if log_samples is None:
        log_samples = num_samples
    
    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature=2048)
    fid_metric = fid_metric.to(model_m.device)
    
    # Generate samples and calculate FID
    last_log = 0
    with torch.no_grad():
        samples_generated = 0
        data_iter = iter(dataloader)
        
        pbar = tqdm(total=num_samples, desc="Generating Samples", unit="samples")
        
        while samples_generated < num_samples:
            batch = next(data_iter)
            batch = recursive_to(batch, model_m.device)
            
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
                with torch.autocast(device_type="cuda", dtype=dtype):
                    if not model_g or guidance_scale == 1.0:
                        model_output = model_m(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                    else:
                        model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                        model_output_g = model_g(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                        model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
                
                samples = scheduler.step(model_output, t, samples).prev_sample
            
            # Process and update FID metrics
            real_samples = batch['image']
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
@click.option("--main-config", type=click.Path(exists=True), required=True, help="Path to main model config (default: required)")
@click.option("--guide-config", type=click.Path(exists=True), required=False, help="Path to guidance model config (default: None)", default=None)
@click.option("--main-sigma-rel", type=float, default=0.05, help="EMA sigma_rel for main model (default: 0.05)")
@click.option("--guide-sigma-rel", type=float, default=0.05, help="EMA sigma_rel for guidance model (default: 0.05)")
@click.option("--guide-ema-step", type=int, default=None, help="EMA step for guidance model (default: None)")
@click.option("--guidance-scale", type=float, default=1.0, help="Guidance scale (default: 1.0)")
@click.option("--num-samples", type=int, default=2048, help="Number of samples to generate (Default and minimum 2048)")
@click.option("--batch-size", type=int, default=64, help="Batch size for generation (default: 64)")
@click.option("--scheduler-steps", type=int, default=15, help="Number of steps for scheduler (default: 15)")
@click.option("--log-samples", type=int, default=None, help="Number of samples to generate between logs (Default: num_samples)")
@click.option("--save-samples-dir", type=click.Path(file_okay=False, writable=True), help="Directory to save generated samples (default: None)", default=None)
@click.option("--save-models", is_flag=True, help="Save models to disk (default: False)", default=False)
@click.option("--device", type=str, default='cuda', help="Device to run evaluation on (default: 'cuda')")
@click.option("--dtype", type=str, default='float32', help="Data type for inputs (default: 'float32')")
def evaluate_sr_fid_cli(main_config, 
                        guide_config, 
                        main_sigma_rel, 
                        guide_sigma_rel, 
                        guide_ema_step, 
                        guidance_scale, 
                        num_samples, 
                        batch_size, 
                        scheduler_steps, 
                        log_samples, 
                        save_samples_dir, 
                        save_models, 
                        device, 
                        dtype):
    build_registry()
    main_cfg = Config().from_disk(main_config)
    guide_cfg = Config().from_disk(guide_config) if guide_config else None
    
    main_resolved = registry.resolve(main_cfg, validate=False)
    guide_resolved = registry.resolve(guide_cfg, validate=False) if guide_cfg else None
    
    dataloader = get_dataloader(main_resolved, batch_size)
    model_m, model_g = create_models(main_resolved, 
                                     guide_resolved, 
                                     main_sigma_rel, 
                                     guide_sigma_rel, 
                                     guide_ema_step, 
                                     save_models)
    model_m = model_m.to(device)
    if model_g:
        model_g = model_g.to(device)
        
    scheduler = main_resolved['scheduler']
    
    evaluate_models_fid(
        model_m, 
        model_g, 
        scheduler, 
        dataloader, 
        num_samples, 
        guidance_scale, 
        batch_size, 
        scheduler_steps, 
        save_samples_dir, 
        log_samples, 
        {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype],
        enable_logging=True)


if __name__ == "__main__":
    evaluate_sr_fid_cli()