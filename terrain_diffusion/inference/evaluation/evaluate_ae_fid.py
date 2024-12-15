"""
Script for evaluating autoencoder reconstruction quality using Fréchet Inception Distance (FID).

This script provides functionality to:
1. Load pre-trained autoencoder model
2. Generate reconstructions using the decoder
3. Calculate Fréchet Inception Distance to assess reconstruction quality
4. Support flexible evaluation parameters and batch processing

Typical Use:
python evaluate_ae_fid.py --config path/to/autoencoder_config.cfg \
                         --sigma-rel 0.05 \
                         --max-samples 32768
"""

import os
import click
import torch
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
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to autoencoder config")
@click.option("--sigma-rel", type=float, default=0.05, help="EMA sigma_rel for model")
@click.option("--ema-step", type=int, default=None, help="EMA step for model")
@click.option("--log-samples", type=int, default=2048, help="Number of samples to generate between logs (minimum 2048)")
@click.option("--max-samples", type=int, default=2048*16, help="Max number of samples to generate")
@click.option("--batch-size", type=int, default=64, help="Batch size for generation")
@click.option("--cpu", is_flag=True, help="Use CPU", default=False)
@click.option("--use-wandb", is_flag=True, help="Enable wandb logging")
@click.option("--save-samples-dir", type=click.Path(file_okay=False, writable=True), help="Directory to save generated samples", default=None)
@click.option("--fp16", is_flag=True, help="Use FP16", default=False)
def evaluate_ae_fid(config, sigma_rel, ema_step, log_samples, max_samples, batch_size, cpu, use_wandb, save_samples_dir, fp16):
    """Generate reconstructions using autoencoder and calculate FID score."""
    device = 'cuda' if not cpu else 'cpu'
    
    # Load configs and build registry
    build_registry()
    cfg = Config().from_disk(config)
    
    # Initialize wandb
    wandb.init(project=cfg['wandb']['project'],
               mode='offline' if not use_wandb else 'online',
               job_type='eval-ae-fid',
               config={
                    'config': config,
                    'sigma_rel': sigma_rel,
                    'ema_step': ema_step,
                    'log_samples': log_samples,
                    'max_samples': max_samples,
                    'batch_size': batch_size,
                    'fp16': fp16
               })
    
    # Resolve configs to get models and dataset
    resolved = registry.resolve(cfg, validate=False)
    
    # Initialize model
    model = resolved['model']
    
    # Apply EMA
    phema_dir = f"{cfg['logging']['save_dir']}/phema"
    assert os.path.exists(phema_dir), f"Error: The phema directory {phema_dir} does not exist."
    resolved['ema']['checkpoint_folder'] = phema_dir
    ema = PostHocEMA(model, **resolved['ema'])
    ema.load_state_dict(torch.load(f"{cfg['logging']['save_dir']}/latest_checkpoint/phema.pt"))
    ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    
    # Initialize dataset and dataloader
    val_dataset = resolved['val_dataset']
    dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), batch_size=batch_size,
                          **resolved['dataloader_kwargs'])
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048)
    fid.to(device)
    
    model = model.to(device)
    model = torch.compile(model)
    
    # Generate reconstructions and calculate FID
    last_log = 0
    with torch.no_grad():
        samples_generated = 0
        data_iter = iter(dataloader)
        
        # Create a progress bar
        pbar = tqdm(total=max_samples, desc="Generating Reconstructions", unit="samples")
        
        while samples_generated < max_samples:
            batch = next(data_iter)
            batch = recursive_to(batch, device)
            
            # Get inputs
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs', [])
            
            # Scale images
            sigma_data = cfg['training']['sigma_data']
            scaled_images = images / sigma_data
            if cond_img is not None:
                scaled_images = torch.cat([scaled_images, cond_img], dim=1)
            
            # Generate reconstructions
            with torch.autocast(device_type='cuda', dtype=torch.float16 if fp16 else torch.float32):
                enc_mean, enc_logvar = model.preencode(scaled_images, conditional_inputs)
                z = model.postencode(enc_mean, enc_logvar, use_mode=True)
                reconstructions = model.decode(z) * sigma_data
            
            # Normalize to 0-255 range for FID calculation
            real_samples = images
            real_min = torch.amin(real_samples, dim=(1, 2, 3), keepdim=True)
            real_max = torch.amax(real_samples, dim=(1, 2, 3), keepdim=True)
            
            value_range = torch.maximum(real_max - real_min, torch.tensor(0.1))
            value_mid = (real_min + real_max) / 2
            
            # Normalize samples to the adjusted range
            recon_norm = torch.clamp(((reconstructions - value_mid) / value_range + 0.5) * 255, 0, 255)
            recon_norm = recon_norm.repeat(1, 3, 1, 1)  # Convert to RGB
            recon_norm = recon_norm.to(torch.uint8)
            
            # Update FID metric
            fid.update(recon_norm, real=False)
            
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
            
            # Save samples if requested
            if save_samples_dir is not None:
                os.makedirs(save_samples_dir, exist_ok=True)
                for i, (recon, real) in enumerate(zip(recon_norm, real_norm)):
                    recon_filename = os.path.join(save_samples_dir, f'{samples_generated + i:06d}_recon.png')
                    real_filename = os.path.join(save_samples_dir, f'{samples_generated + i:06d}_real.png')
                    
                    from PIL import Image
                    Image.fromarray(recon.squeeze().cpu().numpy().transpose(1, 2, 0)).save(recon_filename)
                    Image.fromarray(real.squeeze().cpu().numpy().transpose(1, 2, 0)).save(real_filename)
            
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
    evaluate_ae_fid()