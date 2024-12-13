"""
Script for generating and comparing samples from consistency model with real images.

This script provides functionality to:
1. Load pre-trained diffusion models
2. Generate synthetic samples using specified configurations
3. Plot comparisons between real and generated samples

Typical Use:
python plot_consistency.py --config path/to/config.cfg
"""


import os
import click
import torch
import numpy as np
from tqdm import tqdm
from confection import Config, registry
from ema_pytorch import PostHocEMA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.training.diffusion.registry import build_registry
from terrain_diffusion.training.diffusion.unet import EDMUnet2D
from terrain_diffusion.training.utils import recursive_to

@click.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to consistency distillation config")
@click.option("--sigma-rel", type=float, default=0.05, help="EMA sigma_rel for model")
@click.option("--intermediate-timestep", type=float, default=1.1, help="Intermediate timestep for consistency model")
@click.option("--num-samples", type=int, default=4, help="Number of sample pairs to generate")
@click.option("--batch-size", type=int, default=4, help="Batch size for generation")
@click.option("--cpu", is_flag=True, help="Use CPU", default=False)
@click.option("--save-dir", type=click.Path(file_okay=False, writable=True), help="Directory to save plots")
@click.option("--fp16", is_flag=True, help="Use FP16", default=False)
@click.option("--interactive", is_flag=True, help="Enable interactive plotting", default=False)
def plot_consistency(config, sigma_rel, intermediate_timestep,
                    num_samples, batch_size, cpu, save_dir, fp16, interactive):
    """Generate and plot sample pairs using consistency model."""
    # Load configs and build registry
    build_registry()
    cfg = Config().from_disk(config)
    
    # Initialize Accelerator for distributed/mixed precision support
    device = 'cuda' if not cpu else 'cpu'
    
    # Resolve configs to get models and dataset
    resolved = registry.resolve(cfg, validate=False)
    
    # Initialize model (removed guidance model)
    model = EDMUnet2D.from_config(EDMUnet2D.load_config(cfg['model']['path']))
    
    # Apply EMA
    phema_m_dir = f"{cfg['logging']['save_dir']}/phema"
    assert os.path.exists(phema_m_dir), f"Error: The phema directory {phema_m_dir} does not exist."
    resolved['ema']['checkpoint_folder'] = phema_m_dir
    
    print(f"Loading EMA model from {phema_m_dir}")
    ema = PostHocEMA(model, **resolved['ema'])
    ema.load_state_dict(torch.load(f"{cfg['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
    ema.synthesize_ema_model(sigma_rel=sigma_rel).copy_params_from_ema_to_model()
    del ema
    print(f"EMA model loaded and synthesized from {phema_m_dir}")
    
    # Initialize dataset and dataloader
    val_dataset = resolved['val_dataset']
    
    torch.manual_seed(42)
    np.random.seed(42)
    dataloader = DataLoader(LongDataset(val_dataset, shuffle=False), batch_size=batch_size,
                            **resolved['dataloader_kwargs'])
    
    # Create save directory only if not interactive
    if not interactive:
        if save_dir is None:
            raise ValueError("The --save-dir option is required when interactive is false.")
        os.makedirs(save_dir, exist_ok=True)

    torch.set_num_threads(16)

    with torch.no_grad():
        samples_generated = 0
        data_iter = iter(dataloader)
        
        sigma_data = cfg['training']['sigma_data']
        # Create progress bar for total samples
        pbar = tqdm(total=num_samples, desc="Generating samples")
        
        while samples_generated < num_samples:
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
                with torch.autocast(device_type=device, dtype=torch.float16 if fp16 else torch.float32):
                    pred = -model(model_input, noise_labels=t.expand(batch_size).flatten(), conditional_inputs=cond_inputs)
                pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
            
            samples = pred_x0  # Final samples
            
            # Plot and save comparisons
            for i in range(min(batch_size, num_samples - samples_generated)):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Plot real sample
                ax1.imshow(batch['image'][i].squeeze().cpu().numpy())
                ax1.set_title('Real Sample')
                ax1.axis('off')
                
                # Plot generated sample
                ax2.imshow(samples[i].squeeze().cpu().numpy())
                ax2.set_title('Generated Sample')
                ax2.axis('off')
                
                plt.tight_layout()
                if interactive:
                    plt.show()
                else:
                    plt.savefig(os.path.join(save_dir, f'comparison_{samples_generated + i}.png'))
                    plt.close()
            
            samples_generated += batch_size
            pbar.update(min(batch_size, num_samples - (samples_generated - batch_size)))
        
        pbar.close()

if __name__ == "__main__":
    plot_consistency()