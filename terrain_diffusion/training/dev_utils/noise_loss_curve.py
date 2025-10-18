import json
import click
import numpy as np
import torch
from accelerate import Accelerator
from confection import Config, registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.registry import build_registry
from tqdm import tqdm
from torch.utils.data import DataLoader
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.training.utils import *
import matplotlib.pyplot as plt
import os


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True, help="Path to the configuration file")
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=True, help="Path to a checkpoint (folder) to load model from")
@click.option("--override", "-o", multiple=True, help="Override config values (format: key.subkey=value)")
@click.option("--steps", default=50, help="Number of noise levels to test")
@click.option("--samples", default=128, help="Number of samples to evaluate per noise level")
@click.pass_context
def main(ctx, config_path, ckpt_path, override, steps, samples):
    build_registry()
    
    config = Config().from_disk(config_path) if config_path else None
    
    # Handle overrides
    all_overrides = list(override)
    for param in ctx.args:
        if param.startswith('--'):
            key, value = param.lstrip('-').split('=', 1)
            all_overrides.append(f"{key}={value}")
    
    for o in all_overrides:
        key_path, value = o.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        set_nested_value(config, key_path, value, o)
    
    resolved = registry.resolve(config, validate=False)
    
    model = resolved['model']
    assert isinstance(model, EDMUnet2D), "Currently only EDMUnet2D is supported."
    
    val_dataset = resolved['val_dataset']
    scheduler = resolved['scheduler']
    
    # Load model weights
    from safetensors.torch import load_file
    model_state_dict = load_file(os.path.join(ckpt_path, "model.safetensors"))
    model.load_state_dict(model_state_dict, strict=False)
    del model_state_dict
    
    accelerator = Accelerator(mixed_precision=resolved['training']['mixed_precision'])
    
    val_dataloader = DataLoader(
        LongDataset(val_dataset, shuffle=True), 
        batch_size=config['training']['batch_size'],
        **resolved['dataloader_kwargs'], 
        drop_last=True
    )
    
    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    
    def calc_loss(pred_v_t, v_t, logvar, sigma_data):
        loss = (pred_v_t - v_t) ** 2 / sigma_data**2
        if 'loss_groups' not in config['training']:
            return loss.mean()
        loss_groups = []
        c = 0
        for group_channels in config['training']['loss_groups']:
            loss_groups.append(loss[:, c:c+group_channels].mean())
            c += group_channels
        return torch.stack(loss_groups).mean()
    
    # Generate noise levels
    sigma_min = scheduler.config.sigma_min
    sigma_max = scheduler.config.sigma_max
    sigma_data = scheduler.config.sigma_data
    
    sigmas = torch.logspace(np.log10(sigma_min), np.log10(sigma_max), steps)
    losses = []
    
    model.eval()
    val_dataloader_iter = iter(val_dataloader)
    
    print(f"Evaluating noise loss curve with {steps} noise levels...")
    
    # Preload a fixed set of batches to reuse across all noise levels
    fixed_batches = []
    samples_collected = 0
    while samples_collected < samples:
        try:
            batch = next(val_dataloader_iter)
        except StopIteration:
            val_dataloader_iter = iter(val_dataloader)
            batch = next(val_dataloader_iter)
        fixed_batches.append(batch)
        samples_collected += batch['image'].shape[0]
    
    for sigma in tqdm(sigmas, desc="Noise levels"):
        sigma_losses = []
        for batch in fixed_batches:
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            
            batch_size = images.shape[0]
            sigma_batch = sigma.expand(batch_size, 1, 1, 1).to(images.device)
            
            t = torch.atan(sigma_batch / sigma_data)
            cnoise = t.flatten()
            
            noise = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * noise
            
            x = x_t / sigma_data
            if cond_img is not None:
                x = torch.cat([x, cond_img], dim=1)
            
            with torch.no_grad(), accelerator.autocast():
                model_output, logvar = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs, return_logvar=True)
                pred_v_t = -sigma_data * model_output
            
            v_t = torch.cos(t) * noise - torch.sin(t) * images
            loss = calc_loss(pred_v_t, v_t, logvar, sigma_data)
            
            sigma_losses.append(loss.item())
        losses.append(np.mean(sigma_losses))
    
    # Plot the curve
    plt.figure(figsize=(10, 6))
    plt.loglog(sigmas.numpy(), losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Loss')
    plt.title('Loss vs Noise Level')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('noise_loss_curve.png')


if __name__ == '__main__':
    main()
