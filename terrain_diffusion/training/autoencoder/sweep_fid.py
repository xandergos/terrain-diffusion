import json
import os
import random
import click
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from confection import Config, registry
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.training.save_model import load_model_from_checkpoint
from terrain_diffusion.training.registry import build_registry


def calculate_fid(model, dataloader, device, num_samples):
    """
    Calculate FID score for the autoencoder.
    
    Args:
        model: The autoencoder model
        dataloader: DataLoader for validation data
        device: Device to run on
        num_samples: Number of samples to process
        
    Returns:
        FID score as float
    """
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    model = model.to(device)
    model.eval()
    
    pbar = tqdm(total=num_samples, desc="Calculating FID")
    dataloader_iter = iter(dataloader)
    samples_processed = 0
    
    with torch.no_grad():
        while samples_processed < num_samples:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            images = batch['image'].to(device)
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            if cond_img is not None:
                cond_img = cond_img.to(device)
            if conditional_inputs is not None:
                conditional_inputs = conditional_inputs.to(device)
            
            batch_size = images.shape[0]
            
            # Prepare input
            scaled_clean_images = images
            if cond_img is not None:
                scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
            
            # Encode and decode
            z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
            z = model.postencode(z_means, z_logvars)
            decoded_x, logvar = model.decode(z, include_logvar=True)
            
            # Extract first channel (residual/elevation)
            pred_residual = decoded_x[:, :1]
            real_residual = scaled_clean_images[:, :1]
            
            # Normalize to [0, 255] uint8 for FID
            real_min = torch.amin(real_residual, dim=(1, 2, 3), keepdim=True)
            real_max = torch.amax(real_residual, dim=(1, 2, 3), keepdim=True)
            value_range = torch.maximum(real_max - real_min, torch.tensor(1.0, device=real_residual.device))
            value_mid = (real_min + real_max) / 2
            
            samples_norm = torch.clamp(((pred_residual - value_mid) / value_range + 0.5) * 255, 0, 255)
            samples_norm = samples_norm.repeat(1, 3, 1, 1).to(torch.uint8)
            
            real_norm = torch.clamp(((real_residual - value_mid) / value_range + 0.5) * 255, 0, 255)
            real_norm = real_norm.repeat(1, 3, 1, 1).to(torch.uint8)
            
            # Update FID metric
            fid_metric.update(samples_norm, real=False)
            fid_metric.update(real_norm, real=True)
            
            samples_processed += batch_size
            pbar.update(batch_size)
    
    pbar.close()
    
    # Compute FID
    fid_score = fid_metric.compute().item()
    return fid_score


@click.command()
@click.option('-c', '--checkpoint', 'checkpoint_path', required=True, type=click.Path(exists=True),
              help='Path to the autoencoder checkpoint directory')
@click.option('-n', '--num-runs', type=int, required=True, help='Number of FID evaluations to run')
@click.option('--min-step', type=int, required=True, help='Minimum EMA step')
@click.option('--max-step', type=int, required=True, help='Maximum EMA step')
@click.option('--min-sigma', type=float, required=True, help='Minimum sigma_rel value')
@click.option('--max-sigma', type=float, required=True, help='Maximum sigma_rel value')
@click.option('--num-samples', type=int, default=16384, help='Number of samples to use for FID calculation')
@click.option('--batch-size', type=int, default=64, help='Batch size for evaluation')
@click.option('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
@click.option('--seed', type=int, default=None, help='Random seed for sampling step/sigma')
def main(checkpoint_path, num_runs, min_step, max_step, min_sigma, max_sigma, num_samples, batch_size, device, seed):
    """
    Sweep FID scores for different PHEMA configurations.
    
    This script loads an autoencoder checkpoint, randomly samples EMA step and sigma_rel
    values within specified ranges multiple times, loads the model with PHEMA, and calculates FID score.
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Load training config from checkpoint
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Build registry and resolve config
    build_registry()
    config = Config(config_dict)
    
    # Resolve only the dataset and dataloader parts
    keys_to_keep = ['val_dataset', 'dataloader_kwargs']
    minimal_config = {}
    for key in keys_to_keep:
        if key in config:
            minimal_config[key] = config[key]
    
    resolved = registry.resolve(Config(minimal_config), validate=False)
    
    # Create validation dataset and dataloader
    val_dataset = resolved['val_dataset']
    print(f"Loaded validation dataset with {len(val_dataset)} samples")
    
    dataloader_kwargs = resolved.get('dataloader_kwargs', {})
    val_dataloader = DataLoader(
        LongDataset(val_dataset, shuffle=True),
        batch_size=batch_size,
        **dataloader_kwargs,
        drop_last=True
    )
    
    # Track best result
    best_fid = float('inf')
    best_config = None
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"Starting FID sweep with {num_runs} runs")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step range: [{min_step}, {max_step}]")
    print(f"Sigma range: [{min_sigma}, {max_sigma}]")
    print(f"{'='*80}\n")
    
    # Run sweep
    for run_idx in range(num_runs):
        # Sample random step and sigma
        ema_step = random.randint(min_step, max_step)
        sigma_rel = random.uniform(min_sigma, max_sigma)
        
        print(f"\n[Run {run_idx + 1}/{num_runs}]")
        print(f"  Sampled EMA step: {ema_step}")
        print(f"  Sampled sigma_rel: {sigma_rel:.6f}")
        
        # Load model with PHEMA
        print(f"  Loading model...")
        model = load_model_from_checkpoint(checkpoint_path, ema_step=ema_step, sigma_rel=sigma_rel)
        
        # Calculate FID
        print(f"  Calculating FID with {num_samples} samples...")
        fid_score = calculate_fid(model, val_dataloader, device, num_samples)
        
        # Update best
        if fid_score < best_fid:
            best_fid = fid_score
            best_config = {
                'ema_step': ema_step,
                'sigma_rel': sigma_rel,
                'fid_score': fid_score
            }
        
        # Store result
        result = {
            'run': run_idx + 1,
            'ema_step': ema_step,
            'sigma_rel': sigma_rel,
            'fid_score': fid_score
        }
        all_results.append(result)
        
        # Print result
        print(f"  FID: {fid_score:.4f}")
        print(f"  Best FID so far: {best_fid:.4f} (step={best_config['ema_step']}, sigma={best_config['sigma_rel']:.6f})")
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FID Sweep Complete!")
    print(f"{'='*80}")
    print(f"\nBest Configuration:")
    print(f"  EMA step: {best_config['ema_step']}")
    print(f"  Sigma rel: {best_config['sigma_rel']:.6f}")
    print(f"  FID score: {best_config['fid_score']:.4f}")
    
    print(f"\nAll Results:")
    for result in all_results:
        print(f"  Run {result['run']}: FID={result['fid_score']:.4f} (step={result['ema_step']}, sigma={result['sigma_rel']:.6f})")
    
    # Create scatter plot
    print(f"\nCreating visualization...")
    steps = [r['ema_step'] for r in all_results]
    sigmas = [r['sigma_rel'] for r in all_results]
    fids = [r['fid_score'] for r in all_results]
    
    plt.figure(figsize=(10, 8))
    
    # Use logarithmic normalization for the colorbar
    norm = mcolors.LogNorm(vmin=min(fids), vmax=max(fids))
    scatter = plt.scatter(steps, sigmas, c=fids, cmap='viridis', s=100, alpha=0.7, 
                         edgecolors='black', linewidth=1, norm=norm)
    
    # Mark the best configuration with a star
    plt.scatter([best_config['ema_step']], [best_config['sigma_rel']], 
                marker='*', s=500, c='red', edgecolors='black', linewidth=2, 
                label=f'Best (FID={best_config["fid_score"]:.4f})', zorder=5)
    
    plt.xlabel('EMA Step', fontsize=12)
    plt.ylabel('Sigma Rel', fontsize=12)
    plt.title(f'FID Sweep Results ({num_runs} runs)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='FID Score (log scale)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('experiment_data', exist_ok=True)
    plot_file = 'experiment_data/ae_fid_sweep.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.close()
    
    # Save results to file
    sweep_results = {
        'checkpoint': checkpoint_path,
        'num_runs': num_runs,
        'num_samples': num_samples,
        'step_range': [min_step, max_step],
        'sigma_range': [min_sigma, max_sigma],
        'best_config': best_config,
        'all_results': all_results
    }
    
    results_file = 'experiment_data/ae_fid_sweep_runs.json'
    with open(results_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

