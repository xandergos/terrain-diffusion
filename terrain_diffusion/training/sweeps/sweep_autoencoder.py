"""Bayesian Optimization for optimal EMA sigma_rel parameter using Optuna."""
import json
import click
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
from collections import defaultdict
import optuna
import matplotlib.pyplot as plt

from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from ema_pytorch import PostHocEMA


def evaluate_model(model, val_dataloader, perceptual_loss, 
                   validation_steps, kl_weight, mse_weight, perceptual_weight, 
                   accelerator, device='cuda'):
    """Evaluate the model with given EMA parameters."""
    validation_stats = defaultdict(list)
    pbar = tqdm(total=validation_steps, desc=f"Evaluating σ_rel")
    val_dataloader_iter = iter(val_dataloader)
    
    model.eval()
    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            batch = next(val_dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            
            scaled_clean_images = images
            if cond_img is not None:
                scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
            
            z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
            z = model.postencode(z_means, z_logvars)
            decoded_x = model.decode(z)
            
            # MSE loss
            mse_loss = torch.nn.functional.mse_loss(decoded_x, scaled_clean_images)
            
            # Perceptual loss with normalization
            ref_min = torch.amin(scaled_clean_images, dim=(1, 2, 3), keepdim=True)
            ref_max = torch.amax(scaled_clean_images, dim=(1, 2, 3), keepdim=True)
            eps = 0.1
            
            ref_range = torch.maximum((ref_max - ref_min) * 1.1, torch.tensor(eps).to(device))
            ref_center = (ref_min + ref_max) / 2
            
            normalized_ref = ((scaled_clean_images - ref_center) / ref_range * 2)
            normalized_rec = ((decoded_x - ref_center) / ref_range * 2)
            normalized_rec = normalized_rec.clamp(-1, 1)
            
            perceptual_loss_val = perceptual_loss(
                normalized_ref.repeat(1, 3, 1, 1), 
                normalized_rec.repeat(1, 3, 1, 1)
            ).mean()
            
            # Weighted combination
            recon_loss = mse_weight * mse_loss + perceptual_weight * perceptual_loss_val
            
            # KL loss
            ndz_logvars = z_logvars[:, :model.config.latent_channels]
            ndz_means = z_means[:, :model.config.latent_channels]
            kl_loss = -0.5 * torch.mean(1 + ndz_logvars - ndz_means**2 - ndz_logvars.exp())
            
            total_loss = recon_loss + kl_loss * kl_weight
            
            validation_stats['loss'].append(total_loss.item())
            validation_stats['recon_loss'].append(recon_loss.item())
            validation_stats['mse_loss'].append(mse_loss.item())
            validation_stats['perceptual_loss'].append(perceptual_loss_val.item())
            validation_stats['kl_loss'].append(kl_loss.item())
            
            pbar.update(images.shape[0])
            pbar.set_postfix({k: f"{np.nanmean(v):.3f}" for k, v in validation_stats.items()})
    
    pbar.close()
    
    # Return the mean validation reconstruction loss (primary metric to minimize)
    return np.nanmean(validation_stats['recon_loss'])




@click.command()
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True, 
              help='Path to the autoencoder configuration file')
@click.option('--n-trials', type=int, default=20,
              help='Number of Bayesian optimization trials')
@click.option('--validation-steps', type=int, default=None,
              help='Number of validation steps (overrides config)')
@click.option('--study-name', type=str, default=None,
              help='Name for the Optuna study (for resuming)')
@click.option('--storage', type=str, default=None,
              help='Database URL for Optuna storage (for resuming)')
def main(config_path, n_trials, validation_steps, study_name, storage):
    """
    Perform Bayesian Optimization to find optimal EMA sigma_rel value.
    
    This script loads EMA checkpoints from the phema folder specified in the config
    and uses Optuna to search for the optimal EMA sigma_rel parameter that minimizes validation loss.
    """
    build_registry()
    
    # Load config
    config = Config().from_disk(config_path)
    
    # Get sweep parameters
    if 'sweep' not in config:
        raise ValueError("Config must have a [sweep] section with min_ema_sigma and max_ema_sigma")
    
    min_sigma = config['sweep']['min_ema_sigma']
    max_sigma = config['sweep']['max_ema_sigma']
    
    print(f"Loaded config from: {config_path}")
    print(f"Sweep range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
    
    # Get save directory from config
    save_dir = config['logging']['save_dir']
    phema_folder = os.path.join(save_dir, 'phema')
    
    if not os.path.exists(phema_folder):
        raise ValueError(f"PHEMA folder not found: {phema_folder}")
    
    print(f"Using PHEMA folder: {phema_folder}")
    
    # Override validation steps if provided
    if validation_steps is not None:
        config['evaluation']['validation_steps'] = validation_steps
    
    # Resolve config to get datasets and model
    resolved = registry.resolve(config, validate=False)
    
    # Get base model from config (we'll reload it for each trial)
    base_model = resolved['model']
    print(f"Model has {base_model.count_parameters()} parameters")
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=1,
    )
    
    device = accelerator.device
    print(f"Using device: {device}")
    
    # Setup validation dataloader
    val_dataset = resolved['val_dataset']
    val_dataloader = DataLoader(
        LongDataset(val_dataset, shuffle=True),
        batch_size=config['training']['train_batch_size'],
        **resolved['dataloader_kwargs'],
        drop_last=True
    )
    
    # Setup perceptual loss
    perceptual_loss = lpips.LPIPS(net='alex', spatial=True)
    
    # Prepare with accelerator
    val_dataloader, perceptual_loss = accelerator.prepare(val_dataloader, perceptual_loss)
    
    # Get loss weights
    kl_weight = config['training'].get('kl_weight', 0.05)
    mse_weight = config['training'].get('mse_weight', 1.0)
    perceptual_weight = config['training'].get('perceptual_weight', 1.0)
    val_steps = config['evaluation']['validation_steps']
    
    # Setup EMA once
    ema_config = resolved['ema'].copy()
    ema_config['checkpoint_folder'] = phema_folder
    ema = PostHocEMA(base_model, **ema_config)
    
    # Move to device
    base_model = base_model.to(device)
    ema = ema.to(device)
    
    # Prepare model with accelerator
    base_model = accelerator.prepare(base_model)
    
    # Define objective function for Optuna
    def objective(trial):
        """Optuna objective function to minimize."""
        # Suggest sigma_rel value
        sigma_rel = trial.suggest_float('sigma_rel', min_sigma, max_sigma, log=False)
        
        print(f"\n{'='*80}")
        print(f"Trial {trial.number}: Evaluating σ_rel = {sigma_rel:.6f}")
        print(f"{'='*80}")
        
        # Synthesize EMA model with this sigma_rel and copy params to base_model
        ema_model = ema.synthesize_ema_model(sigma_rel=sigma_rel, step=None)
        ema_model.copy_params_from_ema_to_model()
        
        # Evaluate
        loss = evaluate_model(
            base_model, val_dataloader, perceptual_loss,
            val_steps, kl_weight, mse_weight, perceptual_weight,
            accelerator, device
        )
        
        print(f"Result: loss = {loss:.6f}")
        
        return loss
    
    # Create Optuna study
    if study_name is None:
        study_name = f"autoencoder_ema_sweep_{os.path.basename(save_dir)}"
    
    print(f"\n{'='*80}")
    print(f"Starting Bayesian Optimization with Optuna")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Search range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
    if storage:
        print(f"Storage: {storage}")
    print(f"{'='*80}\n")
    
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.GPSampler(seed=42)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Get best results
    best_trial = study.best_trial
    optimal_sigma = best_trial.params['sigma_rel']
    optimal_loss = best_trial.value
    
    print("\n" + "=" * 80)
    print(f"Optimization complete!")
    print(f"Optimal σ_rel: {optimal_sigma:.6f}")
    print(f"Optimal loss: {optimal_loss:.6f}")
    print(f"Best trial: {best_trial.number}")
    print("=" * 80)
    
    # Save results
    results = {
        'optimal_sigma_rel': float(optimal_sigma),
        'optimal_loss': float(optimal_loss),
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'sigma_rel': t.params['sigma_rel'],
                'loss': t.value,
                'state': t.state.name
            }
            for t in study.trials
        ],
        'config_path': config_path,
        'save_dir': save_dir,
        'sweep_range': [min_sigma, max_sigma],
        'study_name': study_name
    }
    
    output_path = os.path.join(save_dir, 'optuna_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print top 5 trials
    print("\nTop 5 trials:")
    print("-" * 60)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"{i}. Trial {trial.number}: σ_rel={trial.params['sigma_rel']:.6f}, loss={trial.value:.6f}")
    print("-" * 60)
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Extract completed trials
    completed_trials = [t for t in study.trials if t.value is not None]
    if len(completed_trials) > 0:
        # Sort by sigma_rel for plotting
        sigma_values = np.array([t.params['sigma_rel'] for t in completed_trials])
        loss_values = np.array([t.value for t in completed_trials])
        sorted_indices = np.argsort(sigma_values)
        sigma_values = sigma_values[sorted_indices]
        loss_values = loss_values[sorted_indices]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(sigma_values, loss_values, 'b-', alpha=0.6, linewidth=2, label='Loss curve')
        plt.scatter(sigma_values, loss_values, c='red', s=100, zorder=5, label='Evaluated trials')
        
        # Mark the optimal point
        best_idx = np.argmin(loss_values)
        plt.scatter(sigma_values[best_idx], loss_values[best_idx], 
                   c='green', s=200, marker='*', zorder=10, 
                   label=f'Optimal (σ={sigma_values[best_idx]:.4f})')
        
        plt.xlabel('EMA σ_rel', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Bayesian Optimization: EMA σ_rel vs Validation Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, 'optuna_sweep_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.close()
    else:
        print("No completed trials to plot.")


if __name__ == '__main__':
    main()

