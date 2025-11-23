"""Bayesian Optimization sweep for consistency model hyperparameters using decoder image metrics (KID or FID)."""
import json
import click
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna

from ema_pytorch import PostHocEMA
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import laplacian_decode

def _normalize_uint8_three_channel(images: torch.Tensor) -> torch.Tensor:
    """Normalize single-channel images to uint8 [0, 255] repeated to 3 channels."""
    image_min = torch.amin(images, dim=(1, 2, 3), keepdim=True)
    image_max = torch.amax(images, dim=(1, 2, 3), keepdim=True)
    image_range = torch.maximum(image_max - image_min, torch.tensor(255.0, device=images.device))
    image_mid = (image_min + image_max) / 2
    normalized = torch.clamp(((images - image_mid) / image_range + 0.5) * 255, 0, 255)
    return normalized.repeat(1, 3, 1, 1).to(torch.uint8)

def _get_weights(size):
    s = size
    mid = (s - 1) / 2
    y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
    epsilon = 1e-3
    distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y - mid).float() / mid, 0, 1)
    distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x - mid).float() / mid, 0, 1)
    return (distance_y * distance_x)[None, None, :, :]

def evaluate_decoder_kid(model, scheduler, config, val_dataloader=None, kid_n_images=None, accelerator=None, tile_size=128, intermediate_sigma=1.0, metric='kid'):
    """Compute image metric (KID or FID) for consistency decoder using 2-step consistency sampling."""
    metric = (metric or 'kid').lower()
    pbar = tqdm(total=kid_n_images, desc=f"Calculating Decoder {metric.upper()}")
    if metric == 'kid':
        image_metric = KernelInceptionDistance(normalize=True,).to(accelerator.device)
    elif metric == 'fid':
        image_metric = FrechetInceptionDistance(normalize=True,).to(accelerator.device)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'kid' or 'fid'.")
    generator = torch.Generator(device=accelerator.device).manual_seed(548)
    val_dataloader_iter = iter(val_dataloader)

    sigma_data = config['sweep_dataset']['sigma_data']
    init_t = torch.tensor(np.arctan(scheduler.sigmas[0] / sigma_data), device=accelerator.device)
    intermediate_t = torch.tensor(np.arctan(intermediate_sigma / sigma_data), device=accelerator.device)


    weights = _get_weights(tile_size).to(accelerator.device)
    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            batch = recursive_to(next(val_dataloader_iter), device=accelerator.device)
            images = batch['image']
            lowfreq = batch['lowfreq']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            assert not conditional_inputs

            output = torch.zeros(images.shape, device=accelerator.device)
            output_weights = torch.zeros(images.shape, device=accelerator.device)
            num_tiles = (images.shape[-1] - tile_size // 2) // (tile_size // 2)

            # 2-step consistency sampling with tiling
            for i in range(num_tiles):
                for j in range(num_tiles):
                    samples = torch.zeros(
                        (images.shape[0], images.shape[1], tile_size, tile_size),
                        device=accelerator.device
                    )
                    tile_cond_img = cond_img[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2]
                    
                    # First timestep: start from high noise
                    t_values = [init_t, intermediate_t]
                    
                    for t_scalar in t_values:
                        t = t_scalar.view(1, 1, 1, 1).expand(samples.shape[0], 1, 1, 1).to(device=images.device, dtype=images.dtype)
                        z = torch.randn(samples.shape, generator=generator, device=images.device) * sigma_data
                        x_t = torch.cos(t) * samples + torch.sin(t) * z
                        model_input = x_t / sigma_data
                        if tile_cond_img is not None:
                            model_input = torch.cat([model_input, tile_cond_img], dim=1)
                        pred = -model(model_input, noise_labels=t.flatten(), conditional_inputs=[])
                        samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
                    
                    output[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2] += samples * weights
                    output_weights[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2] += weights

            output = output / output_weights / sigma_data
            images = images / sigma_data
            
            output_full = laplacian_decode(output * config['sweep_dataset']['residual_std'] + config['sweep_dataset']['residual_mean'], lowfreq, extrapolate=True)
            images_full = laplacian_decode(images * config['sweep_dataset']['residual_std'] + config['sweep_dataset']['residual_mean'], lowfreq, extrapolate=True)
            
            output_full = torch.sign(output_full) * torch.square(output_full)
            images_full = torch.sign(images_full) * torch.square(images_full)
            
            with torch.autocast(device_type=accelerator.device.type, enabled=False):
                image_metric.update(_normalize_uint8_three_channel(output_full), real=False)
                image_metric.update(_normalize_uint8_three_channel(images_full), real=True)

            pbar.update(images.shape[0])

    pbar.close()
    if metric == 'kid':
        kid_mean, _ = image_metric.compute()
        return kid_mean.item()
    else:
        fid = image_metric.compute()
        return fid.item()


@click.command()
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True,
              help='Path to the consistency decoder configuration file')
@click.option('--n-trials', type=int, default=100, help='Number of Bayesian optimization trials')
@click.option('--study-name', type=str, default=None, help='Name for the Optuna study (for resuming)')
@click.option('--storage', is_flag=True, default=False, help='Enable persistent Optuna storage under save_dir')
def main(config_path, n_trials, study_name, storage):
    """Run Bayesian Optimization over hyperparameters to minimize decoder image metric (KID or FID) for consistency models."""
    build_registry()

    # Load config
    config = Config().from_disk(config_path)

    # Basic checks and setup
    save_dir = config['logging']['save_dir']
    phema_folder = os.path.join(save_dir, 'phema')
    if not os.path.exists(phema_folder):
        raise ValueError(f"PHEMA folder not found: {phema_folder}")

    # Resolve full config
    resolved = registry.resolve(config, validate=False)

    # Model and scheduler
    model = EDMUnet2D.from_pretrained(resolved['model']['main_path'])
    scheduler = resolved['scheduler']

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=1,
    )
    device = accelerator.device

    # Validation dataloader
    val_dataset = resolved['sweep_dataset']

    # EMA setup
    ema_config = resolved['ema'].copy()
    ema_config['checkpoint_folder'] = phema_folder
    ema = PostHocEMA(model, **ema_config).to(device)
    model = model.to(device)
    model = accelerator.prepare(model)

    # Evaluation parameters
    kid_n_images = int(config['sweep']['kid_n_images'])
    tile_size = int(config['sweep']['tile_size'])
    
    # Metric selection
    metric_name = str(config['sweep'].get('metric', 'kid')).lower()
    if metric_name not in ('kid', 'fid'):
        raise ValueError(f"Invalid sweep metric '{metric_name}'. Expected 'kid' or 'fid'.")

    min_ema_sigma = config['sweep']['min_ema_sigma']
    max_ema_sigma = config['sweep']['max_ema_sigma']
    min_ema_step = config['sweep']['min_ema_step']
    max_ema_step = config['sweep']['max_ema_step']
    min_intermediate_sigma = config['sweep']['min_intermediate_sigma']
    max_intermediate_sigma = config['sweep']['max_intermediate_sigma']

    print(f"Loaded config from: {config_path}")
    print(f"Using PHEMA folder: {phema_folder}")
    print(f"Sweeping metric: {metric_name.upper()}")
    print(f"Search range: EMA σ_rel ∈ [{min_ema_sigma}, {max_ema_sigma}]")
    print(f"Search range: EMA step ∈ [{min_ema_step}, {max_ema_step}]")
    print(f"Search range: intermediate σ ∈ [{min_intermediate_sigma}, {max_intermediate_sigma}]")

    # Objective function
    def objective(trial):
        ema_sigma = trial.suggest_float('ema_sigma', min_ema_sigma, max_ema_sigma, log=False)
        ema_step = trial.suggest_int('ema_step', min_ema_step, max_ema_step, log=True)
        intermediate_sigma = trial.suggest_float('intermediate_sigma', min_intermediate_sigma, max_intermediate_sigma, log=True)

        print("\n" + "=" * 80)
        print(f"Trial {trial.number}: EMA σ = {ema_sigma:.6f}, EMA step = {ema_step}, intermediate_sigma = {intermediate_sigma:.6f}")
        print("=" * 80)

        ema_model = ema.synthesize_ema_model(sigma_rel=ema_sigma, step=ema_step)
        ema_model.copy_params_from_ema_to_model()

        val_dataloader = DataLoader(
            LongDataset(val_dataset, shuffle=True, seed=958),
            batch_size=config['sweep']['batch_size']
        )
        metric_score = evaluate_decoder_kid(
            model=model,
            scheduler=scheduler,
            config=config,
            val_dataloader=val_dataloader,
            kid_n_images=kid_n_images,
            accelerator=accelerator,
            tile_size=tile_size,
            intermediate_sigma=intermediate_sigma,
            metric=metric_name,
        )
        del val_dataloader
        print(f"Result: {metric_name.upper()} = {metric_score:.6f}")
        return metric_score

    # Study setup
    if study_name is None:
        study_name = f"consistency_decoder_{metric_name}_sweep_{os.path.basename(save_dir)}"

    print("\n" + "=" * 80)
    print("Starting Bayesian Optimization with Optuna")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Sweeping metric: {metric_name.upper()}")
    print(f"Search range: EMA σ ∈ [{min_ema_sigma}, {max_ema_sigma}]")
    print(f"Search range: EMA step ∈ [{min_ema_step}, {max_ema_step}]")
    print(f"Search range: intermediate_sigma ∈ [{min_intermediate_sigma}, {max_intermediate_sigma}]")
    
    # Configure Optuna storage
    if storage:
        storage_dir = os.path.join(save_dir, 'storage')
        os.makedirs(storage_dir, exist_ok=True)
        storage_url = f"sqlite:///{os.path.join(storage_dir, 'optuna.db')}"
    else:
        storage_url = None
    if storage_url:
        print(f"Storage: {storage_url}")
    print("=" * 80 + "\n")

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, 
                                           n_startup_trials=config['sweep'].get('n_startup_trials', 24),
                                           multivariate=True,
                                           group=True,
                                           n_ei_candidates=config['sweep'].get('n_ei_candidates', 64),
                                           prior_weight=config['sweep'].get('prior_weight', 1.0)),
    )

    # Optionally force baseline trials
    baseline_first = bool(config['sweep'].get('baseline_first_trial', True))
    if baseline_first and len(study.trials) == 0:
        baseline_params = {
            'ema_sigma': 0.05,
            'ema_step': min_ema_step,
            'intermediate_sigma': max_intermediate_sigma,
        }
        print("Enqueuing baseline first trial:", baseline_params)
        study.enqueue_trial(baseline_params)

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Best results
    best_trial = study.best_trial
    optimal_ema_sigma = float(best_trial.params['ema_sigma'])
    optimal_ema_step = int(best_trial.params['ema_step'])
    optimal_intermediate_sigma = float(best_trial.params['intermediate_sigma'])
    optimal_score = float(best_trial.value)

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print(f"Optimal EMA σ: {optimal_ema_sigma:.6f}")
    print(f"Optimal EMA step: {optimal_ema_step}")
    print(f"Optimal intermediate_sigma: {optimal_intermediate_sigma:.6f}")
    print(f"Optimal {metric_name.upper()}: {optimal_score:.6f}")
    print(f"Best trial: {best_trial.number}")
    print("=" * 80)

    # Save results
    results = {
        'metric': metric_name,
        'optimal_ema_sigma': optimal_ema_sigma,
        'optimal_ema_step': optimal_ema_step,
        'optimal_intermediate_sigma': optimal_intermediate_sigma,
        'optimal_score': optimal_score,
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'ema_sigma': t.params.get('ema_sigma'),
                'ema_step': t.params.get('ema_step'),
                'intermediate_sigma': t.params.get('intermediate_sigma'),
                'score': t.value,
                'state': t.state.name,
            }
            for t in study.trials
        ],
        'config_path': config_path,
        'save_dir': save_dir,
        'study_name': study_name,
    }

    output_path = os.path.join(save_dir, 'optuna_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()


