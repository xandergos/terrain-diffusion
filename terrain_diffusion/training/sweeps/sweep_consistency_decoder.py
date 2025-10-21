"""Bayesian Optimization sweep for consistency model hyperparameters using decoder KID metric."""
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
from optuna.exceptions import TrialPruned
import math

from ema_pytorch import PostHocEMA
from torchmetrics.image.kid import KernelInceptionDistance

from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import recursive_to

def _normalize_uint8_three_channel(images: torch.Tensor) -> torch.Tensor:
    """Normalize single-channel images to uint8 [0, 255] repeated to 3 channels."""
    image_min = torch.amin(images, dim=(1, 2, 3), keepdim=True)
    image_max = torch.amax(images, dim=(1, 2, 3), keepdim=True)
    image_range = torch.maximum(image_max - image_min, torch.tensor(1.0, device=images.device))
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

def evaluate_decoder_kid(model, scheduler, val_dataloader=None, kid_n_images=None, accelerator=None, trial=None, check_interval=None, prune_probability_threshold=None, tile_size=128, intermediate_sigma=1.0, sigma_data=None):
    """Compute KID for consistency decoder using 2-step consistency sampling with optional interim pruning."""
    pbar = tqdm(total=kid_n_images, desc="Calculating Decoder KID")
    kid = KernelInceptionDistance(normalize=True,).to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(548)
    val_dataloader_iter = iter(val_dataloader)

    init_t = torch.tensor(np.arctan(scheduler.sigmas[0] / sigma_data), device=accelerator.device)
    intermediate_t = torch.tensor(np.arctan(intermediate_sigma / sigma_data), device=accelerator.device)

    def _norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _maybe_prune(cur_mean, cur_std):
        if trial is None or prune_probability_threshold is None:
            return False
        study = trial.study
        if study is None:
            return False
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if t.number == trial.number:
                continue
            other_mean = t.value
            other_std = t.user_attrs.get('kid_std')
            if other_mean is None or other_std is None:
                continue
            denom = math.sqrt(cur_std * cur_std + float(other_std) * float(other_std))
            z = -(cur_mean - float(other_mean)) / denom
            p_cur_less = _norm_cdf(z)
            if p_cur_less < prune_probability_threshold:
                return True
        return False

    weights = _get_weights(tile_size).to(accelerator.device)
    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            batch = recursive_to(next(val_dataloader_iter), device=accelerator.device)
            images = batch['image']
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
                        t = t_scalar.view(1, 1, 1, 1).expand(samples.shape[0], 1, 1, 1).to(images.device)
                        z = torch.randn(samples.shape, generator=generator, device=images.device) * sigma_data
                        x_t = torch.cos(t) * samples + torch.sin(t) * z
                        model_input = x_t / sigma_data
                        if tile_cond_img is not None:
                            model_input = torch.cat([model_input, tile_cond_img], dim=1)
                        pred = -model(model_input, noise_labels=t.flatten(), conditional_inputs=[])
                        samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
                    
                    output[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2] += samples * weights
                    output_weights[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2] += weights

            output = output / output_weights
            samples = output

            kid.update(_normalize_uint8_three_channel(samples / sigma_data), real=False)
            kid.update(_normalize_uint8_three_channel(images / sigma_data), real=True)

            pbar.update(images.shape[0])

            # Interim pruning checks
            if check_interval and (pbar.n % check_interval == 0 or pbar.n >= pbar.total):
                cur_mean_t, cur_std_t = kid.compute()
                cur_mean = float(cur_mean_t.item())
                cur_std = float(cur_std_t.item())
                if trial is not None:
                    trial.report(cur_mean, step=pbar.n)
                if _maybe_prune(cur_mean, cur_std):
                    return cur_mean_t.item(), cur_std_t.item()

    pbar.close()
    kid_mean, kid_std = kid.compute()
    return kid_mean.item(), kid_std.item()


@click.command()
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True,
              help='Path to the consistency decoder configuration file')
@click.option('--n-trials', type=int, default=100, help='Number of Bayesian optimization trials')
@click.option('--study-name', type=str, default=None, help='Name for the Optuna study (for resuming)')
@click.option('--storage', is_flag=True, default=False, help='Enable persistent Optuna storage under save_dir')
def main(config_path, n_trials, study_name, storage):
    """Run Bayesian Optimization over hyperparameters to minimize decoder KID for consistency models."""
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
    model = resolved['model']
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
    sigma_data = float(config['training'].get('sigma_data', 1.0))
    
    # Pruning parameters
    intermediate_steps = int(config['sweep'].get('intermediate_steps', 1024))
    prune_probability_threshold = float(config['sweep'].get('prune_probability_threshold', 0.05))

    min_ema_sigma = config['sweep']['min_ema_sigma']
    max_ema_sigma = config['sweep']['max_ema_sigma']
    min_ema_step = config['sweep']['min_ema_step']
    max_ema_step = config['sweep']['max_ema_step']
    min_intermediate_sigma = config['sweep']['min_intermediate_sigma']
    max_intermediate_sigma = config['sweep']['max_intermediate_sigma']

    print(f"Loaded config from: {config_path}")
    print(f"Using PHEMA folder: {phema_folder}")
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
        kid_mean, kid_std = evaluate_decoder_kid(
            model=model,
            scheduler=scheduler,
            val_dataloader=val_dataloader,
            kid_n_images=kid_n_images,
            accelerator=accelerator,
            trial=trial,
            check_interval=intermediate_steps,
            prune_probability_threshold=prune_probability_threshold,
            tile_size=tile_size,
            intermediate_sigma=intermediate_sigma,
            sigma_data=sigma_data,
        )
        del val_dataloader

        # Store std on trial for future pruning comparisons
        trial.set_user_attr('kid_std', float(kid_std))

        print(f"Result: KID mean = {kid_mean:.6f} (std = {kid_std:.6f})")
        return kid_mean

    # Study setup
    if study_name is None:
        study_name = f"consistency_decoder_kid_sweep_{os.path.basename(save_dir)}"

    print("\n" + "=" * 80)
    print("Starting Bayesian Optimization with Optuna")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Search range: EMA σ ∈ [{min_ema_sigma}, {max_ema_sigma}]")
    print(f"Search range: EMA step ∈ [{min_ema_step}, {max_ema_step}]")
    print(f"Search range: intermediate_t ∈ [{min_intermediate_t}, {max_intermediate_t}]")
    
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
            'intermediate_t': 1.1,
        }
        print("Enqueuing baseline first trial:", baseline_params)
        study.enqueue_trial(baseline_params)
            
        baseline_params = {
            'ema_sigma': 0.05,
            'ema_step': min_ema_step,
            'intermediate_t': min_intermediate_t,
        }
        print("Enqueuing baseline second trial:", baseline_params)
        study.enqueue_trial(baseline_params)

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Best results
    best_trial = study.best_trial
    optimal_ema_sigma = float(best_trial.params['ema_sigma'])
    optimal_ema_step = int(best_trial.params['ema_step'])
    optimal_intermediate_t = float(best_trial.params['intermediate_t'])
    optimal_kid = float(best_trial.value)

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print(f"Optimal EMA σ: {optimal_ema_sigma:.6f}")
    print(f"Optimal EMA step: {optimal_ema_step}")
    print(f"Optimal intermediate_t: {optimal_intermediate_t:.6f}")
    print(f"Optimal KID mean: {optimal_kid:.6f}")
    print(f"Best trial: {best_trial.number}")
    print("=" * 80)

    # Save results
    results = {
        'optimal_ema_sigma': optimal_ema_sigma,
        'optimal_ema_step': optimal_ema_step,
        'optimal_intermediate_t': optimal_intermediate_t,
        'optimal_kid_mean': optimal_kid,
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'ema_sigma': t.params.get('ema_sigma'),
                'ema_step': t.params.get('ema_step'),
                'intermediate_t': t.params.get('intermediate_t'),
                'kid_mean': t.value,
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


