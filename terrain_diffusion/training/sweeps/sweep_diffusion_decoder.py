"""Bayesian Optimization sweep for σ_rel using decoder KID metric."""
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

from terrain_diffusion.data.laplacian_encoder import laplacian_decode
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import recursive_to

#autoencoder = EDMAutoencoder.from_pretrained("checkpoints/models/autoencoder_x8").to('cuda')
#autoencoder.eval()
#autoencoder.requires_grad_(False)

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
    
def evaluate_decoder_kid(model, g_model=None, guidance_scale=1.0, scheduler=None, config=None, val_dataloader=None, kid_n_images=None, kid_scheduler_steps=None, accelerator=None, tile_size=128, metric='kid'):
    """Compute image metric (KID or FID) for the decoder using diffusion sampling on the validation set.
    Returns a single float score."""
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
            initial_noise = torch.randn(images.shape, device=accelerator.device, generator=generator) * scheduler.sigmas[0]
            num_tiles = (images.shape[-1] - tile_size // 2) // (tile_size // 2)

            # Sampling loop
            for i in range(num_tiles):
                for j in range(num_tiles):
                    samples = initial_noise[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2]
                    tile_cond_img = cond_img[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2]
                    scheduler.set_timesteps(kid_scheduler_steps)
                    for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                        t, sigma = t.to(samples.device), sigma.to(samples.device)

                        scaled_input = scheduler.precondition_inputs(samples, sigma)
                        cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))

                        model_input = torch.cat([scaled_input, tile_cond_img], dim=1)
                        if not g_model or guidance_scale == 1.0:
                            model_output = model(model_input, noise_labels=cnoise, conditional_inputs=[])
                        else:
                            model_output_m = model(model_input, noise_labels=cnoise, conditional_inputs=[])
                            model_output_g = g_model(model_input, noise_labels=cnoise, conditional_inputs=[])
                            model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
                        samples = scheduler.step(model_output, t, samples).prev_sample
                    
                    _real = images[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2]
                    _fake = samples
                    #_fake_ae = autoencoder.decode(tile_cond_img[..., ::8, ::8])
                    
                    output[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2] += samples * weights
                    output_weights[..., i*tile_size//2:(i+2)*tile_size//2, j*tile_size//2:(j+2)*tile_size//2] += weights

            output = output / output_weights / config['sweep_dataset']['sigma_data']
            images = images / config['sweep_dataset']['sigma_data']
            
            output_full = laplacian_decode(output * config['sweep_dataset']['residual_std'] + config['sweep_dataset']['residual_mean'], lowfreq, extrapolate=True)
            images_full = laplacian_decode(images * config['sweep_dataset']['residual_std'] + config['sweep_dataset']['residual_mean'], lowfreq, extrapolate=True)
            
            output_full = torch.sign(output_full) * torch.square(output_full)
            images_full = torch.sign(images_full) * torch.square(images_full)

            assert torch.isfinite(output_full).all()
            assert torch.isfinite(images_full).all()

            # Disable autocast for metric calculation to prevent NaN in Inception features
            with torch.autocast(device_type=accelerator.device.type, enabled=False):
                image_metric.update(_normalize_uint8_three_channel(output_full.float()), real=False)
                image_metric.update(_normalize_uint8_three_channel(images_full.float()), real=True)

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
              help='Path to the diffusion decoder configuration file')
@click.option('-gc', '--guide-config', 'guide_config_path', type=click.Path(exists=True), required=False,
              help='Path to the guide model configuration file')
@click.option('--n-trials', type=int, default=100, help='Number of Bayesian optimization trials')
@click.option('--study-name', type=str, default=None, help='Name for the Optuna study (for resuming)')
@click.option('--storage', is_flag=True, default=False, help='Enable persistent Optuna storage under save_dir')
def main(config_path, guide_config_path, n_trials, study_name, storage):
    """Run Bayesian Optimization over hyperparameters to minimize decoder KID."""
    build_registry()

    # Load config
    config = Config().from_disk(config_path)
    guide_config = Config().from_disk(guide_config_path) if guide_config_path else None

    # Basic checks and setup
    save_dir = config['logging']['save_dir']
    phema_folder = os.path.join(save_dir, 'phema')
    if not os.path.exists(phema_folder):
        raise ValueError(f"PHEMA folder not found: {phema_folder}")
    guide_save_dir = guide_config['logging']['save_dir']
    guide_phema_folder = os.path.join(guide_save_dir, 'phema')
    if not os.path.exists(guide_phema_folder):
        raise ValueError(f"PHEMA folder not found: {guide_phema_folder}")

    # Resolve full config
    resolved = registry.resolve(config, validate=False)
    guide_resolved = registry.resolve(guide_config, validate=False) if guide_config else None

    # Model and scheduler
    model = resolved['model']
    guide_model = guide_resolved['model'] if guide_resolved else None
    scheduler = resolved['scheduler']

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=config['training']['mixed_precision'],
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
    if guide_model:
        guide_ema_config = guide_resolved['ema'].copy()
        guide_ema_config['checkpoint_folder'] = guide_phema_folder
        guide_ema = PostHocEMA(guide_model, **guide_ema_config).to(device)
        guide_model = guide_model.to(device)
        guide_model = accelerator.prepare(guide_model)

    # Evaluation parameters
    kid_n_images = int(config['sweep']['kid_n_images'])
    kid_scheduler_steps = int(config['sweep']['kid_scheduler_steps'])
    tile_size = int(config['sweep']['tile_size'])
    
    # Metric selection
    metric_name = str(config['sweep'].get('metric', 'kid')).lower()
    if metric_name not in ('kid', 'fid'):
        raise ValueError(f"Invalid sweep metric '{metric_name}'. Expected 'kid' or 'fid'.")

    min_sigma = config['sweep']['min_ema_sigma']
    max_sigma = config['sweep']['max_ema_sigma']
    min_guidance_scale = config['sweep']['min_guidance_scale']
    max_guidance_scale = config['sweep']['max_guidance_scale']
    min_guide_ema_step = config['sweep']['min_ema_step']
    max_guide_ema_step = config['sweep']['max_ema_step']

    print(f"Loaded config from: {config_path}")
    print(f"Using PHEMA folder: {phema_folder}")
    print(f"Sweeping metric: {metric_name.upper()}")
    print(f"Search range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
    print(f"Search range: guidance_scale ∈ [{min_guidance_scale}, {max_guidance_scale}]")

    # Objective function
    def objective(trial):
        sigma_rel = trial.suggest_float('sigma_rel', min_sigma, max_sigma, log=False)
        if guide_model:
            guide_sigma_rel = trial.suggest_float('guide_sigma_rel', min_sigma, max_sigma, log=False)
            guide_ema_step = trial.suggest_int('guide_ema_step', min_guide_ema_step, max_guide_ema_step, log=True)
            guidance_scale = trial.suggest_float('guidance_scale', min_guidance_scale, max_guidance_scale, log=False)
        else:
            guide_sigma_rel = None
            guide_ema_step = None
            guidance_scale = None

        print("\n" + "=" * 80)
        print(f"Trial {trial.number}: Evaluating σ_rel = {sigma_rel:.6f}, guide_σ_rel = {guide_sigma_rel:.6f}, guide_ema_step = {guide_ema_step}, guidance_scale = {guidance_scale:.6f}")
        print("=" * 80)

        ema_model = ema.synthesize_ema_model(sigma_rel=sigma_rel, step=None)
        ema_model.copy_params_from_ema_to_model()

        if guide_model:
            guide_ema_model = guide_ema.synthesize_ema_model(sigma_rel=guide_sigma_rel, step=guide_ema_step)
            guide_ema_model.copy_params_from_ema_to_model()

        val_dataloader = DataLoader(
            LongDataset(val_dataset, shuffle=True, seed=958),
            batch_size=config['sweep']['batch_size']
        )
        metric_score = evaluate_decoder_kid(
            model=model,
            g_model=guide_model,
            guidance_scale=guidance_scale,
            scheduler=scheduler,
            config=config,
            val_dataloader=val_dataloader,
            kid_n_images=kid_n_images,
            kid_scheduler_steps=kid_scheduler_steps,
            accelerator=accelerator,
            tile_size=tile_size,
            metric=metric_name,
        )
        del val_dataloader
        print(f"Result: {metric_name.upper()} = {metric_score:.6f}")
        return metric_score

    # Study setup
    if study_name is None:
        study_name = f"decoder_{metric_name}_sweep_{os.path.basename(save_dir)}"

    print("\n" + "=" * 80)
    print("Starting Bayesian Optimization with Optuna")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Sweeping metric: {metric_name.upper()}")
    print(f"Search range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
    print(f"Search range: guidance_scale ∈ [{min_guidance_scale}, {max_guidance_scale}]")
    print(f"Search range: guide_σ_rel ∈ [{min_sigma}, {max_sigma}]")
    print(f"Search range: guide_ema_step ∈ [{min_guide_ema_step}, {max_guide_ema_step}]")
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

    # Optionally force a simple baseline as the first trial
    baseline_first = bool(config['sweep'].get('baseline_first_trial', True))
    if baseline_first and len(study.trials) == 0:
        baseline_params = {
            'sigma_rel': 0.05,
        }
        if guide_model:
            baseline_params.update({
                'guidance_scale': 1.0,
                'guide_sigma_rel': 0.05,
                'guide_ema_step': min_guide_ema_step,
            })

        # Only enqueue as first if the study has no trials yet
        print("Enqueuing baseline first trial:", baseline_params)
        study.enqueue_trial(baseline_params)

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Best results
    best_trial = study.best_trial
    optimal_sigma = float(best_trial.params['sigma_rel'])
    optimal_guide_sigma = float(best_trial.params['guide_sigma_rel'])
    optimal_guide_ema_step = int(best_trial.params['guide_ema_step'])
    optimal_guidance_scale = float(best_trial.params['guidance_scale'])
    optimal_score = float(best_trial.value)

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print(f"Optimal σ_rel: {optimal_sigma:.6f}")
    print(f"Optimal guide_σ_rel: {optimal_guide_sigma:.6f}")
    print(f"Optimal guide_ema_step: {optimal_guide_ema_step}")
    print(f"Optimal guidance_scale: {optimal_guidance_scale}")
    print(f"Optimal {metric_name.upper()}: {optimal_score:.6f}")
    print(f"Best trial: {best_trial.number}")
    print("=" * 80)

    # Save results
    results = {
        'metric': metric_name,
        'optimal_sigma_rel': optimal_sigma,
        'optimal_guide_sigma_rel': optimal_guide_sigma,
        'optimal_guide_ema_step': optimal_guide_ema_step,
        'optimal_guidance_scale': optimal_guidance_scale,
        'optimal_score': optimal_score,
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'sigma_rel': t.params.get('sigma_rel'),
                'guide_sigma_rel': t.params.get('guide_sigma_rel'),
                'guide_ema_step': t.params.get('guide_ema_step'),
                'guidance_scale': t.params.get('guidance_scale'),
                'score': t.value,
                'state': t.state.name,
            }
            for t in study.trials
        ],
        'config_path': config_path,
        'save_dir': save_dir,
        'sweep_range': [min_sigma, max_sigma],
        'study_name': study_name,
    }

    output_path = os.path.join(save_dir, 'optuna_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()


