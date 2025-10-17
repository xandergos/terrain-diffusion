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
from collections import defaultdict
import optuna
import matplotlib.pyplot as plt

from ema_pytorch import PostHocEMA
from torchmetrics.image.kid import KernelInceptionDistance

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


def evaluate_decoder_kid(model, g_model, guidance_scale, score_scaling, scheduler, val_dataloader, kid_n_images, kid_scheduler_steps, accelerator):
    """Compute KID for the decoder using diffusion sampling on the validation set."""
    pbar = tqdm(total=kid_n_images, desc="Calculating Decoder KID")
    kid = KernelInceptionDistance(normalize=True).to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(548)
    val_dataloader_iter = iter(val_dataloader)

    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            batch = recursive_to(next(val_dataloader_iter), device=accelerator.device)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')

            samples = torch.randn(images.shape, generator=generator, device=images.device) * scheduler.sigmas[0]

            # Sampling loop
            scheduler.set_timesteps(kid_scheduler_steps)
            for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                t, sigma = t.to(samples.device), sigma.to(samples.device)

                scaled_input = scheduler.precondition_inputs(samples, sigma)
                cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))

                x = torch.cat([scaled_input, cond_img], dim=1)
                if not g_model or guidance_scale == 1.0:
                    model_output = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs)
                else:
                    model_output_m = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs)
                    model_output_g = g_model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs)
                    model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
                model_output = model_output * score_scaling

                samples = scheduler.step(model_output, t, samples).prev_sample

            kid.update(_normalize_uint8_three_channel(samples), real=False)
            kid.update(_normalize_uint8_three_channel(images), real=True)

            pbar.update(images.shape[0])

    pbar.close()
    kid_mean, kid_std = kid.compute()
    return kid_mean.item(), kid_std.item()


@click.command()
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True,
              help='Path to the diffusion decoder configuration file')
@click.option('-gc', '--guide-config', 'guide_config_path', type=click.Path(exists=True), required=False,
              help='Path to the guide model configuration file')
@click.option('--n-trials', type=int, default=20, help='Number of Bayesian optimization trials')
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

    # Resolve full config
    resolved = registry.resolve(config, validate=False)
    guide_resolved = registry.resolve(guide_config, validate=False) if guide_config else None

    # Model and scheduler
    model = resolved['model']
    guide_model = guide_resolved['model'] if guide_resolved else None
    scheduler = resolved['scheduler']

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=1,
    )
    device = accelerator.device

    # Validation dataloader
    val_dataset = resolved['val_dataset']

    # EMA setup
    ema_config = resolved['ema'].copy()
    ema_config['checkpoint_folder'] = phema_folder
    ema = PostHocEMA(model, **ema_config).to(device)
    model = model.to(device)
    model = accelerator.prepare(model)
    if guide_model:
        guide_ema_config = guide_resolved['ema'].copy()
        guide_ema_config['checkpoint_folder'] = phema_folder
        guide_ema = PostHocEMA(guide_model, **guide_ema_config).to(device)
        guide_model = guide_model.to(device)
        guide_model = accelerator.prepare(guide_model)

    # Evaluation parameters
    kid_n_images = int(config['evaluation']['kid_n_images'])
    kid_scheduler_steps = int(config['evaluation']['kid_scheduler_steps'])
    
    min_sigma = config['sweep']['min_ema_sigma']
    max_sigma = config['sweep']['max_ema_sigma']
    min_guidance_scale = config['sweep']['min_guidance_scale']
    max_guidance_scale = config['sweep']['max_guidance_scale']
    min_score_scaling = config['sweep']['min_score_scaling']
    max_score_scaling = config['sweep']['max_score_scaling']
    min_guide_ema_step = config['sweep']['min_ema_step']
    max_guide_ema_step = config['sweep']['max_ema_step']

    print(f"Loaded config from: {config_path}")
    print(f"Using PHEMA folder: {phema_folder}")
    print(f"Search range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
    print(f"Search range: guidance_scale ∈ [{min_guidance_scale}, {max_guidance_scale}]")
    print(f"Search range: score_scaling ∈ [{min_score_scaling}, {max_score_scaling}]")

    # Objective function
    def objective(trial):
        sigma_rel = trial.suggest_float('sigma_rel', min_sigma, max_sigma, log=False)
        if guide_model:
            guide_sigma_rel = trial.suggest_float('guide_sigma_rel', min_sigma, max_sigma, log=False)
            guide_ema_step = trial.suggest_int('guide_ema_step', min_guide_ema_step, max_guide_ema_step, step=1024)
            guidance_scale = trial.suggest_float('guidance_scale', min_guidance_scale, max_guidance_scale, log=False)
        else:
            guide_sigma_rel = None
            guide_ema_step = None
            guidance_scale = None
        score_scaling = trial.suggest_float('score_scaling', min_score_scaling, max_score_scaling, log=False)

        print("\n" + "=" * 80)
        print(f"Trial {trial.number}: Evaluating σ_rel = {sigma_rel:.6f}, guide_σ_rel = {guide_sigma_rel:.6f}, guide_ema_step = {guide_ema_step}, guidance_scale = {guidance_scale:.6f}, score_scaling = {score_scaling:.6f}")
        print("=" * 80)

        ema_model = ema.synthesize_ema_model(sigma_rel=sigma_rel, step=None)
        ema_model.copy_params_from_ema_to_model()

        if guide_model:
            guide_ema_model = guide_ema.synthesize_ema_model(sigma_rel=guide_sigma_rel, step=guide_ema_step)
            guide_ema_model.copy_params_from_ema_to_model()

        val_dataloader = DataLoader(
            LongDataset(val_dataset, shuffle=True, seed=958),
            batch_size=config['training']['train_batch_size'],
            **resolved['dataloader_kwargs']
        )
        kid_mean, kid_std = evaluate_decoder_kid(
            model=model,
            scheduler=scheduler,
            val_dataloader=val_dataloader,
            kid_n_images=kid_n_images,
            kid_scheduler_steps=kid_scheduler_steps,
            accelerator=accelerator,
        )

        print(f"Result: KID mean = {kid_mean:.6f} (std = {kid_std:.6f})")
        return kid_mean

    # Study setup
    if study_name is None:
        study_name = f"decoder_kid_sweep_{os.path.basename(save_dir)}"

    print("\n" + "=" * 80)
    print("Starting Bayesian Optimization with Optuna")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Search range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
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
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Best results
    best_trial = study.best_trial
    optimal_sigma = float(best_trial.params['sigma_rel'])
    optimal_kid = float(best_trial.value)

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print(f"Optimal σ_rel: {optimal_sigma:.6f}")
    print(f"Optimal KID mean: {optimal_kid:.6f}")
    print(f"Best trial: {best_trial.number}")
    print("=" * 80)

    # Save results
    results = {
        'optimal_sigma_rel': optimal_sigma,
        'optimal_kid_mean': optimal_kid,
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'sigma_rel': t.params.get('sigma_rel'),
                'kid_mean': t.value,
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


