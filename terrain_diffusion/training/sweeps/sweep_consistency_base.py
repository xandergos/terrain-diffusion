"""Sweep for consistency base models using KID (no guidance).

Optimizes three hyperparameters with Optuna:
- ema_sigma (σ_rel)
- ema_step
- intermediate_sigma (for second consistency step)

Sampling uses 2-step consistency with optional tiling, then decodes
to terrain via the decoder using consistency sampling as well.
"""

import json
import os
import math

import click
import torch
import optuna
from tqdm import tqdm
from accelerate import Accelerator
from confection import Config, registry
from torch.utils.data import DataLoader
from torchmetrics.image.kid import KernelInceptionDistance
from ema_pytorch import PostHocEMA

from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import laplacian_denoise, laplacian_decode
from terrain_diffusion.training.evaluation.sample_diffusion_decoder import sample_decoder_consistency_tiled
from terrain_diffusion.training.evaluation.sample_diffusion_base import sample_base_consistency_tiled


def _normalize_uint8_three_channel(images: torch.Tensor) -> torch.Tensor:
    image_min = torch.amin(images, dim=(1, 2, 3), keepdim=True)
    image_max = torch.amax(images, dim=(1, 2, 3), keepdim=True)
    image_range = torch.maximum(image_max - image_min, torch.tensor(1.0, device=images.device))
    image_mid = (image_min + image_max) / 2
    normalized = torch.clamp(((images - image_mid) / image_range + 0.5) * 255, 0, 255)
    return normalized.repeat(1, 3, 1, 1).to(torch.uint8)


def _decode_latents_to_terrain(samples: torch.Tensor, val_dataset, decoder_model, scheduler, generator) -> torch.Tensor:
    device = samples.device
    base_dataset = val_dataset.base_dataset if hasattr(val_dataset, 'base_dataset') else val_dataset

    latents = samples[:, :4]
    lowfreq_input = samples[:, 4:5]

    latents_std = base_dataset.latents_std.to(device)
    latents_mean = base_dataset.latents_mean.to(device)

    latents = (latents / latents_std + latents_mean)
    H, W = lowfreq_input.shape[-2] * 8, lowfreq_input.shape[-1] * 8
    cond_img = torch.nn.functional.interpolate(latents, size=(H, W), mode='nearest')

    noise = torch.randn((latents.shape[0], 1, H, W), generator=generator, device=device, dtype=latents.dtype)
    residual_encoded = sample_decoder_consistency_tiled(
        model=decoder_model,
        scheduler=scheduler,
        cond_img=cond_img,
        noise=noise,
        tile_size=H,
        tile_stride=H,
    )

    residual = residual_encoded
    highfreq = base_dataset.denormalize_residual(residual[:, :1])
    lowfreq = base_dataset.denormalize_lowfreq(lowfreq_input)
    highfreq, lowfreq = laplacian_denoise(highfreq, lowfreq, sigma=5)
    return laplacian_decode(highfreq, lowfreq)


def evaluate_base_kid_consistency(
    model,
    scheduler,
    val_dataloader,
    accelerator,
    *,
    kid_n_images: int,
    tile_size: int,
    intermediate_sigma: float,
    sigma_data: float,
    decoder_model: EDMUnet2D,
    trial=None,
    check_interval: int | None = None,
    prune_probability_threshold: float | None = None,
    crop_size: int,
):
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
            denom = math.sqrt(cur_std * cur_std + other_std * other_std)
            z = -(cur_mean - other_mean) / denom
            p_cur_less = _norm_cdf(z)
            if p_cur_less < prune_probability_threshold:
                return True
        return False

    pbar = tqdm(total=kid_n_images, desc="Calculating Base KID (consistency)")
    kid = KernelInceptionDistance(normalize=True).to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(548)
    val_iter = iter(val_dataloader)

    interm_t = math.atan(intermediate_sigma / sigma_data)

    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            batch = recursive_to(next(val_iter), device=accelerator.device)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs') or []

            noise = torch.randn((2, *images.shape), generator=generator, device=images.device)
            samples = sample_base_consistency_tiled(
                model,
                scheduler,
                cond_img=cond_img,
                noise=noise,
                tile_size=tile_size,
                tile_stride=tile_size,
                intermediate_t=interm_t,
                conditional_inputs=conditional_inputs,
            )
            
            # Central crop samples to crop_size
            _, _, H, W = samples.shape
            crop_size = crop_size if 'crop_size' in locals() else images.shape[-1]
            top = (H - crop_size) // 2
            left = (W - crop_size) // 2
            samples = samples[..., top:top+crop_size, left:left+crop_size]

            terrain_fake = _decode_latents_to_terrain(samples, val_dataloader.dataset, decoder_model, scheduler, generator)
            terrain_real = batch['ground_truth'][..., top*8:top*8+crop_size*8, left*8:left*8+crop_size*8]

            kid.update(_normalize_uint8_three_channel(terrain_fake), real=False)
            kid.update(_normalize_uint8_three_channel(terrain_real), real=True)

            pbar.update(images.shape[0])

            if check_interval and (pbar.n % check_interval == 0 or pbar.n >= pbar.total):
                cur_mean_t, cur_std_t = kid.compute()
                cur_mean = cur_mean_t.item()
                cur_std = cur_std_t.item()
                if trial is not None:
                    trial.report(cur_mean, step=pbar.n)
                if _maybe_prune(cur_mean, cur_std):
                    return cur_mean_t.item(), cur_std_t.item()

    pbar.close()
    kid_mean, kid_std = kid.compute()
    return kid_mean.item(), kid_std.item()


@click.command()
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True,
              help='Path to the consistency base configuration file')
@click.option('--n-trials', type=int, default=100, help='Number of Bayesian optimization trials')
@click.option('--study-name', type=str, default=None, help='Name for the Optuna study (for resuming)')
@click.option('--storage', is_flag=True, default=False, help='Enable persistent Optuna storage under save_dir')
def main(config_path, n_trials, study_name, storage):
    build_registry()

    config = Config().from_disk(config_path)
    resolved = registry.resolve(config, validate=False)

    save_dir = resolved['logging']['save_dir']
    phema_folder = os.path.join(save_dir, 'phema')
    if not os.path.exists(phema_folder):
        raise ValueError(f"PHEMA folder not found: {phema_folder}")

    model = EDMUnet2D.from_pretrained(resolved['model']['main_path'])
    scheduler = resolved['scheduler']

    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=1,
    )
    device = accelerator.device

    val_dataset = resolved['sweep_dataset']

    ema_config = resolved['ema'].copy()
    ema_config['checkpoint_folder'] = phema_folder
    ema = PostHocEMA(model, **ema_config)
    model = model.to(device)
    model = accelerator.prepare(model)

    # Decoder for terrain KID
    ae_path = resolved['evaluation']['kid_autoencoder_path']
    if not os.path.isdir(ae_path):
        raise ValueError(f"Decoder path not found: {ae_path}")
    decoder_model = EDMUnet2D.from_pretrained(ae_path).to(device)
    decoder_model.eval()

    kid_n_images = resolved['sweep']['kid_n_images']
    tile_size = resolved['sweep']['tile_size']
    sigma_data = resolved['training'].get('sigma_data', 1.0)

    intermediate_steps = resolved['sweep'].get('intermediate_steps', 1024)
    prune_probability_threshold = resolved['sweep'].get('prune_probability_threshold', 0.05)

    min_ema_sigma = resolved['sweep']['min_ema_sigma']
    max_ema_sigma = resolved['sweep']['max_ema_sigma']
    min_ema_step = resolved['sweep']['min_ema_step']
    max_ema_step = resolved['sweep']['max_ema_step']
    min_intermediate_sigma = resolved['sweep']['min_intermediate_sigma']
    max_intermediate_sigma = resolved['sweep']['max_intermediate_sigma']

    print(f"Loaded config from: {config_path}")
    print(f"Using PHEMA folder: {phema_folder}")
    print(f"Search range: EMA σ_rel ∈ [{min_ema_sigma}, {max_ema_sigma}]")
    print(f"Search range: EMA step ∈ [{min_ema_step}, {max_ema_step}]")
    print(f"Search range: intermediate σ ∈ [{min_intermediate_sigma}, {max_intermediate_sigma}]")

    def objective(trial: optuna.trial.Trial):
        ema_sigma = trial.suggest_float('ema_sigma', min_ema_sigma, max_ema_sigma, log=False)
        ema_step = trial.suggest_int('ema_step', min_ema_step, max_ema_step, log=True)
        intermediate_sigma = trial.suggest_float('intermediate_sigma', min_intermediate_sigma, max_intermediate_sigma, log=True)

        print("\n" + "=" * 80)
        print(f"Trial {trial.number}: EMA σ = {ema_sigma:.6f}, EMA step = {ema_step}, intermediate_sigma = {intermediate_sigma:.6f}")
        print("=" * 80)

        model.to('cpu')
        ema_model = ema.synthesize_ema_model(sigma_rel=ema_sigma, step=ema_step)
        ema_model.copy_params_from_ema_to_model()
        model.to(device)
        
        val_loader = DataLoader(
            LongDataset(val_dataset, shuffle=True, seed=958),
            batch_size=resolved['sweep']['batch_size']
        )

        kid_mean, kid_std = evaluate_base_kid_consistency(
            model=model,
            scheduler=scheduler,
            val_dataloader=val_loader,
            accelerator=accelerator,
            kid_n_images=kid_n_images,
            tile_size=tile_size,
            intermediate_sigma=intermediate_sigma,
            sigma_data=sigma_data,
            decoder_model=decoder_model,
            trial=trial,
            check_interval=intermediate_steps,
            prune_probability_threshold=prune_probability_threshold,
            crop_size=64
        )

        del val_loader
        trial.set_user_attr('kid_std', kid_std)
        print(f"Result: KID mean = {kid_mean:.6f} (std = {kid_std:.6f})")
        return kid_mean

    if study_name is None:
        study_name = f"consistency_base_kid_sweep_{os.path.basename(save_dir)}"

    print("\n" + "=" * 80)
    print("Starting Bayesian Optimization with Optuna")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Search range: EMA σ ∈ [{min_ema_sigma}, {max_ema_sigma}]")
    print(f"Search range: EMA step ∈ [{min_ema_step}, {max_ema_step}]")
    print(f"Search range: intermediate_sigma ∈ [{min_intermediate_sigma}, {max_intermediate_sigma}]")

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
                                           n_startup_trials=resolved['sweep'].get('n_startup_trials', 24),
                                           multivariate=True,
                                           group=True,
                                           n_ei_candidates=resolved['sweep'].get('n_ei_candidates', 64),
                                           prior_weight=resolved['sweep'].get('prior_weight', 1.0)),
    )

    baseline_first = bool(resolved['sweep'].get('baseline_first_trial', True))
    if baseline_first and len(study.trials) == 0:
        baseline_params = {
            'ema_sigma': 0.05,
            'ema_step': min_ema_step,
            'intermediate_sigma': 2.0,
        }
        print("Enqueuing baseline first trial:", baseline_params)
        study.enqueue_trial(baseline_params)

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    optimal_ema_sigma = best_trial.params['ema_sigma']
    optimal_ema_step = best_trial.params['ema_step']
    optimal_intermediate_sigma = best_trial.params['intermediate_sigma']
    optimal_kid = best_trial.value

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print(f"Optimal EMA σ: {optimal_ema_sigma:.6f}")
    print(f"Optimal EMA step: {optimal_ema_step}")
    print(f"Optimal intermediate_sigma: {optimal_intermediate_sigma:.6f}")
    print(f"Optimal KID mean: {optimal_kid:.6f}")
    print(f"Best trial: {best_trial.number}")
    print("=" * 80)

    results = {
        'optimal_ema_sigma': optimal_ema_sigma,
        'optimal_ema_step': optimal_ema_step,
        'optimal_intermediate_sigma': optimal_intermediate_sigma,
        'optimal_kid_mean': optimal_kid,
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'ema_sigma': t.params.get('ema_sigma'),
                'ema_step': t.params.get('ema_step'),
                'intermediate_sigma': t.params.get('intermediate_sigma'),
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


