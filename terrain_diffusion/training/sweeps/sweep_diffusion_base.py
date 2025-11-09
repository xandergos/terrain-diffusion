"""Bayesian Optimization sweep for base diffusion KID (decoded terrain).

This mirrors the decoder sweep but evaluates KID after decoding latents
to terrain using the diffusion decoder, as in the trainer's base KID.
Includes optional guidance.
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
from terrain_diffusion.training.evaluation.sample_diffusion_base import sample_base_diffusion


def _normalize_uint8_three_channel(images: torch.Tensor) -> torch.Tensor:
    """Normalize single-channel images to uint8 [0, 255] repeated to 3 channels."""
    image_min = torch.amin(images, dim=(1, 2, 3), keepdim=True)
    image_max = torch.amax(images, dim=(1, 2, 3), keepdim=True)
    image_range = torch.maximum(image_max - image_min, torch.tensor(1.0, device=images.device))
    image_mid = (image_min + image_max) / 2
    normalized = torch.clamp(((images - image_mid) / image_range + 0.5) * 255, 0, 255)
    return normalized.repeat(1, 3, 1, 1).to(torch.uint8)


def _decode_latents_to_terrain(samples: torch.Tensor, val_dataset, decoder_model, scheduler, generator) -> torch.Tensor:
    """Decode [latents(4ch), lowfreq(1ch)] into terrain using evaluation primitives."""
    device = samples.device
    base_dataset = val_dataset.base_dataset if hasattr(val_dataset, 'base_dataset') else val_dataset

    latents = samples[:, :4]
    lowfreq_input = samples[:, 4:5]

    latents_std = base_dataset.latents_std.to(device)
    latents_mean = base_dataset.latents_mean.to(device)

    # Normalize latents for decoder and upsample to target resolution (nearest to match trainer)
    latents = (latents / latents_std + latents_mean)
    H, W = lowfreq_input.shape[-2] * 8, lowfreq_input.shape[-1] * 8
    cond_img = torch.nn.functional.interpolate(latents, size=(H, W), mode='nearest')

    # Consistency decode residual via tiled primitive (single tile covering full image)
    noise = torch.randn((latents.shape[0], 1, H, W), generator=generator, device=device, dtype=latents.dtype)
    residual_encoded = sample_decoder_consistency_tiled(
        model=decoder_model,
        scheduler=scheduler,
        cond_img=cond_img,
        noise=noise,
        tile_size=H,
        tile_stride=H
    )

    # Convert from trig-flow space
    residual = residual_encoded

    # Denormalize and compose terrain
    highfreq = base_dataset.denormalize_residual(residual[:, :1])
    lowfreq = base_dataset.denormalize_lowfreq(lowfreq_input)
    highfreq, lowfreq = laplacian_denoise(highfreq, lowfreq, sigma=5)
    return laplacian_decode(highfreq, lowfreq)


def evaluate_base_kid(
    model,
    scheduler,
    val_dataloader,
    autoencoder,
    accelerator,
    kid_n_images: int,
    kid_scheduler_steps: int,
    g_model=None,
    guidance_scale: float = 1.0,
    trial=None,
    check_interval: int | None = None,
    prune_probability_threshold: float | None = None,
):
    """Compute KID for base diffusion by decoding latents to terrain, with optional guidance."""

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

    device = accelerator.device
    pbar = tqdm(total=kid_n_images, desc="Calculating Base KID")
    kid = KernelInceptionDistance(normalize=True).to(device)
    generator = torch.Generator(device=device).manual_seed(548)
    val_iter = iter(val_dataloader)

    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            batch = recursive_to(next(val_iter), device=device)
            images = batch['image']
            cond_inputs = batch['cond_inputs_img']
            histogram_raw = batch['histogram_raw']
            bs = images.shape[0]

            # Sample latents using evaluation primitive
            model.eval()
            if g_model:
                g_model.eval()
            samples = sample_base_diffusion(
                model=model,
                scheduler=scheduler,
                shape=images.shape,
                cond_inputs=cond_inputs,
                cond_means=torch.zeros(7, device=device),
                cond_stds=torch.ones(7, device=device),
                noise_level=torch.zeros(bs, 1, device=device),
                histogram_raw=histogram_raw,
                steps=int(kid_scheduler_steps),
                guide_model=g_model,
                guidance_scale=float(guidance_scale),
                generator=generator,
                tile_size=64
            )

            # Decode to terrain
            terrain_fake = _decode_latents_to_terrain(samples, val_dataloader.dataset, autoencoder, scheduler, generator)

            # Real terrain
            ground_truth = batch['ground_truth']
            terrain_real = ground_truth

            kid.update(_normalize_uint8_three_channel(terrain_fake), real=False)
            kid.update(_normalize_uint8_three_channel(terrain_real), real=True)

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
    kid.to('cpu')
    del kid
    return kid_mean.item(), kid_std.item()


@click.command()
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True,
              help='Path to the base diffusion configuration file')
@click.option('-gc', '--guide-config', 'guide_config_path', type=click.Path(exists=True), required=False,
              help='Path to the guide model configuration file')
@click.option('--n-trials', type=int, default=100, help='Number of Bayesian optimization trials')
@click.option('--study-name', type=str, default=None, help='Name for the Optuna study (for resuming)')
@click.option('--storage', is_flag=True, default=False, help='Enable persistent Optuna storage under save_dir')
def main(config_path, guide_config_path, n_trials, study_name, storage):
    """Run Bayesian Optimization over hyperparameters to minimize base KID."""
    build_registry()

    # Load config(s)
    config = Config().from_disk(config_path)
    guide_config = Config().from_disk(guide_config_path) if guide_config_path else None

    # Resolve
    resolved = registry.resolve(config, validate=False)
    guide_resolved = registry.resolve(guide_config, validate=False) if guide_config else None

    save_dir = resolved['logging']['save_dir']
    phema_folder = os.path.join(save_dir, 'phema')
    if not os.path.exists(phema_folder):
        raise ValueError(f"PHEMA folder not found: {phema_folder}")

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

    # Dataset
    val_dataset = resolved['sweep_dataset']

    # Autoencoder for decoding
    ae_path = resolved['evaluation']['kid_autoencoder_path']
    if not os.path.isdir(ae_path):
        raise ValueError(f"Autoencoder path not found: {ae_path}")
    autoencoder = EDMUnet2D.from_pretrained(ae_path).to(device)
    autoencoder.eval()

    # EMA setup
    ema_config = resolved['ema'].copy()
    ema_config['checkpoint_folder'] = phema_folder
    ema = PostHocEMA(model, **ema_config, allow_different_devices=True)
    model = model.to(device)
    model = accelerator.prepare(model)

    if guide_model:
        guide_save_dir = guide_resolved['logging']['save_dir']
        guide_phema_folder = os.path.join(guide_save_dir, 'phema')
        if not os.path.exists(guide_phema_folder):
            raise ValueError(f"PHEMA folder not found: {guide_phema_folder}")
        guide_ema_config = guide_resolved['ema'].copy()
        guide_ema_config['checkpoint_folder'] = guide_phema_folder
        guide_ema = PostHocEMA(guide_model, **guide_ema_config, allow_different_devices=True)
        guide_model = guide_model.to(device)
        guide_model = accelerator.prepare(guide_model)
    else:
        guide_ema = None

    # Evaluation params
    kid_n_images = int(resolved['sweep']['kid_n_images'])
    kid_scheduler_steps = int(resolved['sweep']['kid_scheduler_steps'])

    # Pruning params
    intermediate_steps = int(resolved['sweep']['intermediate_steps'])
    prune_probability_threshold = float(resolved['sweep']['prune_probability_threshold'])

    # Sweep ranges
    sweep_cfg = resolved['sweep']
    min_sigma = float(sweep_cfg['min_ema_sigma'])
    max_sigma = float(sweep_cfg['max_ema_sigma'])
    min_guide_sigma = sweep_cfg.get('min_guide_ema_sigma', min_sigma)
    max_guide_sigma = sweep_cfg.get('max_guide_ema_sigma', max_sigma)
    min_guidance_scale = float(sweep_cfg['min_guidance_scale'])
    max_guidance_scale = float(sweep_cfg['max_guidance_scale'])
    min_main_ema_step = int(sweep_cfg['min_main_ema_step'])
    max_main_ema_step = int(sweep_cfg['max_main_ema_step'])
    min_guide_ema_step = int(sweep_cfg['min_ema_step'])
    max_guide_ema_step = int(sweep_cfg['max_ema_step'])

    print(f"Loaded config from: {config_path}")
    print(f"Using PHEMA folder: {phema_folder}")
    print(f"Search range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
    print(f"Search range: ema_step ∈ [{min_main_ema_step}, {max_main_ema_step}]")
    if guide_model:
        print(f"Search range: guidance_scale ∈ [{min_guidance_scale}, {max_guidance_scale}]")
        print(f"Search range: guide_σ_rel ∈ [{min_guide_sigma}, {max_guide_sigma}]")
        print(f"Search range: guide_ema_step ∈ [{min_guide_ema_step}, {max_guide_ema_step}]")

    def objective(trial: optuna.trial.Trial):
        sigma_rel = trial.suggest_float('sigma_rel', min_sigma, max_sigma, log=False)
        ema_step = trial.suggest_int('ema_step', min_main_ema_step, max_main_ema_step, log=True)
        if guide_model:
            guide_sigma_rel = trial.suggest_float('guide_sigma_rel', min_guide_sigma, max_guide_sigma, log=False)
            guide_ema_step = trial.suggest_int('guide_ema_step', min_guide_ema_step, max_guide_ema_step, log=True) if max_guide_ema_step > min_guide_ema_step else min_guide_ema_step
            guidance_scale = trial.suggest_float('guidance_scale', min_guidance_scale, max_guidance_scale, log=False)
        else:
            guide_sigma_rel = None
            guide_ema_step = None
            guidance_scale = 1.0

        print("\n" + "=" * 80)
        print(f"Trial {trial.number}: σ_rel = {sigma_rel:.6f}, ema_step = {ema_step}, guide_σ_rel = {guide_sigma_rel}, guide_ema_step = {guide_ema_step}, guidance_scale = {guidance_scale:.6f}")
        print("=" * 80)

        model.to('cpu')
        ema_model = ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step)
        ema_model.copy_params_from_ema_to_model()

        if guide_model and guide_ema is not None:
            guide_model.to('cpu')
            guide_ema_model = guide_ema.synthesize_ema_model(sigma_rel=guide_sigma_rel, step=guide_ema_step)
            guide_ema_model.copy_params_from_ema_to_model()
            guide_model.to(device)
        
        model.to(device)

        val_loader = DataLoader(
            LongDataset(val_dataset, shuffle=True, seed=958),
            batch_size=resolved['sweep']['batch_size'],
        )

        kid_mean, kid_std = evaluate_base_kid(
            model=model,
            scheduler=scheduler,
            val_dataloader=val_loader,
            autoencoder=autoencoder,
            accelerator=accelerator,
            kid_n_images=kid_n_images,
            kid_scheduler_steps=kid_scheduler_steps,
            g_model=guide_model,
            guidance_scale=guidance_scale,
            trial=trial,
            check_interval=intermediate_steps,
            prune_probability_threshold=prune_probability_threshold,
        )

        del val_loader
        
        del ema_model
        if guide_model and guide_ema is not None:
            del guide_ema_model
        torch.cuda.empty_cache()
        
        trial.set_user_attr('kid_std', float(kid_std))
        print(f"Result: KID mean = {kid_mean:.6f} (std = {kid_std:.6f})")
        return kid_mean

    if study_name is None:
        study_name = f"base_kid_sweep_{os.path.basename(save_dir)}"

    print("\n" + "=" * 80)
    print("Starting Bayesian Optimization with Optuna")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Search range: σ_rel ∈ [{min_sigma}, {max_sigma}]")
    if guide_model:
        print(f"Search range: guidance_scale ∈ [{min_guidance_scale}, {max_guidance_scale}]")
        print(f"Search range: guide_σ_rel ∈ [{min_guide_sigma}, {max_guide_sigma}]")
        print(f"Search range: guide_ema_step ∈ [{min_guide_ema_step}, {max_guide_ema_step}]")

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
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=resolved['sweep'].get('n_startup_trials', 12),
            multivariate=True,
            group=True,
            n_ei_candidates=resolved['sweep'].get('n_ei_candidates', 24),
            prior_weight=resolved['sweep'].get('prior_weight', 1.0),
        ),
    )

    # Optional baseline enqueued trials
    baseline_first = bool(resolved['sweep']['baseline_first_trial'])
    if baseline_first and len(study.trials) == 0:
        baseline_params = {
            'sigma_rel': 0.05,
            'ema_step': max_main_ema_step,
        }
        if guide_model:
            baseline_params.update({
                'guidance_scale': 1.0,
                'guide_sigma_rel': 0.05,
                'guide_ema_step': min_guide_ema_step,
            })
        print("Enqueuing baseline first trial:", baseline_params)
        study.enqueue_trial(baseline_params)

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    optimal_sigma = float(best_trial.params['sigma_rel'])
    optimal_ema_step = int(best_trial.params['ema_step'])
    optimal_kid = float(best_trial.value)

    if guide_model:
        optimal_guide_sigma = float(best_trial.params.get('guide_sigma_rel', 0.0))
        optimal_guide_ema_step = int(best_trial.params.get('guide_ema_step', 0))
        optimal_guidance_scale = float(best_trial.params.get('guidance_scale', 1.0))
    else:
        optimal_guide_sigma = None
        optimal_guide_ema_step = None
        optimal_guidance_scale = 1.0

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print(f"Optimal σ_rel: {optimal_sigma:.6f}")
    print(f"Optimal ema_step: {optimal_ema_step}")
    if guide_model:
        print(f"Optimal guide_σ_rel: {optimal_guide_sigma:.6f}")
        print(f"Optimal guide_ema_step: {optimal_guide_ema_step}")
        print(f"Optimal guidance_scale: {optimal_guidance_scale}")
    print(f"Optimal KID mean: {optimal_kid:.6f}")
    print(f"Best trial: {best_trial.number}")
    print("=" * 80)

    results = {
        'optimal_sigma_rel': optimal_sigma,
        'optimal_ema_step': optimal_ema_step,
        'optimal_guide_sigma_rel': optimal_guide_sigma,
        'optimal_guide_ema_step': optimal_guide_ema_step,
        'optimal_guidance_scale': optimal_guidance_scale,
        'optimal_kid_mean': optimal_kid,
        'best_trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'sigma_rel': t.params.get('sigma_rel'),
                'ema_step': t.params.get('ema_step'),
                'guide_sigma_rel': t.params.get('guide_sigma_rel'),
                'guide_ema_step': t.params.get('guide_ema_step'),
                'guidance_scale': t.params.get('guidance_scale'),
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


