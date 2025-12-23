import click
import torch
import numpy as np
import math
import os
from pathlib import Path
from PIL import Image
from accelerate import Accelerator
from confection import Config, registry
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.common.model_utils import resolve_model_path, MODEL_PATHS
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import laplacian_denoise, laplacian_decode
from terrain_diffusion.training.evaluation.sample_diffusion_decoder import sample_decoder_consistency_tiled
from terrain_diffusion.training.evaluation import _linear_weight_window, _constant_weight_window, _tile_starts
from terrain_diffusion.training.evaluation.sample_diffusion_base import _process_cond_img
from terrain_diffusion.inference.relief_map import get_relief_map

def _normalize_uint8_three_channel(images: torch.Tensor) -> torch.Tensor:
    """Normalize single-channel images to uint8 [0, 255] repeated to 3 channels."""
    image_min = torch.amin(images, dim=(1, 2, 3), keepdim=True)
    image_max = torch.amax(images, dim=(1, 2, 3), keepdim=True)
    image_range = torch.maximum(image_max - image_min, torch.tensor(255.0, device=images.device))
    image_mid = (image_min + image_max) / 2
    normalized = torch.clamp(((images - image_mid) / image_range + 0.5) * 255, 0, 255)
    return normalized.repeat(1, 3, 1, 1).to(torch.uint8)

def _decode_latents_to_terrain(samples: torch.Tensor, val_dataset, decoder_model, scheduler, generator, inter_t) -> torch.Tensor:
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
        tile_stride=H,
        intermediate_t=inter_t
    )

    # Convert from trig-flow space
    residual = residual_encoded

    # Denormalize and compose terrain
    highfreq = base_dataset.denormalize_residual(residual[:, :1])
    lowfreq = base_dataset.denormalize_lowfreq(lowfreq_input)
    highfreq, lowfreq = laplacian_denoise(highfreq, lowfreq, sigma=5)
    
    # When calculating FID/KID, we can approximate padding with extrapolation. But typically we can get padding from InfiniteDiffusion directly.
    return laplacian_decode(highfreq, lowfreq, extrapolate=True)

@click.command()
@click.option('-m', '--model', 'model_path', type=str, default=None, help='Path to pretrained base model (local or HuggingFace repo)')
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), help='Path to the base diffusion configuration file', default='configs/diffusion_base/consistency_base_192-3.cfg')
@click.option('--num-images', type=int, default=50000, help='Number of images to evaluate')
@click.option('--batch-size', type=int, default=40, help='Batch size for evaluation')
@click.option('--decoder-batch-size', type=int, default=10, help='Batch size for decoder evaluation')
@click.option('--metric', type=click.Choice(['fid', 'kid'], case_sensitive=False), default='fid', help='Metric to evaluate')
@click.option('--inter-t', type=float, default=0.61, help='Intermediate t for 2-step sampling. Use 0 or less for one-step model.')
@click.option('--decoder-inter-t', type=float, default=0.13, help='Intermediate t for 2-step decoder sampling. Use 0 or less for one-step model.')
@click.option('--weight-window-fn', type=click.Choice(['linear', 'constant'], case_sensitive=False), default='linear', help='Weight window function to use for tiling')
@click.option('--save-images', type=int, default=0, help='Number of images to save to results directory')
@click.option('--experiment-name', type=str, default='infinite_consistency', help='Name of experiment folder in results/')
def main(model_path, config_path, num_images, batch_size, decoder_batch_size, metric, inter_t, decoder_inter_t, weight_window_fn, save_images, experiment_name):
    """Evaluate base diffusion using FID/KID on decoded terrain."""
    build_registry()
    
    # Load config
    config = Config().from_disk(config_path)
    kept_keys = {'scheduler', 'results_dataset', 'evaluation'}
    keys_to_delete = [k for k in config.keys() if k not in kept_keys]
    for k in keys_to_delete:
        del config[k]
    config['results_dataset']['crop_size'] = 192
    resolved = registry.resolve(config, validate=False)
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    # Resolve model paths (user override -> local default -> HuggingFace)
    resolved_model_path = resolve_model_path(model_path, *MODEL_PATHS["base"])
    resolved_decoder_path = resolve_model_path(
        resolved['evaluation'].get('kid_autoencoder_path'),
        *MODEL_PATHS["decoder"]
    )
    
    # Load Models
    print(f"Loading model from {resolved_model_path}...")
    model = EDMUnet2D.from_pretrained(resolved_model_path)
    model = model.to(device)
    model.eval()
        
    # Load decoder model
    print(f"Loading decoder model from {resolved_decoder_path}...")
    decoder_model = EDMUnet2D.from_pretrained(resolved_decoder_path).to(device)
    decoder_model.eval()
    
    model, decoder_model = accelerator.prepare(model, decoder_model)
    
    # Scheduler
    scheduler = resolved['scheduler']
    
    # Dataset
    val_dataset_config = resolved['results_dataset']
    val_dataset = LongDataset(val_dataset_config, shuffle=True, seed=958)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    val_dataloader_iter = iter(val_dataloader)
    
    # Metric setup
    metric_name = metric.lower()
    if metric_name == 'kid':
        image_metric = KernelInceptionDistance(normalize=True).to(device)
    else:
        image_metric = FrechetInceptionDistance(normalize=True).to(device)
        
    decoder_inter_t_list = [decoder_inter_t] if decoder_inter_t > 0 else None
    
    print(f"Evaluating {metric_name.upper()} on {num_images} images")
    
    # Setup results directory for saving images
    results_dir = Path("results") / experiment_name
    if save_images > 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    
    generator = torch.Generator(device=device).manual_seed(548)
    
    pbar = tqdm(total=num_images, desc=f"Calculating {metric_name.upper()}")
    
    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            try:
                batch = recursive_to(next(val_dataloader_iter), device=device)
            except StopIteration:
                val_dataloader_iter = iter(val_dataloader)
                batch = recursive_to(next(val_dataloader_iter), device=device)
                
            images = batch['image']
            cond_inputs = batch['cond_inputs_img']
            histogram_raw = batch['histogram_raw']
            
            # Adjust batch size if we need fewer images to finish
            remaining = num_images - pbar.n
            if images.shape[0] > remaining:
                images = images[:remaining]
                cond_inputs = cond_inputs[:remaining]
                histogram_raw = histogram_raw[:remaining]
                
            bs = images.shape[0]
            
            # Sample latents using evaluation primitive
            with accelerator.autocast():
                device = accelerator.device
                dtype = torch.float32
                sigma0 = scheduler.sigmas[0].to(device)
                sigma_data = scheduler.config.sigma_data
                
                init_t = torch.atan(sigma0 / sigma_data).to(device=device, dtype=dtype)
                if inter_t > 0:
                    t_scalars = (init_t, torch.tensor(inter_t, device=device, dtype=dtype))
                else:
                    t_scalars = (init_t,)

                # Tiled sampling
                B, C, H, W = shape = images.shape
                tile_size = 64
                stride = 64 // 2
                if weight_window_fn == 'linear':
                    weights = _linear_weight_window(tile_size, device, dtype)
                elif weight_window_fn == 'constant':
                    weights = _constant_weight_window(tile_size, device, dtype)
                else:
                    raise ValueError(f"Invalid weight window function: {weight_window_fn}")

                h_starts = _tile_starts(H, tile_size, stride)
                w_starts = _tile_starts(W, tile_size, stride)
                
                if cond_inputs.ndim == 1 and len(h_starts) * len(w_starts) > 1:
                    raise ValueError(f"cond_inputs must be a tensor image for tiled sampling. Cond inputs must have width {len(w_starts)+3} and height {len(h_starts)+3}.")
                elif cond_inputs.ndim == 4:
                    assert cond_inputs.shape[-1] == len(w_starts)+3
                    assert cond_inputs.shape[-2] == len(h_starts)+3

                sample = torch.zeros(shape, device=device, dtype=dtype)
                num_steps = len(t_scalars)
                for step, t_scalar in enumerate(t_scalars):
                    step_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                    output = torch.zeros(shape, device=device, dtype=dtype)
                    output_weights = torch.zeros(shape, device=device, dtype=dtype)
                    for ic, i0 in enumerate(h_starts):
                        for jc, j0 in enumerate(w_starts):
                            # Don't need to sample edges on the last step (they get cropped)
                            if step == num_steps - 1 and (ic == 0 or jc == 0 or ic == len(h_starts) - 1 or jc == len(w_starts) - 1):
                                continue
                            
                            if cond_inputs.ndim == 4:
                                tile_cond = cond_inputs[..., ic:ic+4, jc:jc+4]
                                tile_cond = [_process_cond_img(tile_cond, histogram_raw, torch.zeros(7, device=device), torch.ones(7, device=device), torch.zeros(bs, 1, device=device))]
                            else:
                                tile_cond = [cond_inputs]
                            
                            i1, j1 = i0 + tile_size, j0 + tile_size
                            z = step_noise[..., i0:i1, j0:j1] * sigma_data
                            tile_sample = sample[..., i0:i1, j0:j1]

                            t = t_scalar.view(1, 1, 1, 1).expand(B, 1, 1, 1)
                            x_t = torch.cos(t) * tile_sample + torch.sin(t) * z
                            
                            model_in = x_t / sigma_data
                            pred = -model(model_in, noise_labels=t.flatten(), conditional_inputs=tile_cond)
                            tile_samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred

                            output[..., i0:i1, j0:j1] += tile_samples * weights
                            output_weights[..., i0:i1, j0:j1] += weights

                    sample = output / output_weights

                samples = sample / scheduler.config.sigma_data

            # Central crop samples to 64x64
            start_h = (samples.shape[-2] - 64) // 2
            start_w = (samples.shape[-1] - 64) // 2
            samples = samples[..., start_h:start_h+64, start_w:start_w+64]

            terrain_fake = torch.zeros(bs, 1, samples.shape[-2]*8, samples.shape[-1]*8, device=device)
            with accelerator.autocast():
                assert terrain_fake.shape[0] % decoder_batch_size == 0
                for i in range(terrain_fake.shape[0]//decoder_batch_size):
                    in_samples = samples[i*decoder_batch_size:(i+1)*decoder_batch_size]
                    terrain_fake[i*decoder_batch_size:(i+1)*decoder_batch_size] = _decode_latents_to_terrain(in_samples, val_dataset, decoder_model, scheduler, generator, inter_t=decoder_inter_t_list)
            terrain_fake = torch.sign(terrain_fake) * torch.square(terrain_fake)
            
            # Real terrain
            ground_truth = batch['ground_truth']
            if ground_truth.shape[0] > bs:
                 ground_truth = ground_truth[:bs]
            terrain_real = ground_truth
            terrain_real = torch.sign(terrain_real) * torch.square(terrain_real)
            
            # Central crop ground_truth to 512x512
            start_h = (terrain_real.shape[-2] - 512) // 2
            start_w = (terrain_real.shape[-1] - 512) // 2
            terrain_real = terrain_real[..., start_h:start_h+512, start_w:start_w+512]
            
            # Save images if requested
            if saved_count < save_images:
                for i in range(min(bs, save_images - saved_count)):
                    # terrain_fake/real are in meters after signed-square transform
                    fake_img = terrain_fake[i, 0].cpu().numpy()
                    real_img = terrain_real[i, 0].cpu().numpy()
                    fake_img_norm = ((fake_img - fake_img.min()) / (fake_img.max() - fake_img.min() + 1e-8) * 255).astype(np.uint8)
                    real_img_norm = ((real_img - real_img.min()) / (real_img.max() - real_img.min() + 1e-8) * 255).astype(np.uint8)
                    Image.fromarray(fake_img_norm).save(results_dir / f"fake_{saved_count:05d}.png")
                    Image.fromarray(real_img_norm).save(results_dir / f"real_{saved_count:05d}.png")
                    # Save shaded relief maps (get_relief_map expects elevation in meters)
                    fake_relief = get_relief_map(fake_img, None, None, None)
                    real_relief = get_relief_map(real_img, None, None, None)
                    Image.fromarray((fake_relief * 255).astype(np.uint8)).save(results_dir / f"fake_relief_{saved_count:05d}.png")
                    Image.fromarray((real_relief * 255).astype(np.uint8)).save(results_dir / f"real_relief_{saved_count:05d}.png")
                    saved_count += 1
            
            # Disable autocast for metric calculation
            with torch.autocast(device_type=device.type, enabled=False):
                image_metric.update(_normalize_uint8_three_channel(terrain_fake.float()), real=False)
                image_metric.update(_normalize_uint8_three_channel(terrain_real.float()), real=True)
                
            pbar.update(bs)
            
    pbar.close()
    
    if metric_name == 'kid':
        score, _ = image_metric.compute()
    else:
        score = image_metric.compute()
        
    print(f"{metric_name.upper()}: {score.item()}")

if __name__ == '__main__':
    main()
