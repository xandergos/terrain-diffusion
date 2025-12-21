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
@click.option('--save-images', type=int, default=0, help='Number of images to save to results directory')
@click.option('--experiment-name', type=str, default='infinite_consistency_naive', help='Name of experiment folder in results/')
def main(model_path, config_path, num_images, batch_size, decoder_batch_size, metric, inter_t, decoder_inter_t, save_images, experiment_name):
    """Evaluate base diffusion using FID/KID on decoded terrain."""
    build_registry()
    
    # Load config
    config = Config().from_disk(config_path)
    kept_keys = {'scheduler', 'results_dataset', 'evaluation'}
    keys_to_delete = [k for k in config.keys() if k not in kept_keys]
    for k in keys_to_delete:
        del config[k]
    config['results_dataset']['crop_size'] = 192  # same dataset as infinite_consistency.py
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
            
            # Central crop cond_inputs from 8x8 to 6x6 for 2x2 tiling
            if cond_inputs.ndim == 4:
                cond_inputs = cond_inputs[..., 1:7, 1:7]
            
            # Adjust batch size if we need fewer images to finish
            remaining = num_images - pbar.n
            if images.shape[0] > remaining:
                images = images[:remaining]
                cond_inputs = cond_inputs[:remaining]
                histogram_raw = histogram_raw[:remaining]
                
            bs = images.shape[0]
            
            # Naive tiling: generate 4 independent tiles per batch item
            # Each tile is processed independently through both base model and decoder
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

                B, C, _, _ = images.shape
                tile_size = 64  # model expects 64x64 tiles
                
                # Generate 4 tiles per batch item (2x2 grid)
                num_tiles = 4
                tile_shape = (B * num_tiles, C, tile_size, tile_size)
                
                # Prepare conditioning for all tiles (4 independent regions)
                tile_conds_list = []
                if cond_inputs.ndim == 4:
                    for ic in range(2):
                        for jc in range(2):
                            tile_cond = cond_inputs[..., ic:ic+4, jc:jc+4]
                            tile_cond = _process_cond_img(tile_cond, histogram_raw, torch.zeros(7, device=device), torch.ones(7, device=device), torch.zeros(bs, 1, device=device))
                            tile_conds_list.append(tile_cond)
                else:
                    tile_conds_list = [cond_inputs] * num_tiles
                
                # Sample each tile separately to save memory
                samples = torch.zeros(tile_shape, device=device, dtype=dtype)
                for tile_idx in range(num_tiles):
                    tile_cond = [tile_conds_list[tile_idx]]
                    tile_sample = torch.zeros((B, C, tile_size, tile_size), device=device, dtype=dtype)
                    
                    for step, t_scalar in enumerate(t_scalars):
                        step_noise = torch.randn((B, C, tile_size, tile_size), generator=generator, device=device, dtype=dtype)
                        z = step_noise * sigma_data
                        
                        t = t_scalar.view(1, 1, 1, 1).expand(B, 1, 1, 1)
                        x_t = torch.cos(t) * tile_sample + torch.sin(t) * z
                        
                        model_in = x_t / sigma_data
                        pred = -model(model_in, noise_labels=t.flatten(), conditional_inputs=tile_cond)
                        tile_sample = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
                    
                    # Store samples grouped by batch item: [b0_t0, b0_t1, b0_t2, b0_t3, b1_t0, ...]
                    for b in range(B):
                        samples[b * num_tiles + tile_idx] = tile_sample[b]

                samples = samples / scheduler.config.sigma_data

            # Decode each tile independently to 512x512, then stitch
            tile_terrain_size = 64 * 8  # 64 * 8 = 512
            terrain_fake = torch.zeros(bs, 1, tile_terrain_size * 2, tile_terrain_size * 2, device=device)
            
            with accelerator.autocast():
                # Decode all tiles
                all_decoded = torch.zeros(B * num_tiles, 1, tile_terrain_size, tile_terrain_size, device=device)
                total_tiles = B * num_tiles
                for i in range(0, total_tiles, decoder_batch_size):
                    end_i = min(i + decoder_batch_size, total_tiles)
                    in_samples = samples[i:end_i]
                    all_decoded[i:end_i] = _decode_latents_to_terrain(in_samples, val_dataset, decoder_model, scheduler, generator, inter_t=decoder_inter_t_list)
                
                # Stitch tiles into 2x2 grid per batch item
                for b in range(bs):
                    tile_idx = b * num_tiles
                    # Top-left
                    terrain_fake[b, :, :tile_terrain_size, :tile_terrain_size] = all_decoded[tile_idx]
                    # Top-right
                    terrain_fake[b, :, :tile_terrain_size, tile_terrain_size:] = all_decoded[tile_idx + 1]
                    # Bottom-left
                    terrain_fake[b, :, tile_terrain_size:, :tile_terrain_size] = all_decoded[tile_idx + 2]
                    # Bottom-right
                    terrain_fake[b, :, tile_terrain_size:, tile_terrain_size:] = all_decoded[tile_idx + 3]
            
            # Central crop from 1024x1024 to 512x512
            crop_start = (terrain_fake.shape[-1] - 512) // 2
            terrain_fake = terrain_fake[..., crop_start:crop_start+512, crop_start:crop_start+512]
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
                    # Save shaded relief maps
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
