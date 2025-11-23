import click
import torch
import numpy as np
import math
import os
from accelerate import Accelerator
from confection import Config, registry
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

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
    #highfreq, lowfreq = laplacian_denoise(highfreq, lowfreq, sigma=5)
    
    # When calculating FID/KID, we can approximate padding with extrapolation. But typically we can get padding from InfiniteDiffusion directly.
    return laplacian_decode(highfreq, lowfreq, extrapolate=True)

@click.command()
@click.option('-m', '--model', 'model_path', type=click.Path(exists=True), help='Path to the pretrained base model checkpoint directory', default='checkpoints/models/diffusion_base-192x3')
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), help='Path to the base diffusion configuration file', default='configs/diffusion_base/diffusion_192-3.cfg')
@click.option('--num-images', type=int, default=50000, help='Number of images to evaluate')
@click.option('--batch-size', type=int, default=40, help='Batch size for evaluation')
@click.option('--decoder-batch-size', type=int, default=10, help='Batch size for decoder evaluation')
@click.option('--steps', type=int, default=32, help='Number of diffusion steps (overrides config)')
@click.option('--metric', type=click.Choice(['fid', 'kid'], case_sensitive=False), default='fid', help='Metric to evaluate')
@click.option('-gm', '--guide-model', 'guide_model_path', type=click.Path(exists=True), help='Path to the guide model checkpoint directory', default='checkpoints/models/diffusion_base-128x3')
@click.option('--guidance-scale', type=float, default=2.15, help='Guidance scale')
@click.option('--inter-t', type=float, default=0.13, help='Intermediate t for 2-step decoder sampling.')
def main(model_path, config_path, num_images, batch_size, decoder_batch_size, steps, metric, guide_model_path, guidance_scale, inter_t):
    """Evaluate base diffusion using FID/KID on decoded terrain."""
    build_registry()
    
    # Load config
    config = Config().from_disk(config_path)
    kept_keys = {'scheduler', 'results_dataset', 'evaluation'}
    keys_to_delete = [k for k in config.keys() if k not in kept_keys]
    for k in keys_to_delete:
        del config[k]
    resolved = registry.resolve(config, validate=False)
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    # Load Models
    print(f"Loading model from {model_path}...")
    model = EDMUnet2D.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    guide_model = None
    if guide_model_path:
        print(f"Loading guide model from {guide_model_path}...")
        guide_model = EDMUnet2D.from_pretrained(guide_model_path)
        guide_model = guide_model.to(device)
        guide_model.eval()
        
    # Load decoder model
    ae_path = resolved['evaluation']['kid_autoencoder_path']
    print(f"Loading decoder model from {ae_path}...")
    if not os.path.isdir(ae_path):
        raise ValueError(f"Autoencoder path not found: {ae_path}")
    decoder_model = EDMUnet2D.from_pretrained(ae_path).to(device)
    decoder_model.eval()
    
    model, guide_model, decoder_model = accelerator.prepare(model, guide_model, decoder_model)
    
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
        
    scheduler_steps = steps
    inter_t = [inter_t] if inter_t is not None else None
    
    print(f"Evaluating {metric_name.upper()} on {num_images} images with {scheduler_steps} steps...")
    
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
                samples = sample_base_diffusion(
                    model=model,
                    scheduler=scheduler,
                    shape=(bs, 5, 64, 64), # Latents + lowfreq
                    cond_inputs=cond_inputs,
                    cond_means=torch.zeros(7, device=device),
                    cond_stds=torch.ones(7, device=device),
                    noise_level=torch.zeros(bs, 1, device=device),
                    histogram_raw=histogram_raw,
                    steps=int(scheduler_steps),
                    guide_model=guide_model,
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    tile_size=64
                )


            terrain_fake = torch.zeros(bs, 1, samples.shape[-2]*8, samples.shape[-1]*8, device=device)
            with accelerator.autocast():
                assert terrain_fake.shape[0] % decoder_batch_size == 0
                for i in range(terrain_fake.shape[0]//decoder_batch_size):
                    in_samples = samples[i*decoder_batch_size:(i+1)*decoder_batch_size]
                    terrain_fake[i*decoder_batch_size:(i+1)*decoder_batch_size] = _decode_latents_to_terrain(in_samples, val_dataset, decoder_model, scheduler, generator, inter_t=inter_t)
            terrain_fake = torch.sign(terrain_fake) * torch.square(terrain_fake)
            
            # Real terrain
            ground_truth = batch['ground_truth']
            if ground_truth.shape[0] > bs:
                 ground_truth = ground_truth[:bs]
            terrain_real = ground_truth
            terrain_real = torch.sign(terrain_real) * torch.square(terrain_real)
            
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
