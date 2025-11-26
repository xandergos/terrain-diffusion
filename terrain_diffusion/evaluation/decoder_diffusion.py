import click
import torch
import numpy as np
from accelerate import Accelerator
from confection import Config, registry
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.data.laplacian_encoder import laplacian_decode
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import recursive_to

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

@click.command()
@click.option('-m', '--model', 'model_path', type=click.Path(exists=True), help='Path to the pretrained decoder model checkpoint directory', default='checkpoints/models/diffusion_decoder-64x3')
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), help='Path to the diffusion decoder configuration file', default='configs/diffusion_decoder/diffusion_decoder_64-3.cfg')
@click.option('--num-images', type=int, default=50000, help='Number of images to evaluate')
@click.option('--batch-size', type=int, default=16, help='Batch size for evaluation')
@click.option('--steps', type=int, default=32, help='Number of diffusion steps (overrides config)')
@click.option('--tile-size', type=int, default=512, help='Tile size for processing')
@click.option('--tile-stride', type=int, default=None, help='Tile stride for processing. Defaults to tile_size // 2')
@click.option('--metric', type=click.Choice(['fid', 'kid'], case_sensitive=False), default='fid', help='Metric to evaluate')
@click.option('-gm', '--guide-model', 'guide_model_path', type=click.Path(exists=True), help='Path to the guide model checkpoint directory', default='checkpoints/models/diffusion_decoder-32x3')
@click.option('--guidance-scale', type=float, default=1.26, help='Guidance scale')
def main(model_path, config_path, num_images, batch_size, steps, tile_size, tile_stride, metric, guide_model_path, guidance_scale):
    """Evaluate diffusion decoder using FID/KID."""
    build_registry()
    
    # Load config
    config = Config().from_disk(config_path)
    
    # Remove all keys from config except 'scheduler' and 'results_dataset'
    kept_keys = {'scheduler', 'results_dataset'}
    keys_to_delete = [k for k in config.keys() if k not in kept_keys]
    for k in keys_to_delete:
        del config[k]
        
    resolved = registry.resolve(config, validate=False)
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = EDMUnet2D.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    model = torch.compile(model)

    guide_model = None
    if guide_model_path:
        print(f"Loading guide model from {guide_model_path}...")
        guide_model = EDMUnet2D.from_pretrained(guide_model_path)
        guide_model = guide_model.to(device)
        guide_model.eval()
        guide_model = torch.compile(guide_model)
        
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
    
    print(f"Evaluating {metric_name.upper()} on {num_images} images with {scheduler_steps} steps...")
    
    generator = torch.Generator(device=device).manual_seed(548)
    weights = _get_weights(tile_size).to(device)
    
    if tile_stride is None:
        tile_stride = tile_size // 2

    pbar = tqdm(total=num_images, desc=f"Calculating {metric_name.upper()}")
    
    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            try:
                batch = recursive_to(next(val_dataloader_iter), device=device)
            except StopIteration:
                val_dataloader_iter = iter(val_dataloader)
                batch = recursive_to(next(val_dataloader_iter), device=device)
                
            images = batch['image']
            lowfreq = batch['lowfreq']
            cond_img = batch.get('cond_img')
            
            # Adjust batch size if we need fewer images to finish
            remaining = num_images - pbar.n
            if images.shape[0] > remaining:
                images = images[:remaining]
                lowfreq = lowfreq[:remaining]
                cond_img = cond_img[:remaining]
            
            current_batch_size = images.shape[0]
            
            output = torch.zeros(images.shape, device=device)
            output_weights = torch.zeros(images.shape, device=device)
            initial_noise = torch.randn(images.shape, device=device, generator=generator) * scheduler.sigmas[0]
            
            num_tiles = (images.shape[-1] - tile_size) // tile_stride + 1
            
            # Sampling loop
            for i in range(num_tiles):
                for j in range(num_tiles):
                    h_start = i * tile_stride
                    h_end = h_start + tile_size
                    w_start = j * tile_stride
                    w_end = w_start + tile_size
                    
                    samples = initial_noise[..., h_start:h_end, w_start:w_end]
                    tile_cond_img = cond_img[..., h_start:h_end, w_start:w_end]
                    
                    scheduler.set_timesteps(scheduler_steps)
                    
                    for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                        t = t.to(device)
                        sigma = sigma.to(device)
                        
                        scaled_input = scheduler.precondition_inputs(samples, sigma)
                        cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))
                        
                        model_input = torch.cat([scaled_input, tile_cond_img], dim=1)
                        
                        if not guide_model or guidance_scale == 1.0:
                            model_output = model(model_input, noise_labels=cnoise, conditional_inputs=[])
                        else:
                            model_output_m = model(model_input, noise_labels=cnoise, conditional_inputs=[])
                            model_output_g = guide_model(model_input, noise_labels=cnoise, conditional_inputs=[])
                            model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
                            
                        samples = scheduler.step(model_output, t, samples).prev_sample
                        
                    output[..., h_start:h_end, w_start:w_end] += samples * weights
                    output_weights[..., h_start:h_end, w_start:w_end] += weights
            
            output = output / output_weights / config['results_dataset']['sigma_data']
            images = images / config['results_dataset']['sigma_data']
            
            output_full = laplacian_decode(output * config['results_dataset']['residual_std'] + config['results_dataset']['residual_mean'], lowfreq, extrapolate=True)
            images_full = laplacian_decode(images * config['results_dataset']['residual_std'] + config['results_dataset']['residual_mean'], lowfreq, extrapolate=True)
            
            output_full = torch.sign(output_full) * torch.square(output_full)
            images_full = torch.sign(images_full) * torch.square(images_full)
            
            # Disable autocast for metric calculation to prevent NaN in Inception features
            with torch.autocast(device_type=device.type, enabled=False):
                image_metric.update(_normalize_uint8_three_channel(output_full.float()), real=False)
                image_metric.update(_normalize_uint8_three_channel(images_full.float()), real=True)
                
            pbar.update(current_batch_size)
            
    pbar.close()
    
    if metric_name == 'kid':
        score, _ = image_metric.compute()
    else:
        score = image_metric.compute()
        
    print(f"{metric_name.upper()}: {score.item()}")

if __name__ == '__main__':
    main()

