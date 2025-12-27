import click
import torch
import numpy as np
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
from terrain_diffusion.data.laplacian_encoder import laplacian_decode
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.inference.relief_map import get_relief_map

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
@click.option('-m', '--model', 'model_path', type=str, default=None, help='Path to pretrained decoder model (local or HuggingFace repo)')
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True, help='Path to the consistency decoder configuration file', default='configs/diffusion_decoder/consistency_decoder_64-3.cfg')
@click.option('--num-images', type=int, default=50000, help='Number of images to evaluate')
@click.option('--batch-size', type=int, default=10, help='Batch size for evaluation')
@click.option('--intermediate-sigma', type=float, default=0.065, help='Intermediate sigma for 2-step sampling. Use 0 or less for one-step model.')
@click.option('--metric', type=click.Choice(['fid', 'kid'], case_sensitive=False), default='fid', help='Metric to evaluate')
@click.option('--save-images', type=int, default=0, help='Number of images to save to results directory')
@click.option('--experiment-name', type=str, default='tiled_decoder_consistency', help='Name of experiment folder in results/')
def main(model_path, config_path, num_images, batch_size, intermediate_sigma, metric, save_images, experiment_name):
    """Evaluate tiled consistency decoder using FID/KID with 2x2 tiling and center crop."""
    build_registry()
    
    # Fixed tiling parameters
    image_size = 896
    tile_size = 512
    tile_stride = 384
    crop_size = 512
    
    # Load config
    config = Config().from_disk(config_path)
    
    # Override crop_size to 896 for 2x2 tiling
    assert 'crop_size' in config['results_dataset'], 'crop_size is not in results_dataset'
    config['results_dataset']['crop_size'] = image_size
    
    # Remove all keys from config except 'scheduler' and 'results_dataset'
    kept_keys = {'scheduler', 'results_dataset'}
    keys_to_delete = [k for k in config.keys() if k not in kept_keys]
    for k in keys_to_delete:
        del config[k]
    resolved = registry.resolve(config, validate=False)
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    # Resolve model path (user override -> local default -> HuggingFace)
    resolved_model_path = resolve_model_path(model_path, *MODEL_PATHS["decoder"])
    
    # Load Model
    print(f"Loading model from {resolved_model_path}...")
    model = EDMUnet2D.from_pretrained(resolved_model_path)
    model = model.to(device)
    model.eval()
    
    model = accelerator.prepare(model)
    
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
    
    print(f"Evaluating {metric_name.upper()} on {num_images} images with 2x2 tiling (tile={tile_size}, stride={tile_stride})...")
    
    # Setup results directory for image saving
    results_dir = Path("results") / experiment_name
    if save_images > 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    
    generator = torch.Generator(device=device).manual_seed(548)
    weights = _get_weights(tile_size).to(device)
    
    sigma_data = config['results_dataset']['sigma_data']
    init_t = torch.tensor(np.arctan(scheduler.sigmas[0] / sigma_data), device=device)
    if intermediate_sigma > 0:
        intermediate_t = torch.tensor(np.arctan(intermediate_sigma / sigma_data), device=device)
    else:
        intermediate_t = None
    
    pbar = tqdm(total=num_images, desc=f"Calculating {metric_name.upper()}")
    
    # Center crop boundaries
    crop_start = (image_size - crop_size) // 2
    crop_end = crop_start + crop_size
    
    with torch.no_grad(), accelerator.autocast():
        while pbar.n < pbar.total:
            try:
                batch = recursive_to(next(val_dataloader_iter), device=device)
            except StopIteration:
                val_dataloader_iter = iter(val_dataloader)
                batch = recursive_to(next(val_dataloader_iter), device=device)
                
            images = batch['image']
            lowfreq_padded = batch['lowfreq_padded']
            cond_img = batch.get('cond_img')
            
            # Adjust batch size if we need fewer images to finish
            remaining = num_images - pbar.n
            if images.shape[0] > remaining:
                images = images[:remaining]
                lowfreq_padded = lowfreq_padded[:remaining]
                cond_img = cond_img[:remaining]
                
            current_batch_size = images.shape[0]
            
            output = torch.zeros(images.shape, device=device)
            output_weights = torch.zeros(images.shape, device=device)
            
            # 2x2 tiling
            for i in range(2):
                for j in range(2):
                    h_start = i * tile_stride
                    h_end = h_start + tile_size
                    w_start = j * tile_stride
                    w_end = w_start + tile_size
                    
                    samples = torch.zeros(
                        (current_batch_size, images.shape[1], tile_size, tile_size),
                        device=device
                    )
                    tile_cond_img = cond_img[..., h_start:h_end, w_start:w_end]
                    
                    # First timestep: start from high noise
                    if intermediate_t is not None:
                        t_values = [init_t, intermediate_t]
                    else:
                        t_values = [init_t]
                    
                    for t_scalar in t_values:
                        t = t_scalar.view(1, 1, 1, 1).expand(samples.shape[0], 1, 1, 1).to(device=device, dtype=images.dtype)
                        z = torch.randn(samples.shape, generator=generator, device=device) * sigma_data
                        x_t = torch.cos(t) * samples + torch.sin(t) * z
                        model_input = x_t / sigma_data
                        if tile_cond_img is not None:
                            model_input = torch.cat([model_input, tile_cond_img], dim=1)
                        
                        pred = -model(model_input, noise_labels=t.flatten(), conditional_inputs=[])
                        samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
                        
                    output[..., h_start:h_end, w_start:w_end] += samples * weights
                    output_weights[..., h_start:h_end, w_start:w_end] += weights
            
            output = output / output_weights / sigma_data
            images = images / sigma_data
            
            output_full = laplacian_decode(output * config['results_dataset']['residual_std'] + config['results_dataset']['residual_mean'], lowfreq_padded, pre_padded=True)
            images_full = laplacian_decode(images * config['results_dataset']['residual_std'] + config['results_dataset']['residual_mean'], lowfreq_padded, pre_padded=True)
            
            output_full = torch.sign(output_full) * torch.square(output_full)
            images_full = torch.sign(images_full) * torch.square(images_full)
            
            # Center crop to 512x512
            output_full = output_full[..., crop_start:crop_end, crop_start:crop_end]
            images_full = images_full[..., crop_start:crop_end, crop_start:crop_end]
            
            # Disable autocast for metric calculation to prevent NaN in Inception features
            with torch.autocast(device_type=device.type, enabled=False):
                image_metric.update(_normalize_uint8_three_channel(output_full.float()), real=False)
                image_metric.update(_normalize_uint8_three_channel(images_full.float()), real=True)
            
            # Save images if requested
            if saved_count < save_images:
                for i in range(min(current_batch_size, save_images - saved_count)):
                    # output_full/images_full are in meters after signed-square transform
                    fake_img = output_full[i, 0].cpu().numpy()
                    real_img = images_full[i, 0].cpu().numpy()
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
                
            pbar.update(current_batch_size)
            
    pbar.close()
    
    if metric_name == 'kid':
        score, _ = image_metric.compute()
    else:
        score = image_metric.compute()
        
    print(f"{metric_name.upper()}: {score.item()}")

if __name__ == '__main__':
    main()

