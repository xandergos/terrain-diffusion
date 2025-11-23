import os
import sys
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from confection import Config, registry
from tqdm import tqdm
import click


# Ensure repo root on sys.path for package imports when running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from terrain_diffusion.training.registry import build_registry  # noqa: E402
from terrain_diffusion.training.datasets import LongDataset  # noqa: E402
from terrain_diffusion.training.utils import recursive_to  # noqa: E402


def _normalize_uint8_three_channel(images: torch.Tensor) -> torch.Tensor:
    """Normalize single-channel images to uint8 [0, 255] repeated to 3 channels."""
    image_min = torch.amin(images, dim=(1, 2, 3), keepdim=True)
    image_max = torch.amax(images, dim=(1, 2, 3), keepdim=True)
    image_range = torch.maximum(image_max - image_min, torch.tensor(1.0, device=images.device))
    image_mid = (image_min + image_max) / 2
    normalized = torch.clamp(((images - image_mid) / image_range + 0.5) * 255, 0, 255)
    return normalized.repeat(1, 3, 1, 1).to(torch.uint8)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _compute_real_vs_real_metric(
    dataset,
    max_images: int,
    seed_a: int,
    seed_b: int,
    batch_size: int,
    metric: str,
    subsets: int,
    subset_size: int,
    device: torch.device,
    sigma_data: float = 1.0,
) -> tuple:
    """Compute KID or FID between two real sets using different random seeds."""
    metric_lower = metric.lower()
    if metric_lower == 'kid':
        image_metric = KernelInceptionDistance(normalize=True, subsets=subsets, subset_size=subset_size).to(device)
    elif metric_lower == 'fid':
        image_metric = FrechetInceptionDistance(normalize=True).to(device)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'kid' or 'fid'.")
    
    # Create two dataloaders with different seeds
    long_dataset_a = LongDataset(dataset, shuffle=True, seed=seed_a)
    long_dataset_b = LongDataset(dataset, shuffle=True, seed=seed_b)
    dataloader_a = DataLoader(long_dataset_a, batch_size=batch_size, num_workers=0)
    dataloader_b = DataLoader(long_dataset_b, batch_size=batch_size, num_workers=0)
    
    dataloader_a_iter = iter(dataloader_a)
    dataloader_b_iter = iter(dataloader_b)
    
    pbar = tqdm(total=max_images, desc=f"Calculating {metric.upper()}")
    
    with torch.no_grad():
        while pbar.n < pbar.total:
            batch_a = recursive_to(next(dataloader_a_iter), device=device)
            batch_b = recursive_to(next(dataloader_b_iter), device=device)
            
            images_a = batch_a['image']
            images_b = batch_b['image']
            
            # Ensure format matches expectation (B, C, H, W)
            # Some datasets might return (B, H, W) or (B, 1, H, W)
            if images_a.dim() == 3:
                images_a = images_a.unsqueeze(1)
            if images_b.dim() == 3:
                images_b = images_b.unsqueeze(1)
                
            # Normalize and update metric
            # Using the logic from sweep_diffusion_decoder.py:
            # image_metric.update(_normalize_uint8_three_channel(images.float() / scheduler.config.sigma_data), real=True)
            
            with torch.autocast(device_type=device.type, enabled=False):
                images_a_norm = _normalize_uint8_three_channel(images_a.float() / sigma_data)
                images_b_norm = _normalize_uint8_three_channel(images_b.float() / sigma_data)
                image_metric.update(images_a_norm, real=True)
                image_metric.update(images_b_norm, real=False)
            
            pbar.update(images_a.shape[0])
            
            if pbar.n >= pbar.total:
                break
    
    pbar.close()
    
    # Compute final metric
    if metric_lower == 'kid':
        kid_mean, kid_std = image_metric.compute()
        return kid_mean.item(), kid_std.item()
    else:
        fid = image_metric.compute()
        return fid.item(), None


@click.command(help="Calculate KID or FID between two real sets from the sweep dataset using different random seeds.")
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True,
              help='Path to the configuration file')
@click.option("--max-images", type=int, default=1024, show_default=True, help="Max images to load from each set.")
@click.option("--metric", type=click.Choice(['kid', 'fid'], case_sensitive=False), default='kid', show_default=True, help="Metric to compute: KID or FID.")
@click.option("--subsets", type=int, default=50, show_default=True, help="Number of random subsets for KID.")
@click.option("--subset-size", type=int, default=100, show_default=True, help="Subset size for KID.")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Batch size for loading images.")
@click.option("--seed-a", type=int, default=42, show_default=True, help="Random seed for set A.")
@click.option("--seed-b", type=int, default=123, show_default=True, help="Random seed for set B.")
def main(config_path: str, max_images: int, metric: str, subsets: int, subset_size: int, batch_size: int, seed_a: int, seed_b: int) -> None:
    build_registry()
    
    # Load config
    config = Config().from_disk(config_path)
    resolved = registry.resolve(config, validate=False)
    
    # Get sweep dataset
    sweep_dataset = resolved.get('sweep_dataset')
    if sweep_dataset is None:
        raise ValueError("Config must contain 'sweep_dataset'.")
    
    sigma_data = resolved['scheduler'].config.sigma_data

    print(f"Loaded config from: {config_path}")
    print(f"Max images per set: {max_images}")
    print(f"Metric: {metric.upper()}")
    print(f"Seed A: {seed_a}, Seed B: {seed_b}")
    
    device = _device()
    
    print(f"\nComputing {metric.upper()}...")
    result_mean, result_std = _compute_real_vs_real_metric(
        sweep_dataset,
        max_images,
        seed_a,
        seed_b,
        batch_size,
        metric,
        subsets,
        subset_size,
        device,
        sigma_data,
    )
    
    if metric.lower() == 'kid':
        print(f"\nKID (mean ± std): {result_mean:.6f} ± {result_std:.6f}")
    else:
        print(f"\nFID: {result_mean:.6f}")


if __name__ == "__main__":
    main()
