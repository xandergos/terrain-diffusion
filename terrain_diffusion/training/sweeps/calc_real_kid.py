import os
import sys
import random
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance
import click


# Ensure repo root on sys.path for package imports when running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from terrain_diffusion.data.laplacian_encoder import laplacian_decode  # noqa: E402


def iter_ground_truth_images(
    h5_path: str,
    crop_size: int,
    max_images: Optional[int] = None,
    split: Optional[str] = None,
    resolutions: Optional[Iterable[int]] = None,
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        res_keys = [k for k in f.keys() if (k.isdigit() and (resolutions is None or int(k) in set(resolutions)))]
        for res_key in res_keys:
            res_group = f[res_key]
            for chunk_id in res_group:
                chunk_group = res_group[chunk_id]
                for subchunk_id in chunk_group:
                    g = chunk_group[subchunk_id]
                    if "lowfreq" not in g or "residual" not in g:
                        continue
                    # Filter by split using the 'latent' dataset attrs if present
                    if "latent" in g and split is not None:
                        try:
                            if g["latent"].attrs.get("split", None) != split:
                                continue
                        except Exception:
                            pass

                    lowfreq_ds = g["lowfreq"]
                    H, W = lowfreq_ds.shape

                    # Center crop (eval-style)
                    li = (H - crop_size) // 2
                    lj = (W - crop_size) // 2
                    h = w = crop_size

                    # Load crops
                    lowfreq = torch.from_numpy(lowfreq_ds[li:li + h, lj:lj + w])[None].float()
                    residual_ds = g["residual"]
                    residual = torch.from_numpy(
                        residual_ds[li * 8:li * 8 + h * 8, lj * 8:lj * 8 + w * 8]
                    )[None].float()

                    # Decode ground truth exactly as in dataset (no flip/rot for eval)
                    gt = laplacian_decode(residual, lowfreq)
                    img = gt.squeeze(0).cpu().numpy()  # [H*8, W*8]
                    images.append(img.astype(np.float32))

                    if max_images is not None and len(images) >= max_images:
                        return images
    return images


def _normalize_and_process_terrain(terrain: torch.Tensor) -> torch.Tensor:
    """Match consistency.py: per-image min/max -> uint8 [0,255], 3-channel."""
    terrain_min = torch.amin(terrain, dim=(1, 2, 3), keepdim=True)
    terrain_max = torch.amax(terrain, dim=(1, 2, 3), keepdim=True)
    terrain_range = torch.maximum(terrain_max - terrain_min, torch.tensor(1.0, device=terrain.device))
    terrain_mid = (terrain_min + terrain_max) / 2
    terrain_norm = torch.clamp(((terrain - terrain_mid) / terrain_range + 0.5) * 255, 0, 255)
    terrain_norm = terrain_norm.repeat(1, 3, 1, 1)
    return terrain_norm.to(torch.uint8)


@torch.no_grad()
def _update_kid(kid: KernelInceptionDistance, images: List[np.ndarray], real: bool, batch_size: int, device: torch.device) -> None:
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        x = torch.stack([torch.from_numpy(im).unsqueeze(0).float() for im in chunk], dim=0).to(device)
        x = _normalize_and_process_terrain(x)
        kid.update(x, real=real)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command(help="Calculate KID between two real sets decoded from a single HDF5 file.")
@click.option("--h5", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True, help="Path to HDF5 file.")
@click.option("--crop-size", type=int, default=512, show_default=True, help="Lowfreq crop size (any positive integer).")
@click.option("--max-images-a", type=int, default=1024, show_default=True, help="Max images to load from set A.")
@click.option("--max-images-b", type=int, default=1024, show_default=True, help="Max images to load from set B.")
@click.option("--split-a", type=str, default='train', show_default=True, help="Optional split for set A (e.g., train/val/test).")
@click.option("--split-b", type=str, default='train', show_default=True, help="Optional split for set B (e.g., train/val/test).")
@click.option("--resolutions-a", type=str, default=90, show_default=True, help="Comma-separated resolutions for A (e.g., 90,480).")
@click.option("--resolutions-b", type=str, default=90, show_default=True, help="Comma-separated resolutions for B (e.g., 90,480).")
@click.option("--subsets", type=int, default=50, show_default=True, help="Number of random subsets for KID.")
@click.option("--subset-size", type=int, default=100, show_default=True, help="Subset size for KID.")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Batch size for metric updates.")
@click.option("--full", is_flag=True, help="Use full decoded images.")
def main(h5: str, crop_size: int, max_images_a: int, max_images_b: int, split_a: Optional[str], split_b: Optional[str],
         resolutions_a: Optional[str], resolutions_b: Optional[str], subsets: int, subset_size: int, batch_size: int, full: bool) -> None:
    if crop_size <= 0:
        raise ValueError("--crop-size must be positive.")

    res_a = [int(x.strip()) for x in resolutions_a.split(",")] if resolutions_a else None
    res_b = [int(x.strip()) for x in resolutions_b.split(",")] if resolutions_b else None

    imgs_a = iter_ground_truth_images(
        h5,
        crop_size=crop_size,
        max_images=max_images_a,
        split=split_a,
        resolutions=res_a,
    )
    imgs_b = iter_ground_truth_images(
        h5,
        crop_size=crop_size,
        max_images=max_images_b,
        split=split_b,
        resolutions=res_b,
        full=full,
    )

    if len(imgs_a) == 0 or len(imgs_b) == 0:
        raise RuntimeError("No images loaded from one or both sets. Check split, resolutions, and crop-size.")

    device = _device()
    kid = KernelInceptionDistance(normalize=True, subsets=subsets, subset_size=subset_size).to(device)
    _update_kid(kid, imgs_a, real=True, batch_size=batch_size, device=device)
    _update_kid(kid, imgs_b, real=False, batch_size=batch_size, device=device)
    kid_mean, kid_std = kid.compute()
    print(f"KID (mean ± std): {kid_mean.item():.6f} ± {kid_std.item():.6f}")


if __name__ == "__main__":
    main()


