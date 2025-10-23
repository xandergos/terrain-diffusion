#!/usr/bin/env python3
"""
Visualization script for H5DecoderTerrainDataset.

This script loads the decoder dataset where:
- 'image' is the real residual image (normalized)
- 'cond_img' is the latent tensor upsampled by 8x (nearest)

For visualization, we downsample the latents by 8x (nearest), decode with a
pretrained autoencoder to reconstruct the residual, denormalize both the
reconstruction and the real residual, and display them side-by-side.
"""

import click
from diffusers import schedulers
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import yaml
import torch.nn.functional as F
from confection import Config, registry
from torch.utils.data import DataLoader
import os
import json

from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to

import torch

def scale_score(
    model_output: torch.Tensor,   
    sample: torch.Tensor,
    sigma: torch.Tensor, 
    sigma_data: float,   
    alpha: float = 1.5,  
):
    v_t = -sigma_data * model_output
    
    # Make sigma broadcast to sample shape if needed
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=sample.dtype, device=sample.device)
    while sigma.ndim < sample.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (sample.ndim - sigma.ndim)))

    sigma_data = torch.as_tensor(sigma_data, dtype=sample.dtype, device=sample.device)

    t = torch.atan(sigma / sigma_data)
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)

    x0_pred = sample * cos_t - v_t * sin_t
    noise_pred = sample * sin_t + v_t * cos_t

    x0_alpha = sample + alpha * (x0_pred - sample)

    v_t_alpha = noise_pred * cos_t - x0_alpha * sin_t

    return v_t_alpha / -sigma_data

class DecoderDatasetVisualizer:
    def __init__(self, model, dataset, dataloader, device='cuda', headless=False,
                 guide_model=None, steps=20, tile_size=128, guidance_scale=1.0, score_scaling=1.0, seed=548,
                 scheduler=None):
        self.model = model
        self.guide_model = guide_model
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.headless = headless
        self.steps = int(steps)
        self.tile_size = int(tile_size)
        self.guidance_scale = float(guidance_scale)
        self.score_scaling = float(score_scaling)
        self.seed = int(seed)

        self.is_diffusion = isinstance(self.model, EDMUnet2D)
        self.scheduler = scheduler

        self.data_iter = iter(dataloader)
        self.current_batch = None
        self.current_idx = 0
        self.batch_idx = 0

        # 1 row x 2 columns: Reconstruction vs Real
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('H5DecoderTerrainDataset Visualization', fontsize=16)

        if not headless:
            self.setup_buttons()

        self.load_next_batch()
        self.update_display()

    def setup_buttons(self):
        ax_prev = plt.axes([0.25, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.36, 0.02, 0.1, 0.04])
        ax_batch_prev = plt.axes([0.48, 0.02, 0.12, 0.04])
        ax_batch_next = plt.axes([0.61, 0.02, 0.12, 0.04])
        ax_save = plt.axes([0.75, 0.02, 0.1, 0.04])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_batch_prev = Button(ax_batch_prev, 'Prev Batch')
        self.btn_batch_next = Button(ax_batch_next, 'Next Batch')
        self.btn_save = Button(ax_save, 'Save')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_batch_prev.on_clicked(self.prev_batch)
        self.btn_batch_next.on_clicked(self.next_batch)
        self.btn_save.on_clicked(self.save_current)

    def load_next_batch(self):
        try:
            self.current_batch = next(self.data_iter)
            self.current_batch = recursive_to(self.current_batch, self.device)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            self.current_batch = next(self.data_iter)
            self.current_batch = recursive_to(self.current_batch, self.device)

        self.model.eval()
        with torch.no_grad():
            real_image = self.current_batch['image']
            if not self.is_diffusion:
                cond_img = self.current_batch['cond_img']
                _, _, H, W = cond_img.shape
                h_down, w_down = H // 8, W // 8
                latents = F.interpolate(cond_img, size=(h_down, w_down), mode='nearest')
                decoded = self.model.decode(latents)
                recon_residual = decoded[:, :1, :, :] * self.scheduler.config.sigma_data
            else:
                images = real_image
                cond_img = self.current_batch.get('cond_img')
                self.scheduler.set_timesteps(self.steps, device=self.device)
                output = torch.zeros_like(images)
                output_weights = torch.zeros_like(images)
                sigma0 = self.scheduler.sigmas[0].to(self.device)
                initial_noise = torch.randn_like(images) * sigma0
                s = self.tile_size
                w = s // 2
                num_tiles = (images.shape[-1] - w) // w
                y_grid, x_grid = torch.meshgrid(torch.arange(s, device=self.device), torch.arange(s, device=self.device), indexing='ij')
                mid = (s - 1) / 2
                epsilon = 1e-3
                distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y_grid - mid) / mid, 0, 1)
                distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x_grid - mid) / mid, 0, 1)
                weights = (distance_y * distance_x)[None, None, :, :]

                for i in range(num_tiles):
                    for j in range(num_tiles):
                        ys, ye = i * w, (i + 2) * w
                        xs, xe = j * w, (j + 2) * w
                        samples = initial_noise[..., ys:ye, xs:xe]
                        tile_cond_img = cond_img[..., ys:ye, xs:xe]
                        self.scheduler.set_timesteps(self.steps, device=self.device)
                        for t, sigma in zip(self.scheduler.timesteps, self.scheduler.sigmas):
                            t = t.to(samples.device)
                            sigma = sigma.to(samples.device)
                            scaled_input = self.scheduler.precondition_inputs(samples, sigma)
                            cnoise = self.scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))
                            model_input = torch.cat([scaled_input, tile_cond_img], dim=1)
                            if not self.guide_model or self.guidance_scale == 1.0:
                                model_output = self.model(model_input, noise_labels=cnoise, conditional_inputs=[])
                            else:
                                mo_m = self.model(model_input, noise_labels=cnoise, conditional_inputs=[])
                                mo_g = self.guide_model(model_input, noise_labels=cnoise, conditional_inputs=[])
                                model_output = mo_g + self.guidance_scale * (mo_m - mo_g)
                                
                            sigma_data = self.scheduler.config.sigma_data
                            model_output = scale_score(model_output, samples, sigma, sigma_data, alpha=self.score_scaling)
                            samples = self.scheduler.step(model_output, t, samples).prev_sample

                        output[..., ys:ye, xs:xe] += samples * weights
                        output_weights[..., ys:ye, xs:xe] += weights

                recon_residual = output / output_weights

            recon_residual = self.dataset.denormalize_residual(recon_residual)
            real_residual = self.dataset.denormalize_residual(real_image)

            self.recon_residual = recon_residual
            self.real_residual = real_residual

        self.current_idx = 0
        self.batch_idx += 1

    def update_display(self):
        if self.current_batch is None:
            return

        batch_size = self.current_batch['image'].shape[0]
        if self.current_idx >= batch_size:
            self.current_idx = 0

        recon = self.recon_residual[self.current_idx, 0]
        real = self.real_residual[self.current_idx, 0]
        path = self.current_batch['path'][self.current_idx]

        for ax in self.axes.flat:
            ax.clear()

        recon_np = recon.cpu().numpy()
        real_np = real.cpu().numpy()
        vmin, vmax = real_np.min(), real_np.max()
        self.axes[0].imshow(recon_np, cmap='terrain', vmin=vmin, vmax=vmax)
        self.axes[0].set_title('Reconstructed (AE from latents)')
        self.axes[0].axis('off')

        self.axes[1].imshow(real_np, cmap='terrain', vmin=vmin, vmax=vmax)
        self.axes[1].set_title('Real Residual')
        self.axes[1].axis('off')

        self.fig.suptitle(
            f'H5DecoderTerrainDataset - Batch {self.batch_idx}, Image {self.current_idx + 1}/{batch_size}\n'
            f'Path: {path}',
            fontsize=14
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.92)
        self.fig.canvas.draw()

    def prev_image(self, event):
        if self.current_batch is not None:
            batch_size = self.current_batch['image'].shape[0]
            self.current_idx = (self.current_idx - 1) % batch_size
            self.update_display()

    def next_image(self, event):
        if self.current_batch is not None:
            batch_size = self.current_batch['image'].shape[0]
            self.current_idx = (self.current_idx + 1) % batch_size
            self.update_display()

    def prev_batch(self, event):
        self.load_next_batch()
        self.update_display()

    def next_batch(self, event):
        self.load_next_batch()
        self.update_display()

    def save_current(self, event):
        if self.current_batch is not None:
            filename = f'decoder_dataset_viz_batch{self.batch_idx}_img{self.current_idx + 1}.png'
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f'Saved visualization to {filename}')

    def save_samples(self, num_samples, output_dir='decoder_viz_output'):
        import os
        os.makedirs(output_dir, exist_ok=True)

        saved_count = 0
        while saved_count < num_samples:
            batch_size = self.current_batch['image'].shape[0]
            for idx in range(batch_size):
                if saved_count >= num_samples:
                    break
                self.current_idx = idx
                self.update_display()
                filename = os.path.join(output_dir, f'sample_{saved_count:04d}.png')
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                print(f'Saved {filename}')
                saved_count += 1
            if saved_count < num_samples:
                self.load_next_batch()

        print(f"\nSaved {saved_count} samples to {output_dir}/")

    def show(self):
        plt.show()


@click.command()
@click.option('--model', 'model_path', type=click.Path(exists=True), required=True,
              help='Path to a saved model (autoencoder or diffusion decoder)')
@click.option('--guide-model', 'guide_model_path', type=click.Path(exists=True), required=False,
              help='Optional path to a guide model for classifier-free guidance')
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to a diffusion config with *_dataset section (e.g., diffusion_decoder_64-3.cfg)')
@click.option('--batch-size', type=int, default=4, help='Batch size (default: 4)')
@click.option('--device', type=str, default='cuda', help='Device (default: cuda)')
@click.option('--num-workers', type=int, default=4, help='Dataloader workers (default: 4)')
@click.option('--headless', is_flag=True, help='Headless mode: save to disk')
@click.option('--num-samples', type=int, default=10, help='Samples to save in headless mode')
@click.option('--output-dir', type=str, default='decoder_viz_output', help='Output dir for headless mode')
@click.option('--steps', type=int, default=20, help='Diffusion steps for decoder models')
@click.option('--tile-size', type=int, default=128, help='Tile size for tiled sampling')
@click.option('--guidance-scale', type=float, default=1.0, help='Guidance scale when using guide model')
@click.option('--score-scaling', type=float, default=1.0, help='Score scaling multiplier')
@click.option('--seed', type=int, default=548, help='Random seed')
def main(model_path, guide_model_path, config, batch_size, device, num_workers, headless, num_samples, output_dir,
         steps, tile_size, guidance_scale, score_scaling, seed):
    if headless:
        matplotlib.use('Agg')

    build_registry()

    def _read_class_name(path):
        cfg = os.path.join(path, 'config.json')
        if cfg is None:
            raise ValueError(f"No config.json found under {path}")
        with open(cfg, 'r') as f:
            return json.load(f).get('_class_name')

    cls_name = _read_class_name(model_path)
    if cls_name == 'EDMAutoencoder':
        print(f"Loading autoencoder from {model_path}...")
        model = EDMAutoencoder.from_pretrained(model_path)
    elif cls_name == 'EDMUnet2D':
        print(f"Loading diffusion decoder from {model_path}...")
        model = EDMUnet2D.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model class: {cls_name}")
    model = model.to(device)
    model.eval()
    try:
        n_params = model.count_parameters()
    except Exception:
        n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {n_params} parameters")

    guide_model = None
    if guide_model_path:
        g_cls_name = _read_class_name(guide_model_path)
        if g_cls_name != 'EDMUnet2D':
            raise ValueError("Guide model must be a diffusion decoder (EDMUnet2D)")
        print(f"Loading guide model from {guide_model_path}...")
        guide_model = EDMUnet2D.from_pretrained(guide_model_path).to(device)
        guide_model.eval()

    print(f"Loading config from {config}...")
    if config.endswith('.json'):
        with open(config, 'r') as f:
            config_dict = json.load(f)
        config_obj = Config(config_dict)
    elif config.endswith('.cfg'):
        config_obj = Config().from_disk(config)
    else:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_obj = Config(config_dict)

    # Keep only dataset and scheduler config
    for key in list(config_obj.keys()):
        if not (key.endswith('_dataset') or key == 'scheduler'):
            del config_obj[key]

    resolved = registry.resolve(config_obj, validate=False)

    try:
        val_dataset = resolved['val_dataset']
        print(f"Using validation dataset with {len(val_dataset)} samples")
    except KeyError:
        try:
            val_dataset = resolved['train_dataset']
            print(f"Using training dataset with {len(val_dataset)} samples")
        except KeyError:
            print("No dataset found in config. Please check your config file.")
            return

    dataloader = DataLoader(
        LongDataset(val_dataset, shuffle=True),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Created dataloader with batch size {batch_size}")

    scheduler = resolved.get('scheduler') if isinstance(resolved, dict) else None
    visualizer = DecoderDatasetVisualizer(
        model=model,
        dataset=val_dataset,
        dataloader=dataloader,
        device=device,
        headless=headless,
        guide_model=guide_model,
        steps=steps,
        tile_size=tile_size,
        guidance_scale=guidance_scale,
        score_scaling=score_scaling,
        seed=seed,
        scheduler=scheduler,
    )

    if headless:
        print("Running in headless mode...")
        print(f"Saving {num_samples} samples to {output_dir}/")
        visualizer.save_samples(num_samples, output_dir)
    else:
        print("Starting interactive visualization...")
        print("Use the buttons to navigate:")
        print("- Previous/Next: Navigate within current batch")
        print("- Prev/Next Batch: Load new batches")
        print("- Save: Save current visualization to PNG file")
        print("- Close the window to exit")
        visualizer.show()


if __name__ == '__main__':
    main()


