#!/usr/bin/env python3
"""
Visualization script for generating and decoding terrain latents with a diffusion model.

This script loads a diffusion UNet and scheduler from a config (e.g., diffusion_192-3.cfg),
samples latents conditioned as specified by the dataset, decodes to terrain via the
consistency decoder, and visualizes merged terrain with low-frequency inputs.

Usage examples:
    python visualize_base_dataset.py \
        --model-path checkpoints/diffusion_base-192x3/latest_checkpoint \
        --config configs/diffusion_base/diffusion_192-3.cfg

    python visualize_base_dataset.py \
        --model-path checkpoints/diffusion_base-192x3/latest_checkpoint \
        --config configs/diffusion_base/diffusion_192-3.cfg \
        --headless --num-samples 8
"""

import click
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import yaml
from confection import Config, registry
from torch.utils.data import DataLoader

from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise


class BaseDatasetVisualizer:
    def __init__(self, model, decoder, scheduler, dataset, dataloader, eval_cfg, device='cuda', headless=False):
        self.model = model
        self.decoder = decoder
        self.scheduler = scheduler
        self.dataset = dataset
        self.dataloader = dataloader
        self.eval_cfg = eval_cfg
        self.device = device
        self.headless = headless
        self.data_iter = iter(dataloader)
        self.current_batch = None
        self.current_idx = 0
        self.batch_idx = 0
        self.generator = torch.Generator(device=device).manual_seed(548)

        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('Diffusion Base Dataset Visualization', fontsize=16)

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

    @torch.no_grad()
    def _generate_samples(self, images, cond_img, cond_inputs):
        scheduler = self.scheduler
        device = images.device

        samples = torch.randn(images.shape, generator=self.generator, device=device) * scheduler.sigmas[0]

        steps = int(self.eval_cfg.get('kid_scheduler_steps', 12))
        scheduler.set_timesteps(steps)
        for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
            t = t.to(device)
            sigma = sigma.to(device)

            scaled_input = scheduler.precondition_inputs(samples, sigma)
            cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))

            if cond_img is not None:
                x = torch.cat([scaled_input, cond_img], dim=1)
            else:
                x = scaled_input

            model_output = self.model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
            samples = scheduler.step(model_output, t, samples, generator=self.generator).prev_sample

        return samples

    @torch.no_grad()
    def _decode_latents_to_terrain(self, latents, lowfreq_input):
        device = latents.device
        base_ds = self.dataset.base_dataset if hasattr(self.dataset, 'base_dataset') else self.dataset

        latents_std = base_ds.latents_std.to(device)
        latents_mean = base_ds.latents_mean.to(device)
        sigma_data = self.scheduler.config.sigma_data

        latents = (latents[:, :4] / latents_std + latents_mean) / sigma_data

        H, W = lowfreq_input.shape[-2], lowfreq_input.shape[-1]
        cond_img = torch.nn.functional.interpolate(latents[:, :4], size=(H, W), mode='nearest')

        samples = torch.zeros(latents.shape[0], 1, H, W, device=device, dtype=latents.dtype)
        t0 = torch.atan(self.scheduler.sigmas[0].to(device) / sigma_data)
        t = t0.view(1, 1, 1, 1).expand(samples.shape[0], 1, 1, 1)
        z = torch.randn_like(samples) * sigma_data
        x_t = torch.cos(t) * samples + torch.sin(t) * z
        model_input = torch.cat([x_t / sigma_data, cond_img], dim=1)
        pred = -self.decoder(model_input, noise_labels=t.flatten(), conditional_inputs=[])
        samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred

        decoded = samples / sigma_data
        residual = decoded[:, :1]

        highfreq = base_ds.denormalize_residual(residual)
        lowfreq = base_ds.denormalize_lowfreq(lowfreq_input)
        highfreq, lowfreq = laplacian_denoise(highfreq, lowfreq, sigma=5)
        return laplacian_decode(highfreq, lowfreq), lowfreq

    def load_next_batch(self):
        try:
            self.current_batch = next(self.data_iter)
            self.current_batch = recursive_to(self.current_batch, self.device)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            self.current_batch = next(self.data_iter)
            self.current_batch = recursive_to(self.current_batch, self.device)

        self.model.eval()
        images = self.current_batch['image']
        cond_img = self.current_batch.get('cond_img')
        cond_inputs = self.current_batch.get('cond_inputs')

        with torch.no_grad():
            samples = self._generate_samples(images, cond_img, cond_inputs)
            merged_terrain, lowfreq = self._decode_latents_to_terrain(samples, images[:, 4:5])
            self.merged_terrain = merged_terrain
            self.lowfreq = lowfreq

        self.current_idx = 0
        self.batch_idx += 1

    def normalize_for_display(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        tmin = tensor.min()
        tmax = tensor.max()
        if tmax > tmin:
            return (tensor - tmin) / (tmax - tmin)
        return np.zeros_like(tensor)

    def update_display(self):
        if self.current_batch is None:
            return
        batch_size = self.current_batch['image'].shape[0]
        if self.current_idx >= batch_size:
            self.current_idx = 0

        merged_terrain = self.merged_terrain[self.current_idx, 0]
        lowfreq = self.lowfreq[self.current_idx, 0]
        path = self.current_batch['path'][self.current_idx]

        for ax in self.axes.flat:
            ax.clear()

        terrain_np = merged_terrain.detach().cpu().numpy() if isinstance(merged_terrain, torch.Tensor) else merged_terrain
        self.axes[0].imshow(terrain_np, cmap='terrain')
        self.axes[0].set_title('Merged Terrain')
        self.axes[0].axis('off')

        lowfreq_np = lowfreq.detach().cpu().numpy() if isinstance(lowfreq, torch.Tensor) else lowfreq
        self.axes[1].imshow(lowfreq_np, cmap='terrain')
        self.axes[1].set_title('Lowfreq')
        self.axes[1].axis('off')

        self.fig.suptitle(
            f'Diffusion Base - Batch {self.batch_idx}, Image {self.current_idx + 1}/{batch_size}\n'
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
            filename = f'base_dataset_viz_batch{self.batch_idx}_img{self.current_idx + 1}.png'
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f'Saved visualization to {filename}')

    def save_samples(self, num_samples, output_dir='base_viz_output'):
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

        print(f'\nSaved {saved_count} samples to {output_dir}/')

    def show(self):
        plt.show()


@click.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to the trained diffusion model checkpoint directory')
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to the diffusion config file (e.g., diffusion_192-3.cfg)')
@click.option('--decoder-path', type=click.Path(exists=True), required=False, default=None,
              help='Override path to the consistency decoder model checkpoint directory')
@click.option('--batch-size', type=int, default=4, help='Batch size (default: 4)')
@click.option('--device', type=str, default='cuda', help='Device (default: cuda)')
@click.option('--num-workers', type=int, default=4, help='Dataloader workers (default: 4)')
@click.option('--headless', is_flag=True, help='Run headless and save to disk')
@click.option('--num-samples', type=int, default=10, help='Samples to save in headless mode')
@click.option('--output-dir', type=str, default='base_viz_output', help='Headless output dir')
def main(model_path, config, decoder_path, batch_size, device, num_workers, headless, num_samples, output_dir):
    if headless:
        matplotlib.use('Agg')

    build_registry()

    print(f"Loading diffusion model from {model_path}...")
    model = EDMUnet2D.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Model loaded with {model.count_parameters()} parameters")

    print(f"Loading config from {config}...")
    if config.endswith('.json'):
        import json
        with open(config, 'r') as f:
            config_dict = json.load(f)
        config_obj = Config(config_dict)
    elif config.endswith('.cfg'):
        config_obj = Config().from_disk(config)
    else:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_obj = Config(config_dict)

    # Keep only relevant sections
    keep_keys = {'model', 'scheduler', 'train_dataset', 'val_dataset', 'dataloader_kwargs', 'evaluation'}
    for key in list(config_obj.keys()):
        if key not in keep_keys:
            del config_obj[key]

    resolved = registry.resolve(config_obj, validate=False)

    # Build scheduler and decoder
    scheduler = resolved['scheduler']
    eval_cfg = resolved.get('evaluation', {})
    decoder_ckpt = decoder_path or eval_cfg.get('kid_autoencoder_path')
    if decoder_ckpt is None:
        raise click.UsageError('Decoder path not provided and not found in config under evaluation.kid_autoencoder_path')
    print(f"Loading decoder from {decoder_ckpt}...")
    decoder = EDMUnet2D.from_pretrained(decoder_ckpt).to(device)
    decoder.eval()

    # Dataset and dataloader
    try:
        val_dataset = resolved['val_dataset']
        print(f"Using validation dataset with {len(val_dataset)} samples")
    except KeyError:
        val_dataset = resolved['train_dataset']
        print(f"Using training dataset with {len(val_dataset)} samples")

    dl_kwargs = resolved.get('dataloader_kwargs', {})
    dataloader = DataLoader(
        LongDataset(val_dataset, shuffle=True),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        drop_last=True,
        **{k: v for k, v in dl_kwargs.items() if k not in {'num_workers', 'persistent_workers', 'pin_memory'}}
    )

    print(f"Created dataloader with batch size {batch_size}")

    visualizer = BaseDatasetVisualizer(model, decoder, scheduler, val_dataset, dataloader, eval_cfg, device=device, headless=headless)

    if headless:
        print("Running in headless mode...")
        print(f"Saving {num_samples} samples to {output_dir}/")
        visualizer.save_samples(num_samples, output_dir)
    else:
        print("Starting interactive visualization...")
        print("- Previous/Next: navigate images in current batch")
        print("- Prev/Next Batch: load new batches")
        print("- Save: save current visualization to PNG")
        visualizer.show()


if __name__ == '__main__':
    main()




