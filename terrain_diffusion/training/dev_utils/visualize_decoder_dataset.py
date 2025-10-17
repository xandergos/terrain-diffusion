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
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import yaml
import torch.nn.functional as F
from confection import Config, registry
from torch.utils.data import DataLoader

from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to


class DecoderDatasetVisualizer:
    def __init__(self, model, dataset, dataloader, device='cuda', headless=False):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.headless = headless
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

        # Prepare reconstruction from downsampled latents
        self.model.eval()
        with torch.no_grad():
            real_image = self.current_batch['image']  # [B, 1, H, W], normalized
            cond_img = self.current_batch['cond_img']  # [B, C, H, W], upsampled latents

            _, _, H, W = cond_img.shape
            h_down, w_down = H // 8, W // 8
            latents = F.interpolate(cond_img, size=(h_down, w_down), mode='nearest')

            decoded = self.model.decode(latents)
            recon_residual = decoded[:, :1, :, :]

            # Denormalize for visualization
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
        self.axes[0].imshow(recon_np, cmap='terrain')
        self.axes[0].set_title('Reconstructed (AE from latents)')
        self.axes[0].axis('off')

        real_np = real.cpu().numpy()
        self.axes[1].imshow(real_np, cmap='terrain')
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
@click.option('--autoencoder-path', type=click.Path(exists=True), required=True,
              help='Path to the trained autoencoder checkpoint directory')
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to a diffusion config with *_dataset section (e.g., diffusion_decoder_64-3.cfg)')
@click.option('--batch-size', type=int, default=4, help='Batch size (default: 4)')
@click.option('--device', type=str, default='cuda', help='Device (default: cuda)')
@click.option('--num-workers', type=int, default=4, help='Dataloader workers (default: 4)')
@click.option('--headless', is_flag=True, help='Headless mode: save to disk')
@click.option('--num-samples', type=int, default=10, help='Samples to save in headless mode')
@click.option('--output-dir', type=str, default='decoder_viz_output', help='Output dir for headless mode')
def main(autoencoder_path, config, batch_size, device, num_workers, headless, num_samples, output_dir):
    if headless:
        matplotlib.use('Agg')

    build_registry()

    print(f"Loading autoencoder from {autoencoder_path}...")
    model = EDMAutoencoder.from_pretrained(autoencoder_path)
    model = model.to(device)
    model.eval()
    print(f"Autoencoder loaded with {model.count_parameters()} parameters")

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

    # Keep only dataset config
    for key in list(config_obj.keys()):
        if not key.endswith('_dataset'):
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

    visualizer = DecoderDatasetVisualizer(model, val_dataset, dataloader, device, headless=headless)

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


