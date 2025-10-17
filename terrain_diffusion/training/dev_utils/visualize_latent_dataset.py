#!/usr/bin/env python3
"""
Visualization script for H5LatentsDataset with autoencoder decoding.

This script loads the H5LatentsDataset and visualizes:
- Merged terrain (decoded residual combined with lowfreq)
- Lowfreq channel

Usage:
    python visualize_latent_dataset.py --autoencoder-path /path/to/autoencoder --config /path/to/config.cfg
    python visualize_latent_dataset.py --autoencoder-path /path/to/autoencoder --config /path/to/config.cfg --headless --num-samples 10
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
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise

class LatentDatasetVisualizer:
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
        
        # Setup matplotlib figure
        # Layout: 1 row x 2 columns
        # Col 1: Merged Terrain (Lowfreq + Residual)
        # Col 2: Lowfreq
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('H5LatentsDataset Visualization', fontsize=16)
        
        # Add navigation buttons only in interactive mode
        if not headless:
            self.setup_buttons()
        
        # Load first batch and display
        self.load_next_batch()
        self.update_display()
        
    def setup_buttons(self):
        """Setup navigation buttons"""
        # Navigation buttons
        ax_prev = plt.axes([0.25, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.36, 0.02, 0.1, 0.04])
        ax_batch_prev = plt.axes([0.48, 0.02, 0.12, 0.04])
        ax_batch_next = plt.axes([0.61, 0.02, 0.12, 0.04])
        ax_save = plt.axes([0.75, 0.02, 0.1, 0.04])
        
        # Create buttons
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_batch_prev = Button(ax_batch_prev, 'Prev Batch')
        self.btn_batch_next = Button(ax_batch_next, 'Next Batch')
        self.btn_save = Button(ax_save, 'Save')
        
        # Connect button callbacks
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_batch_prev.on_clicked(self.prev_batch)
        self.btn_batch_next.on_clicked(self.next_batch)
        self.btn_save.on_clicked(self.save_current)
        
    def load_next_batch(self):
        """Load next batch from dataloader"""
        try:
            self.current_batch = next(self.data_iter)
            self.current_batch = recursive_to(self.current_batch, self.device)
        except StopIteration:
            # Reset iterator if we reach the end
            self.data_iter = iter(self.dataloader)
            self.current_batch = next(self.data_iter)
            self.current_batch = recursive_to(self.current_batch, self.device)
        
        # Decode latents with autoencoder and merge with lowfreq
        self.model.eval()
        with torch.no_grad():
            # Extract components from the concatenated image tensor
            # image = [latents (4ch), lowfreq (1ch)]
            image = self.current_batch['image']

            # Extract latents (first 4 channels) and lowfreq (channel 4)
            latents = image[:, :4, :, :] * 2
            lowfreq = image[:, 4:5, :, :] * 2

            # Decode latents to get residual terrain
            decoded = self.model.decode(latents)
            residual = decoded[:, :1, :, :]

            # Denormalize residual and lowfreq
            residual = self.dataset.denormalize_residual(residual)
            lowfreq = self.dataset.denormalize_lowfreq(lowfreq)

            # Apply laplacian denoise and decode to merge
            residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
            merged_terrain = laplacian_decode(residual, lowfreq)

            # Store results
            self.merged_terrain = merged_terrain
            self.lowfreq = lowfreq
            
        self.current_idx = 0
        self.batch_idx += 1
        
    def normalize_for_display(self, tensor, channel_type='residual'):
        """Normalize tensor for display based on channel type"""
        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        
        if channel_type == 'residual':
            # For residual channel: normalize to 0-1 range (unbounded values)
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max > tensor_min:
                tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
            else:
                tensor = np.zeros_like(tensor)
        elif channel_type == 'water_logits':
            raise ValueError("water_logits channel type not supported for display")
        elif channel_type == 'climate':
            # For climate channels: normalize to 0-1 for display
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max > tensor_min:
                tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
            else:
                tensor = np.zeros_like(tensor)
        elif channel_type == 'lowfreq':
            # For lowfreq: normalize to 0-1 for display
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max > tensor_min:
                tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
            else:
                tensor = np.zeros_like(tensor)
        elif channel_type == 'mask':
            # For mask: already 0-1 or binary
            tensor = np.clip(tensor, 0, 1)
            
        return tensor
        
    def update_display(self):
        """Update the display with current images"""
        if self.current_batch is None:
            return
            
        batch_size = self.current_batch['image'].shape[0]
        if self.current_idx >= batch_size:
            self.current_idx = 0
            
        # Get current data
        merged_terrain = self.merged_terrain[self.current_idx, 0]
        lowfreq = self.lowfreq[self.current_idx, 0]
        path = self.current_batch['path'][self.current_idx]
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            
        # Cols: Merged Terrain, Lowfreq
        terrain_np = merged_terrain.cpu().numpy() if isinstance(merged_terrain, torch.Tensor) else merged_terrain
        self.axes[0].imshow(terrain_np, cmap='terrain')
        self.axes[0].set_title('Merged Terrain')
        self.axes[0].axis('off')

        lowfreq_np = lowfreq.cpu().numpy() if isinstance(lowfreq, torch.Tensor) else lowfreq
        self.axes[1].imshow(lowfreq_np, cmap='terrain')
        self.axes[1].set_title('Lowfreq')
        self.axes[1].axis('off')
        
        # Update title with current position and path
        self.fig.suptitle(
            f'H5LatentsDataset - Batch {self.batch_idx}, Image {self.current_idx + 1}/{batch_size}\n'
            f'Path: {path}',
            fontsize=14
        )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.92)
        self.fig.canvas.draw()
        
    def prev_image(self, event):
        """Go to previous image in current batch"""
        if self.current_batch is not None:
            batch_size = self.current_batch['image'].shape[0]
            self.current_idx = (self.current_idx - 1) % batch_size
            self.update_display()
            
    def next_image(self, event):
        """Go to next image in current batch"""
        if self.current_batch is not None:
            batch_size = self.current_batch['image'].shape[0]
            self.current_idx = (self.current_idx + 1) % batch_size
            self.update_display()
            
    def prev_batch(self, event):
        """Go to previous batch (reload - dataloader doesn't support reverse)"""
        self.load_next_batch()
        self.update_display()
        
    def next_batch(self, event):
        """Load next batch"""
        self.load_next_batch()
        self.update_display()
        
    def save_current(self, event):
        """Save current comparison to file"""
        if self.current_batch is not None:
            filename = f'latent_dataset_viz_batch{self.batch_idx}_img{self.current_idx + 1}.png'
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f'Saved visualization to {filename}')
            
    def save_samples(self, num_samples, output_dir='latent_viz_output'):
        """Save multiple samples to disk (for headless mode)"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        batch_count = 0
        
        while saved_count < num_samples:
            batch_size = self.current_batch['image'].shape[0]
            
            # Save all images in current batch
            for idx in range(batch_size):
                if saved_count >= num_samples:
                    break
                    
                self.current_idx = idx
                self.update_display()
                
                filename = os.path.join(output_dir, f'sample_{saved_count:04d}.png')
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                print(f'Saved {filename}')
                saved_count += 1
            
            # Load next batch if we need more samples
            if saved_count < num_samples:
                self.load_next_batch()
                batch_count += 1
        
        print(f'\nSaved {saved_count} samples to {output_dir}/')
    
    def show(self):
        """Show the interactive plot"""
        plt.show()


@click.command()
@click.option('--autoencoder-path', type=click.Path(exists=True), required=True,
              help='Path to the trained autoencoder model checkpoint directory')
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to the diffusion config file (e.g., diffusion_192-3.cfg)')
@click.option('--batch-size', type=int, default=4,
              help='Batch size for loading data (default: 4)')
@click.option('--device', type=str, default='cuda',
              help='Device to run on (default: cuda)')
@click.option('--num-workers', type=int, default=4,
              help='Number of data loading workers (default: 4)')
@click.option('--headless', is_flag=True,
              help='Run in headless mode (no display, save to disk)')
@click.option('--num-samples', type=int, default=10,
              help='Number of samples to save in headless mode (default: 10)')
@click.option('--output-dir', type=str, default='latent_viz_output',
              help='Output directory for saved visualizations (default: latent_viz_output)')
def main(autoencoder_path, config, batch_size, device, num_workers, headless, num_samples, output_dir):
    """Visualize H5LatentsDataset with autoencoder decoding."""
    
    # Set matplotlib backend for headless mode
    if headless:
        matplotlib.use('Agg')
    
    build_registry()
    
    # Load autoencoder
    print(f"Loading autoencoder from {autoencoder_path}...")
    model = EDMAutoencoder.from_pretrained(autoencoder_path)
    model = model.to(device)
    model.eval()
    print(f"Autoencoder loaded successfully with {model.count_parameters()} parameters")
    
    # Load config and dataset
    print(f"Loading config from {config}...")
    
    # Load and resolve config
    if config.endswith('.json'):
        import json
        with open(config, 'r') as f:
            config_dict = json.load(f)
        config_obj = Config(config_dict)
    elif config.endswith('.cfg'):
        # Parse .cfg format using confection
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
    
    # Create dataset and dataloader
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
    
    # Create dataloader
    dataloader = DataLoader(
        LongDataset(val_dataset, shuffle=True),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Created dataloader with batch size {batch_size}")
    
    # Create visualizer
    visualizer = LatentDatasetVisualizer(model, val_dataset, dataloader, device, headless=headless)
    
    if headless:
        # Headless mode: save samples to disk
        print(f"Running in headless mode...")
        print(f"Saving {num_samples} samples to {output_dir}/")
        visualizer.save_samples(num_samples, output_dir)
    else:
        # Interactive mode: show GUI
        print("Starting interactive visualization...")
        print("Use the buttons to navigate:")
        print("- Previous/Next: Navigate within current batch")
        print("- Prev/Next Batch: Load new batches")
        print("- Save: Save current visualization to PNG file")
        print("- Close the window to exit")
        visualizer.show()


if __name__ == '__main__':
    main()

