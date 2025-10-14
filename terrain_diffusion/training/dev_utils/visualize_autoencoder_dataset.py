#!/usr/bin/env python3
"""
Visualization script for H5AutoencoderDataset.

This script loads the H5AutoencoderDataset and visualizes:
- Normalized residual terrain data
- Water coverage data

Usage:
    python visualize_autoencoder_dataset.py @autoencoder_x8_ganft.yaml
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
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to


class AutoencoderDatasetVisualizer:
    def __init__(self, dataset, dataloader, device='cpu', headless=False):
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.headless = headless
        self.data_iter = iter(dataloader)
        self.current_batch = None
        self.current_idx = 0
        self.batch_idx = 0
        self.colorbars = [None, None]  # Track colorbars to remove them
        
        # Setup matplotlib figure - 2 columns for residual + water
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('H5AutoencoderDataset Visualization', fontsize=16)
        
        # Add navigation buttons only in interactive mode
        if not headless:
            self.setup_buttons()
        
        # Load first batch and display
        self.load_next_batch()
        self.update_display()
        
    def setup_buttons(self):
        """Setup navigation buttons"""
        # Create button axes
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.21, 0.02, 0.1, 0.04])
        ax_batch_prev = plt.axes([0.4, 0.02, 0.15, 0.04])
        ax_batch_next = plt.axes([0.56, 0.02, 0.15, 0.04])
        ax_save = plt.axes([0.8, 0.02, 0.1, 0.04])
        
        # Create buttons
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_batch_prev = Button(ax_batch_prev, 'Previous Batch')
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
        
        self.current_idx = 0
        self.batch_idx += 1
        
    def to_numpy(self, tensor):
        """Convert tensor to numpy for display"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
        
    def update_display(self):
        """Update the display with current images"""
        if self.current_batch is None:
            return
            
        batch_size = self.current_batch['image'].shape[0]
        if self.current_idx >= batch_size:
            self.current_idx = 0
            
        # Get current image
        img = self.current_batch['image'][self.current_idx]
        
        # Remove old colorbars
        for i, cbar in enumerate(self.colorbars):
            if cbar is not None:
                cbar.remove()
                self.colorbars[i] = None
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            
        # Channel 0: Residual, Channel 1: Water Coverage
        channel_names = ['Residual', 'Water Coverage']
        colormaps = ['terrain', 'Blues']
        
        for i in range(2):
            # Convert to numpy
            channel_data = self.to_numpy(img[i])
            
            # Display - let matplotlib auto-scale residual, constrain water to 0-1
            if i == 0:  # Residual
                im = self.axes[i].imshow(channel_data, cmap=colormaps[i])
            else:  # Water
                im = self.axes[i].imshow(channel_data, cmap=colormaps[i], vmin=0, vmax=1)
            
            self.axes[i].set_title(channel_names[i])
            self.axes[i].axis('off')
            
            # Add colorbar and track it
            self.colorbars[i] = plt.colorbar(im, ax=self.axes[i], fraction=0.046, pad=0.04)
            
        # Get statistics (raw values before normalization)
        residual_tensor = img[0]
        water_tensor = img[1]
        
        if isinstance(residual_tensor, torch.Tensor):
            residual_min = residual_tensor.min().item()
            residual_max = residual_tensor.max().item()
            residual_mean = residual_tensor.mean().item()
            water_min = water_tensor.min().item()
            water_max = water_tensor.max().item()
            water_mean = water_tensor.mean().item()
        else:
            residual_min = residual_tensor.min()
            residual_max = residual_tensor.max()
            residual_mean = residual_tensor.mean()
            water_min = water_tensor.min()
            water_max = water_tensor.max()
            water_mean = water_tensor.mean()
        
        # Update title with current position and statistics
        stats_text = (f'Residual (raw): min={residual_min:.3f}, max={residual_max:.3f}, mean={residual_mean:.3f} | '
                     f'Water (raw): min={water_min:.3f}, max={water_max:.3f}, mean={water_mean:.3f}')
        
        self.fig.suptitle(
            f'H5AutoencoderDataset - Batch {self.batch_idx}, Image {self.current_idx + 1}/{batch_size}\n{stats_text}',
            fontsize=11
        )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.88)
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
        """Save current visualization to file"""
        if self.current_batch is not None:
            filename = f'autoencoder_dataset_viz_batch{self.batch_idx}_img{self.current_idx + 1}.png'
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f'Saved visualization to {filename}')
    
    def save_samples(self, num_samples, output_dir='autoencoder_dataset_viz_output'):
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
@click.argument('config', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=8,
              help='Batch size for loading data (default: 8)')
@click.option('--device', type=str, default='cpu',
              help='Device to run on (default: cpu)')
@click.option('--num-workers', type=int, default=4,
              help='Number of data loading workers (default: 4)')
@click.option('--use-train', is_flag=True,
              help='Use training dataset instead of validation dataset')
@click.option('--headless', is_flag=True,
              help='Run in headless mode (no display, save to disk)')
@click.option('--num-samples', type=int, default=10,
              help='Number of samples to save in headless mode (default: 10)')
@click.option('--output-dir', type=str, default='autoencoder_dataset_viz_output',
              help='Output directory for saved visualizations (default: autoencoder_dataset_viz_output)')
def main(config, batch_size, device, num_workers, use_train, headless, num_samples, output_dir):
    """Visualize H5AutoencoderDataset interactively.
    
    CONFIG can be a path to a YAML config file, or a config name with @ prefix (e.g., @autoencoder_x8_ganft.yaml)
    """
    
    # Set matplotlib backend for headless mode
    if headless:
        matplotlib.use('Agg')
    
    build_registry()
    
    # Handle @ prefix for config files in configs/ directory
    if config.startswith('@'):
        config = f'configs/autoencoder/{config[1:]}'
    
    # Load config and dataset
    print(f"Loading config from {config}...")
    
    # Load and resolve config
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config_obj = Config(config_dict)
    
    # Keep only dataset and dataloader config
    keys_to_keep = []
    for key in config_obj.keys():
        if key.endswith('_dataset') or key == 'dataloader_kwargs':
            keys_to_keep.append(key)
    
    for key in list(config_obj.keys()):
        if key not in keys_to_keep:
            del config_obj[key]
    
    resolved = registry.resolve(config_obj, validate=False)
    
    # Create dataset
    dataset_key = 'train_dataset' if use_train else 'val_dataset'
    try:
        dataset = resolved[dataset_key]
        print(f"Using {dataset_key} with {len(dataset)} samples")
    except KeyError:
        # Try the other dataset
        dataset_key = 'val_dataset' if use_train else 'train_dataset'
        try:
            dataset = resolved[dataset_key]
            print(f"Using {dataset_key} (fallback) with {len(dataset)} samples")
        except KeyError:
            print("No dataset found in config. Please check your config file.")
            return
    
    # Create dataloader
    dataloader_kwargs = resolved.get('dataloader_kwargs', {})
    dataloader_kwargs['num_workers'] = num_workers
    
    dataloader = DataLoader(
        LongDataset(dataset, shuffle=True),
        batch_size=batch_size,
        **dataloader_kwargs,
        drop_last=True
    )
    
    print(f"Created dataloader with batch size {batch_size}")
    
    # Create visualizer
    visualizer = AutoencoderDatasetVisualizer(dataset, dataloader, device, headless=headless)
    
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
        print("- Previous/Next Batch: Load new batches")
        print("- Save: Save current visualization to PNG file")
        print("- Close the window to exit")
        visualizer.show()


if __name__ == '__main__':
    main()

