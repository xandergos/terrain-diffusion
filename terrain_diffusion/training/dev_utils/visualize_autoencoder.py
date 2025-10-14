#!/usr/bin/env python3
"""
Visualization script for autoencoder reconstructions.

This script loads a trained autoencoder model and visualizes real images alongside their reconstructions.
It provides an interactive matplotlib interface with navigation buttons to scroll through images.

Usage:
    python visualize_autoencoder.py --model-path /path/to/model/checkpoint --config /path/to/config.yaml
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
from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to


class AutoencoderVisualizer:
    def __init__(self, model, dataloader, device='cuda', headless=False):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.headless = headless
        self.data_iter = iter(dataloader)
        self.current_batch = None
        self.current_idx = 0
        self.batch_idx = 0
        
        # Setup matplotlib figure - hardcoded for 2 channels (residual + water)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Autoencoder Reconstruction Comparison (Residual + Water)', fontsize=16)
        
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
        
        # Generate reconstructions
        self.model.eval()
        with torch.no_grad():
            images = self.current_batch['image']
            cond_img = self.current_batch.get('cond_img')
            conditional_inputs = self.current_batch.get('cond_inputs')
            
            # Prepare input for encoding
            scaled_clean_images = images
            if cond_img is not None:
                scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
            
            # Encode and decode
            z_means, z_logvars = self.model.preencode(scaled_clean_images, conditional_inputs)
            z = self.model.postencode(z_means, z_logvars)
            self.reconstructions = self.model.decode(z)
            
        self.current_idx = 0
        self.batch_idx += 1
        
    def normalize_for_display(self, tensor, channel_type='residual'):
        """Normalize tensor for display based on channel type"""
        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        
        if channel_type == 'residual':
            tensor = tensor
        elif channel_type == 'water_real':
            # For water coverage real data: already 0-1, just clamp
            tensor = np.clip(tensor, 0, 1)
        elif channel_type == 'water_logits':
            # For water coverage logits: apply sigmoid to convert to probabilities
            tensor = 1 / (1 + np.exp(-tensor))  # sigmoid
            tensor = np.clip(tensor, 0, 1)
            
        return tensor
        
    def update_display(self):
        """Update the display with current images"""
        if self.current_batch is None:
            return
            
        batch_size = self.current_batch['image'].shape[0]
        if self.current_idx >= batch_size:
            self.current_idx = 0
            
        # Get current images
        real_img = self.current_batch['image'][self.current_idx]
        recon_img = self.reconstructions[self.current_idx]
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            
        # Hardcoded for exactly 2 channels: residual (0) and water (1)
        channel_names = ['Residual', 'Water Coverage']
        channel_types = ['residual', 'water_real']  # For real images
        recon_channel_types = ['residual', 'water_real']  # For reconstructions
        
        for i in range(2):  # Exactly 2 channels
            # Set vmin/vmax for water coverage channels
            vmin, vmax = (0, 1) if i == 1 else (None, None)  # Water channel uses 0-1 range
            
            # Real image
            real_channel = self.normalize_for_display(real_img[i], channel_types[i])
            self.axes[0, i].imshow(real_channel, cmap='viridis', vmin=vmin, vmax=vmax)
            self.axes[0, i].set_title(f'Real - {channel_names[i]}')
            self.axes[0, i].axis('off')
            
            # Reconstructed image
            recon_channel = self.normalize_for_display(recon_img[i], recon_channel_types[i])
            self.axes[1, i].imshow(recon_channel, cmap='viridis', vmin=vmin, vmax=vmax)
            self.axes[1, i].set_title(f'Reconstruction - {channel_names[i]}')
            self.axes[1, i].axis('off')
            
        # Update title with current position
        self.fig.suptitle(f'Autoencoder Reconstruction - Batch {self.batch_idx}, Image {self.current_idx + 1}/{batch_size}', fontsize=16)
        
        # Calculate and display reconstruction error
        with torch.no_grad():
            mse_loss = torch.nn.functional.mse_loss(recon_img, real_img).item()
            mae_loss = torch.nn.functional.l1_loss(recon_img, real_img).item()
            
        # Add error text
        error_text = f'MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}'
        self.fig.text(0.5, 0.95, error_text, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.9)
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
        """Go to previous batch (reload current batch - dataloader doesn't support reverse)"""
        # Since we can't go backwards in dataloader, just reload current batch
        self.load_next_batch()
        self.update_display()
        
    def next_batch(self, event):
        """Load next batch"""
        self.load_next_batch()
        self.update_display()
        
    def save_current(self, event):
        """Save current comparison to file"""
        if self.current_batch is not None:
            filename = f'autoencoder_comparison_batch{self.batch_idx}_img{self.current_idx + 1}.png'
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f'Saved comparison to {filename}')
    
    def save_samples(self, num_samples, output_dir='autoencoder_viz_output'):
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
@click.option('--model-path', type=click.Path(exists=True), required=True, 
              help='Path to the saved autoencoder model checkpoint directory')
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to the training config file')
@click.option('--batch-size', type=int, default=8, 
              help='Batch size for loading data (default: 8)')
@click.option('--device', type=str, default='cuda',
              help='Device to run on (default: cuda)')
@click.option('--num-workers', type=int, default=4,
              help='Number of data loading workers (default: 4)')
@click.option('--headless', is_flag=True,
              help='Run in headless mode (no display, save to disk)')
@click.option('--num-samples', type=int, default=10,
              help='Number of samples to save in headless mode (default: 10)')
@click.option('--output-dir', type=str, default='autoencoder_viz_output',
              help='Output directory for saved visualizations (default: autoencoder_viz_output)')
def main(model_path, config, batch_size, device, num_workers, headless, num_samples, output_dir):
    """Visualize autoencoder reconstructions interactively."""
    
    # Set matplotlib backend for headless mode
    if headless:
        matplotlib.use('Agg')
    
    build_registry()
    
    # Load model
    print(f"Loading autoencoder from {model_path}...")
    model = EDMAutoencoder.from_pretrained(model_path)
    print("Loaded model using from_pretrained")
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully with {model.count_parameters()} parameters")
    
    # Load config and dataset
    print(f"Loading config from {config}...")
    
    # Load and resolve config
    if config.endswith('.json'):
        import json
        with open(config, 'r') as f:
            config_dict = json.load(f)
        config_obj = Config(config_dict)
    else:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_obj = Config(config_dict)
    
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
    dataloader_kwargs = resolved.get('dataloader_kwargs', {})
    dataloader_kwargs['num_workers'] = num_workers
    
    dataloader = DataLoader(
        LongDataset(val_dataset, shuffle=True),
        batch_size=batch_size,
        **dataloader_kwargs,
        drop_last=True
    )
    
    print(f"Created dataloader with batch size {batch_size}")
    
    # Create visualizer
    visualizer = AutoencoderVisualizer(model, dataloader, device, headless=headless)
    
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
        print("- Save: Save current comparison to PNG file")
        print("- Close the window to exit")
        visualizer.show()


if __name__ == '__main__':
    main()
