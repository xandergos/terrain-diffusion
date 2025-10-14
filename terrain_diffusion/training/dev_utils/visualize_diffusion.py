#!/usr/bin/env python3
"""
Visualization script for diffusion model generation with autoencoder decoding.

This script generates samples using a diffusion model and visualizes:
- Generated latents, lowfreq, and climate channels
- Decoded terrain (residual + water) from latents using an autoencoder
- Climate channels (temperature, precipitation, seasonality)
- Lowfreq and climate mask channels

The generation process follows the same approach as calc_base_fid.py, supporting
both standard diffusion models and consistency models.

Usage:
    python visualize_diffusion.py --autoencoder-path /path/to/autoencoder \
                                  --diffusion-model-path /path/to/diffusion \
                                  --config /path/to/config.cfg
    
    python visualize_diffusion.py --autoencoder-path /path/to/autoencoder \
                                  --diffusion-model-path /path/to/diffusion \
                                  --config /path/to/config.cfg \
                                  --headless --num-samples 10 \
                                  --scheduler-steps 20 --guidance-scale 1.5
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
from terrain_diffusion.training.datasets.datasets import H5LatentsDataset, LongDataset
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import recursive_to
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise

class BaseDiffusionVisualizer:
    def __init__(self, ae_model, diffusion_model, scheduler, dataset, dataloader, device='cuda', headless=False, 
                 is_consistency_model=False, scheduler_steps=15, guidance_scale=1.0, guide_model=None, dtype=torch.float32):
        self.ae_model = ae_model
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.headless = headless
        self.is_consistency_model = is_consistency_model
        self.scheduler_steps = scheduler_steps
        self.guidance_scale = guidance_scale
        self.guide_model = guide_model
        self.dtype = dtype
        self.data_iter = iter(dataloader)
        self.current_batch = None
        self.current_idx = 0
        self.batch_idx = 0
        
        # Setup matplotlib figure
        # Layout: 2 rows x 4 columns
        # Row 1: Merged Terrain (Lowfreq + Residual), Water Coverage, Lowfreq, Climate Mask
        # Row 2: Temperature, Temp Seasonality, Precipitation, Precip Seasonality
        self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        self.fig.suptitle('Diffusion Model Generated Samples', fontsize=16)
        
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
        """Load next batch from dataloader and generate samples using diffusion"""
        try:
            batch = next(self.data_iter)
            batch = recursive_to(batch, self.device)
        except StopIteration:
            # Reset iterator if we reach the end
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
            batch = recursive_to(batch, self.device)
        
        # Generate samples using diffusion model
        self.ae_model.eval()
        self.diffusion_model.eval()
        
        with torch.no_grad():
            # Get conditional inputs
            cond_img = batch.get('cond_img')
            cond_inputs = batch.get('cond_inputs', [])
            
            if not self.is_consistency_model:
                # Generate samples using standard diffusion
                samples = torch.randn_like(batch['image']) * self.scheduler.sigmas[0]
                
                # Sampling loop
                self.scheduler.set_timesteps(self.scheduler_steps)
                for t, sigma in zip(self.scheduler.timesteps, self.scheduler.sigmas):
                    t, sigma = t.to(samples.device), sigma.to(samples.device)
                    
                    scaled_input = self.scheduler.precondition_inputs(samples, sigma)
                    cnoise = self.scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))
                    
                    # Get model predictions
                    x = torch.cat([scaled_input, cond_img], dim=1) if cond_img is not None else scaled_input
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        if not self.guide_model or self.guidance_scale == 1.0:
                            model_output = self.diffusion_model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                        else:
                            model_output_m = self.diffusion_model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                            model_output_g = self.guide_model(x, noise_labels=cnoise, conditional_inputs=cond_inputs)
                            model_output = model_output_g + self.guidance_scale * (model_output_m - model_output_g)
                    
                    samples = self.scheduler.step(model_output, t, samples).prev_sample
            else:
                # Generate samples using consistency model
                samples = torch.zeros_like(batch['image'])
                
                for t in [np.arctan(self.scheduler.sigmas[0] / self.scheduler.config.sigma_data), 
                          np.arctan(0.78 / self.scheduler.config.sigma_data)]:
                    t = torch.tensor([t], device=batch['image'].device).view(1, 1, 1, 1).expand(batch['image'].shape[0], 1, 1, 1)
                    z = torch.randn_like(batch['image']) * self.scheduler.config.sigma_data
                    x_t = torch.cos(t) * samples + torch.sin(t) * z
                    
                    if cond_img is not None:
                        model_input = torch.cat([x_t / self.scheduler.config.sigma_data, cond_img], dim=1)
                    else:
                        model_input = x_t / self.scheduler.config.sigma_data
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        pred = -self.diffusion_model(model_input, noise_labels=t.flatten(), conditional_inputs=cond_inputs)
                    
                    samples = torch.cos(t) * x_t - torch.sin(t) * self.scheduler.config.sigma_data * pred
            
            # Process generated samples
            # samples = [latents (4ch), lowfreq (1ch), climate (4ch), climate_mask (1ch)]
            base_dataset = self.dataset.base_dataset if hasattr(self.dataset, 'base_dataset') else self.dataset
            latents_std = base_dataset.latents_std.to(samples.device)
            latents_mean = base_dataset.latents_mean.to(samples.device)
            sigma_data = self.scheduler.config.sigma_data
            
            # Extract and denormalize latents
            latent = (samples[:, :4] / latents_std + latents_mean) / sigma_data
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                decoded = self.ae_model.decode(latent)
            highfreq = decoded[:, :1]
            watercover = decoded[:, 1:2]
            
            # Denormalize residual and watercover
            highfreq = base_dataset.denormalize_residual(highfreq)
            watercover = base_dataset.denormalize_watercover(watercover)
            
            # Denormalize lowfreq
            lowfreq = base_dataset.denormalize_lowfreq(samples[:, 4:5])
            
            # Apply laplacian denoise and decode to merge
            highfreq, lowfreq = laplacian_denoise(highfreq, lowfreq, sigma=5)
            merged_terrain = laplacian_decode(highfreq, lowfreq)
            
            # Store results
            self.merged_terrain = merged_terrain
            self.watercover = watercover
            
            # Store generated batch (including climate channels for visualization)
            self.current_batch = {
                'image': samples,  # Full generated sample with all channels
                'path': [f'generated_sample_{i}' for i in range(samples.shape[0])]
            }
            
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
        image = self.current_batch['image'][self.current_idx]
        merged_terrain = self.merged_terrain[self.current_idx, 0]
        watercover = self.watercover[self.current_idx, 0]
        path = self.current_batch['path'][self.current_idx]
        
        # Extract components
        # image = [latents (4ch), lowfreq (1ch), climate (4ch), climate_mask (1ch)]
        latents = image[:4]
        lowfreq = image[4]
        climate = image[5:9]
        climate_mask = image[9]
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            
        # Row 1: Merged Terrain, Decoded Water, Lowfreq, Climate Mask
        terrain_np = merged_terrain.cpu().numpy() if isinstance(merged_terrain, torch.Tensor) else merged_terrain
        self.axes[0, 0].imshow(terrain_np, cmap='terrain')
        self.axes[0, 0].set_title('Merged Terrain')
        self.axes[0, 0].axis('off')
        
        water_np = watercover.cpu().numpy() if isinstance(watercover, torch.Tensor) else watercover
        self.axes[0, 1].imshow(water_np, cmap='Blues', vmin=0, vmax=1)
        self.axes[0, 1].set_title('Water Coverage')
        self.axes[0, 1].axis('off')
        
        lowfreq_np = lowfreq.cpu().numpy() if isinstance(lowfreq, torch.Tensor) else lowfreq
        self.axes[0, 2].imshow(lowfreq_np, cmap='terrain')
        self.axes[0, 2].set_title('Lowfreq')
        self.axes[0, 2].axis('off')
        
        mask_np = climate_mask.cpu().numpy() if isinstance(climate_mask, torch.Tensor) else climate_mask
        self.axes[0, 3].imshow(mask_np, cmap='gray')
        self.axes[0, 3].set_title('Climate Mask')
        self.axes[0, 3].axis('off')
        
        # Row 2: All climate channels
        climate_names = ['Temperature', 'Temp Seasonality', 'Precipitation', 'Precip Seasonality']
        for i in range(4):
            climate_channel = climate[i]
            climate_np = climate_channel.cpu().numpy() if isinstance(climate_channel, torch.Tensor) else climate_channel
            self.axes[1, i].imshow(climate_np, cmap='RdYlBu_r')
            self.axes[1, i].set_title(climate_names[i])
            self.axes[1, i].axis('off')
        
        # Update title with current position and path
        self.fig.suptitle(
            f'Diffusion Generated Samples - Batch {self.batch_idx}, Image {self.current_idx + 1}/{batch_size}\n'
            f'ID: {path}',
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
@click.option('--diffusion-model-path', type=click.Path(exists=True), required=True,
              help='Path to the trained diffusion model checkpoint directory')
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
@click.option('--output-dir', type=str, default='diffusion_output',
              help='Output directory for saved visualizations (default: diffusion_output)')
@click.option('--is-consistency-model', is_flag=True, default=False,
              help='Use consistency model sampling (default: False)')
@click.option('--scheduler-steps', type=int, default=15,
              help='Number of scheduler steps for diffusion sampling (default: 15)')
@click.option('--guidance-scale', type=float, default=1.0,
              help='Guidance scale for classifier-free guidance (default: 1.0)')
@click.option('--guide-model-path', type=click.Path(exists=True), default=None,
              help='Path to guidance model checkpoint (default: None)')
@click.option('--dtype', type=str, default='float32',
              help='Data type for model inference: float16, bfloat16, or float32 (default: float32)')
def main(autoencoder_path, diffusion_model_path, config, batch_size, device, num_workers, headless, num_samples, output_dir,
         is_consistency_model, scheduler_steps, guidance_scale, guide_model_path, dtype):
    """Generate and visualize diffusion model samples with latents, lowfreq, and climate channels."""
    
    # Set matplotlib backend for headless mode
    if headless:
        matplotlib.use('Agg')
    
    build_registry()
    
    # Load autoencoder
    print(f"Loading autoencoder from {autoencoder_path}...")
    ae_model = EDMAutoencoder.from_pretrained(autoencoder_path)
    ae_model = ae_model.to(device)
    ae_model.eval()
    print(f"Autoencoder loaded successfully with {ae_model.count_parameters()} parameters")
    
    # Load diffusion model
    print(f"Loading diffusion model from {diffusion_model_path}...")
    diffusion_model = EDMUnet2D.from_pretrained(diffusion_model_path)
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    print(f"Diffusion model loaded successfully with {diffusion_model.count_parameters()} parameters")
    
    # Load guidance model if specified
    guide_model = None
    if guide_model_path:
        print(f"Loading guidance model from {guide_model_path}...")
        guide_model = EDMUnet2D.from_pretrained(guide_model_path)
        guide_model = guide_model.to(device)
        guide_model.eval()
        print(f"Guidance model loaded successfully with {guide_model.count_parameters()} parameters")

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
        if not (key.endswith('_dataset') or key == 'scheduler'):
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

    scheduler = resolved['scheduler']
    
    print(f"Created dataloader with batch size {batch_size}")
    
    # Convert dtype string to torch dtype
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    torch_dtype = dtype_map[dtype]
    
    # Create visualizer
    visualizer = BaseDiffusionVisualizer(
        ae_model, diffusion_model, scheduler, val_dataset, dataloader, device, 
        headless=headless, is_consistency_model=is_consistency_model, 
        scheduler_steps=scheduler_steps, guidance_scale=guidance_scale,
        guide_model=guide_model, dtype=torch_dtype
    )
    
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

