import numpy as np
import os
import torch
from tqdm import tqdm
import warnings
from ema_pytorch import PostHocEMA
from torchmetrics.image.kid import KernelInceptionDistance

from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.data.laplacian_encoder import laplacian_denoise, laplacian_decode
from terrain_diffusion.training.utils import recursive_to, temporary_ema_to_model

from torch.utils.data import DataLoader
from terrain_diffusion.training.datasets import LongDataset


class DiffusionTrainer(Trainer):
    """Trainer for diffusion models."""
    
    def __init__(self, config, resolved, accelerator, state):
        self.config = config
        self.resolved = resolved
        self.accelerator = accelerator
        self.state = state
        
        # Load model and datasets
        self.model = resolved['model']
        assert isinstance(self.model, EDMUnet2D), "Currently only EDMUnet2D is supported."
        
        self.scheduler = resolved['scheduler']
        self.train_dataset = resolved['train_dataset']
        self.val_dataset = LongDataset(resolved['val_dataset'], shuffle=True)
        self.optimizer = self._get_optimizer(self.model, config)
        
        # Initialize EMA
        resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
        self.ema = PostHocEMA(self.model, **resolved['ema'])
        
        autoencoder_path = self.config['evaluation'].get('kid_autoencoder_path')
        if autoencoder_path is not None:
            self.autoencoder = EDMUnet2D.from_pretrained(autoencoder_path)
            self.autoencoder.eval()
            self.autoencoder.requires_grad_(False)
            self.autoencoder = torch.compile(self.autoencoder)
        else:
            self.autoencoder = None
        
        print(f"Training model with {self.model.count_parameters()} parameters.")
    
    def _get_optimizer(self, model, config):
        """Get optimizer based on config settings."""
        if config['optimizer']['type'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), **config['optimizer']['kwargs'])
        else:
            raise ValueError(f"Invalid optimizer type: {config['optimizer']['type']}")
        return optimizer
    
    def get_accelerate_modules(self):
        """Returns modules that need to be passed to accelerator.prepare."""
        return (self.model, self.optimizer)
    
    def set_prepared_modules(self, prepared_modules):
        """Set the modules returned from accelerator.prepare back into the trainer."""
        self.model, self.optimizer = prepared_modules
        # Move EMA to device after models are prepared
        self.ema = self.ema.to(self.accelerator.device)
    
    def get_checkpoint_modules(self):
        """Returns a list of modules that need to be registered for checkpointing."""
        return [self.ema]
    
    def load_model_checkpoint(self, model_ckpt_path):
        """Load model weights from a checkpoint file."""
        temp_model_statedict = type(self.model).from_pretrained(model_ckpt_path).state_dict()
        filtered_state_dict = {}
        for name, param in temp_model_statedict.items():
            if name in self.model.state_dict():
                if param.shape == self.model.state_dict()[name].shape:
                    filtered_state_dict[name] = param
                else:
                    print(f"Skipping parameter {name} due to shape mismatch. "
                          f"Loaded shape: {param.shape}, "
                          f"Model shape: {self.model.state_dict()[name].shape}")
            else:
                print(f"Skipping parameter {name} as it doesn't exist in the current model.")
        try:
            self.model.load_state_dict(filtered_state_dict)
        except Exception as e:
            print("Loading model with strict=False")
            self.model.load_state_dict(filtered_state_dict, strict=False)
        
        # Reset logvar for diffusion models
        if hasattr(self.model, 'logvar_linear'):
            with torch.no_grad():
                self.model.logvar_linear.weight.copy_(torch.randn_like(self.model.logvar_linear.weight))
    
    def get_model_for_saving(self):
        """Returns the main model to save config for."""
        return self.model
    
    def _calc_loss(self, pred_v_t, v_t, logvar, sigma_data):
        """Calculate loss."""
        loss = 1 / (logvar.exp() * sigma_data ** 2) * ((pred_v_t - v_t) ** 2) + logvar
        return loss.mean()
    
    def train_step(self, state, batch):
        """Perform one training step."""
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                images = batch['image']
                cond_img = batch.get('cond_img')
                conditional_inputs = batch.get('cond_inputs')
                
                sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
                sigma = (sigma * self.config['training']['P_std'] + self.config['training']['P_mean']).exp()
                
                sigma_data = self.scheduler.config.sigma_data
                if self.config['training'].get('scale_sigma', False):
                    calc_channels = self.config['training']['scaling_channels']
                    sigma = sigma * torch.maximum(
                        torch.std(images[:, calc_channels], dim=[1, 2, 3], keepdim=True) / sigma_data, 
                        torch.tensor(self.config['training'].get('sigma_scale_eps', 0.05), device=images.device)
                    )
            
                t = torch.atan(sigma / sigma_data)
                cnoise = t.flatten()
            
                noise = torch.randn_like(images) * sigma_data
                x_t = torch.cos(t) * images + torch.sin(t) * noise

                x = x_t / sigma_data
                if cond_img is not None:
                    x = torch.cat([x, cond_img], dim=1)
                
                self.model.train()
                model_output, logvar = self.model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs, return_logvar=True)
                
                pred_v_t = -sigma_data * model_output
                v_t = torch.cos(t) * noise - torch.sin(t) * images

                loss = self._calc_loss(pred_v_t, v_t, logvar, sigma_data)

            # Update state
            state['seen'] += images.shape[0]
            state['step'] += 1
            
            lr = self.resolved['lr_sched'].get(state['seen'])
            for g in self.optimizer.param_groups:
                g['lr'] = lr
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config['training'].get('gradient_clip_val', 10.0)
                ).item()
            else:
                grad_norm = 0.0
            self.optimizer.step()

        if self.accelerator.is_main_process:
            self.ema.update()
        
        return {
            'loss': loss.item(),
            'lr': lr,
            'grad_norm': grad_norm
        }
    
    def _normalize_and_process_terrain(self, terrain):
        """Normalize terrain to [0, 255] uint8 format for KID calculation."""
        terrain_min = torch.amin(terrain, dim=(1, 2, 3), keepdim=True)
        terrain_max = torch.amax(terrain, dim=(1, 2, 3), keepdim=True)
        terrain_range = torch.maximum(terrain_max - terrain_min, torch.tensor(255.0, device=terrain.device))
        terrain_mid = (terrain_min + terrain_max) / 2
        
        terrain_norm = torch.clamp(((terrain - terrain_mid) / terrain_range + 0.5) * 255, 0, 255)
        terrain_norm = terrain_norm.repeat(1, 3, 1, 1)
        return terrain_norm.to(torch.uint8)
    
    def _decode_latents_to_terrain(self, latents, lowfreq_input, autoencoder, scheduler):
        """Decode latents to terrain using consistency decoder and laplacian decoding."""
        device = latents.device
    
        latents_std = self.val_dataset.base_dataset.latents_std.to(latents.device)
        latents_mean = self.val_dataset.base_dataset.latents_mean.to(latents.device)
        sigma_data = scheduler.config.sigma_data
        
        latents = (latents / latents_std + latents_mean)
        
        # Build decoder conditioning by upsampling latent channels to target resolution
        H, W = lowfreq_input.shape[-2]*8, lowfreq_input.shape[-1]*8
        cond_img = torch.nn.functional.interpolate(latents, size=(H, W), mode='nearest')

        # Two-step consistency update to reconstruct residual (+ optional water)
        samples = torch.zeros(latents.shape[0], 1, H, W, device=device, dtype=latents.dtype)
        t0 = torch.atan(scheduler.sigmas[0].to(device) / sigma_data)
        for t_scalar in (t0,):
            t = t_scalar.view(1, 1, 1, 1).expand(samples.shape[0], 1, 1, 1)
            z = torch.randn_like(samples) * sigma_data
            x_t = torch.cos(t) * samples + torch.sin(t) * z
            model_input = torch.cat([x_t / sigma_data, cond_img], dim=1)
            with torch.no_grad():
                pred = -autoencoder(model_input, noise_labels=t.flatten(), conditional_inputs=[])
            samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred

        decoded = samples / sigma_data
        residual = decoded[:, :1]

        # Denormalize and compose terrain
        highfreq = self.val_dataset.base_dataset.denormalize_residual(residual)
        lowfreq = self.val_dataset.base_dataset.denormalize_lowfreq(lowfreq_input)
        highfreq, lowfreq = laplacian_denoise(highfreq, lowfreq, sigma=5)
        return laplacian_decode(highfreq, lowfreq)
    
    def _calculate_base_kid(self, data_iter, generator):
        """Calculate Kernel Inception Distance (KID) for generated samples."""
        n_images = self.config['evaluation']['kid_n_images']
        pbar = tqdm(total=n_images, desc="Calculating KID")
        
        scheduler = self.scheduler
        
        autoencoder = self.autoencoder.to(self.accelerator.device)
        
        with torch.no_grad(), self.accelerator.autocast():
            kid = KernelInceptionDistance(normalize=True).to(self.accelerator.device)
            
            samples_generated = 0
            
            while samples_generated < n_images:
                batch = recursive_to(next(data_iter), device=self.accelerator.device)
                images = batch['image']
                cond_img = batch.get('cond_img')
                conditional_inputs = batch.get('cond_inputs')
                
                # Generate samples using diffusion sampling
                samples = torch.randn(images.shape, generator=generator, device=images.device) * scheduler.sigmas[0]
                
                # Sampling loop
                scheduler.set_timesteps(self.config['evaluation']['kid_scheduler_steps'])
                for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                    t, sigma = t.to(samples.device), sigma.to(samples.device)
                    
                    scaled_input = scheduler.precondition_inputs(samples, sigma)
                    cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))
                    
                    # Get model predictions
                    if cond_img is not None:
                        x = torch.cat([scaled_input, cond_img], dim=1)
                    else:
                        x = scaled_input
                    self.model.eval()
                    model_output = self.model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs)
                    
                    samples = scheduler.step(model_output, t, samples).prev_sample
                    
                samples = samples / scheduler.config.sigma_data
                
                # Decode latents to terrain
                terrain = self._decode_latents_to_terrain(samples[:, :4], samples[:, 4:5], autoencoder, scheduler)
                terrain = torch.sign(terrain) * torch.square(terrain)
                
                real_terrain = batch['ground_truth']
                real_terrain = torch.sign(real_terrain) * torch.square(real_terrain)
                
                # Update KID metric for original terrain
                kid.update(self._normalize_and_process_terrain(terrain), real=False)
                kid.update(self._normalize_and_process_terrain(real_terrain), real=True)
                
                samples_generated += images.shape[0]
                pbar.update(images.shape[0])
            
            pbar.close()
            
            autoencoder = autoencoder.to('cpu')
            
            # Calculate final KID scores
            kid_mean, kid_std = kid.compute()
            print(f"Final KID Score (original): {kid_mean.item():.6f} ± {kid_std.item():.6f}")
            return {
                'val/kid_mean': kid_mean.item(), 
                'val/kid_std': kid_std.item()
            }
    
    def _calculate_decoder_kid(self, data_iter, generator):
        """Calculate Kernel Inception Distance (KID) for decoder models."""
        n_images = self.config['evaluation']['kid_n_images']
        pbar = tqdm(total=n_images, desc="Calculating Decoder KID")
        
        scheduler = self.scheduler
        
        with torch.no_grad(), self.accelerator.autocast():
            kid = KernelInceptionDistance(normalize=True).to(self.accelerator.device)
            
            samples_generated = 0
            
            while samples_generated < n_images:
                batch = recursive_to(next(data_iter), device=self.accelerator.device)
                images = batch['image']
                cond_img = batch.get('cond_img')
                lowfreq = batch['lowfreq']
                conditional_inputs = batch.get('cond_inputs')
                
                # Generate samples using diffusion sampling
                samples = torch.randn(images.shape, generator=generator, device=images.device) * scheduler.sigmas[0]
                
                # Sampling loop
                scheduler.set_timesteps(self.config['evaluation']['kid_scheduler_steps'])
                for t, sigma in zip(scheduler.timesteps, scheduler.sigmas):
                    t, sigma = t.to(samples.device), sigma.to(samples.device)
                    
                    scaled_input = scheduler.precondition_inputs(samples, sigma)
                    cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1).expand(samples.shape[0]))
                    
                    # Get model predictions
                    x = torch.cat([scaled_input, cond_img], dim=1)
                    self.model.eval()
                    model_output = self.model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs)
                    
                    samples = scheduler.step(model_output, t, samples).prev_sample
                
                # Only evaluate first channel
                samples = samples[:, :1] / scheduler.config.sigma_data
                real_samples = images[:, :1] / scheduler.config.sigma_data
            
                residual_std = self.val_dataset.base_dataset.residual_std.to(images.device)
                residual_mean = self.val_dataset.base_dataset.residual_mean.to(images.device)
                output_full = laplacian_decode(samples * residual_std + residual_mean, lowfreq, extrapolate=True)
                images_full = laplacian_decode(images * residual_std + residual_mean, lowfreq, extrapolate=True)
                
                output_full = torch.sign(output_full) * torch.square(output_full)
                images_full = torch.sign(images_full) * torch.square(images_full)
                
                # Update KID metric for original samples
                kid.update(self._normalize_and_process_terrain(samples), real=False)
                kid.update(self._normalize_and_process_terrain(real_samples), real=True)
                
                samples_generated += images.shape[0]
                pbar.update(images.shape[0])
            
            pbar.close()
            
            # Calculate final KID scores
            kid_mean, kid_std = kid.compute()
            print(f"Final Decoder KID Score (original): {kid_mean.item():.6f} ± {kid_std.item():.6f}")
            return {
                'val/kid_mean': kid_mean.item(), 
                'val/kid_std': kid_std.item()
            }
                
    
    def evaluate(self):
        """Perform evaluation and return metrics."""
        self.val_dataset.set_seed(self.config['training']['seed'] + 638)
        dl_kwargs = self.resolved['dataloader_kwargs']
        dl_kwargs['persistent_workers'] = False
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.config['training']['batch_size'],
            num_workers=self.resolved['dataloader_kwargs']['num_workers'],
        )
        
        validation_stats = {'loss': []}
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config['training']['seed'] + 548)
        pbar = tqdm(total=self.config['evaluation']['validation_steps'], desc="Validation")
        val_dataloader_iter = iter(self.val_dataloader)
        
        with temporary_ema_to_model(self.ema.ema_models[0]):
            while pbar.n < pbar.total:
                batch = recursive_to(next(val_dataloader_iter), device=self.accelerator.device)
                images = batch['image']
                cond_img = batch.get('cond_img')
                conditional_inputs = batch.get('cond_inputs')
                
                sigma = torch.randn(images.shape[0], device=images.device, generator=generator).reshape(-1, 1, 1, 1)
                sigma = (sigma * self.config['evaluation']['P_std'] + self.config['evaluation']['P_mean']).exp()
                
                sigma_data = self.scheduler.config.sigma_data
                
                t = torch.atan(sigma / sigma_data)
                cnoise = t.flatten()
                
                noise = torch.randn(images.shape, generator=generator, device=images.device) * sigma_data
                x_t = torch.cos(t) * images + torch.sin(t) * noise

                x = x_t / sigma_data
                if cond_img is not None:
                    x = torch.cat([x, cond_img], dim=1)
                    
                self.model.eval()
                with torch.no_grad(), self.accelerator.autocast():
                    model_output, logvar = self.model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs, return_logvar=True)
                    pred_v_t = -sigma_data * model_output
                    
                v_t = torch.cos(t) * noise - torch.sin(t) * images

                loss = self._calc_loss(pred_v_t, v_t, logvar, sigma_data)
                validation_stats['loss'].append(loss.item())
                
                pbar.update(images.shape[0])
                pbar.set_postfix(loss=f"{np.mean(validation_stats['loss']):.4f}")
            pbar.close()
            
            output = {'val/loss': np.mean(validation_stats['loss'])}
            if self.config['evaluation'].get('mode') == 'base':
                self.val_dataset.set_seed(self.config['training']['seed'] + 7843)
                self.val_dataloader = DataLoader(
                    self.val_dataset, 
                    batch_size=self.config['evaluation']['kid_batch_size']
                )
                val_dataloader_iter = iter(self.val_dataloader)
                output.update(self._calculate_base_kid(val_dataloader_iter, generator))
            elif self.config['evaluation'].get('mode') == 'decoder':
                self.val_dataset.set_seed(self.config['training']['seed'] + 7843)
                self.val_dataloader = DataLoader(
                    self.val_dataset, 
                    batch_size=self.config['evaluation']['kid_batch_size']
                )
                val_dataloader_iter = iter(self.val_dataloader)
                output.update(self._calculate_decoder_kid(val_dataloader_iter, generator))
            else:
                print('No mode specified, skipping KID calculation')
            
            return output

