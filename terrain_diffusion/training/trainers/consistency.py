import os
import torch
from tqdm import tqdm
from ema_pytorch import PostHocEMA
from torchmetrics.image.kid import KernelInceptionDistance

from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder
from terrain_diffusion.data.laplacian_encoder import laplacian_denoise, laplacian_decode
from terrain_diffusion.training.utils import recursive_to, temporary_ema_to_model

from torch.utils.data import DataLoader
from terrain_diffusion.training.datasets import LongDataset


class ConsistencyTrainer(Trainer):
    """Trainer for consistency distillation models."""
    
    def __init__(self, config, resolved, accelerator, state):
        self.config = config
        self.resolved = resolved
        self.accelerator = accelerator
        self.state = state
        
        # Load pretrained models
        self.model_m_pretrained = EDMUnet2D.from_pretrained(resolved['model']['main_path'])
        self.model_g_pretrained = None
        if 'guide_path' in resolved['model'] and resolved['model']['guide_path']:
            self.model_g_pretrained = EDMUnet2D.from_pretrained(resolved['model']['guide_path'])
        
        self.model = EDMUnet2D.from_pretrained(resolved['model']['main_path'])
        
        # Reset logvar weights
        self.model.logvar_linear.weight.data.copy_(torch.randn_like(self.model.logvar_linear.weight.data))
        self.model_m_pretrained.eval()
        if self.model_g_pretrained is not None:
            self.model_g_pretrained.eval()
        self.model.eval()
        
        self.train_dataset = resolved['train_dataset']
        # Scheduler and validation dataset (match diffusion trainer wiring)
        self.scheduler = resolved['scheduler']
        self.val_dataset = LongDataset(resolved['val_dataset'], shuffle=True)
        self.optimizer = self._get_optimizer(self.model, config)
        
        # Initialize EMA
        resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
        self.ema = PostHocEMA(self.model, **resolved['ema'])

        # Optional autoencoder for base KID evaluation
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
        opt_config = config['optimizer']
        if opt_config['type'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), **opt_config['kwargs'])
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
        return optimizer
    
    def get_accelerate_modules(self):
        """Returns modules that need to be passed to accelerator.prepare."""
        if self.model_g_pretrained is not None:
            return (self.model, self.model_m_pretrained, self.model_g_pretrained, self.optimizer)
        else:
            return (self.model, self.model_m_pretrained, self.optimizer)
    
    def set_prepared_modules(self, prepared_modules):
        """Set the modules returned from accelerator.prepare back into the trainer."""
        if self.model_g_pretrained is not None:
            self.model, self.model_m_pretrained, self.model_g_pretrained, self.optimizer = prepared_modules
        else:
            self.model, self.model_m_pretrained, self.optimizer = prepared_modules
        # Move EMA to device after models are prepared
        self.ema = self.ema.to(self.accelerator.device)
    
    def get_checkpoint_modules(self):
        """Returns a list of modules that need to be registered for checkpointing."""
        return [self.ema]
    
    def load_model_checkpoint(self, model_ckpt_path):
        """Load model weights from a checkpoint file."""
        # Consistency models load from pretrained diffusion models in __init__
        # so we don't need to do anything here
        pass
    
    def get_model_for_saving(self):
        """Returns the main model to save config for."""
        return self.model
    
    def train_step(self, state, batch):
        """Perform one training step."""
        images = batch['image']
        cond_img = batch.get('cond_img')
        conditional_inputs = batch.get('cond_inputs')

        with self.accelerator.accumulate(self.model):
            sigma_data = self.config['training']['sigma_data']
            sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * self.config['training']['P_std'] + self.config['training']['P_mean']).exp()
            
            t = torch.arctan(sigma / sigma_data)
            t.requires_grad_(True)
            
            # Generate z and x_t
            z = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * z
            
            # Calculate dx_t/dt using pretrained model
            with torch.no_grad():
                scaled_x_t = x_t / sigma_data
                if cond_img is not None:
                    scaled_x_t = torch.cat([scaled_x_t, cond_img], dim=1)
                
                m_pretrain_pred = self.model_m_pretrained(scaled_x_t, noise_labels=t.flatten(), conditional_inputs=conditional_inputs)
                
                if self.model_g_pretrained is not None:
                    g_pretrain_pred = self.model_g_pretrained(scaled_x_t, noise_labels=t.flatten(), conditional_inputs=conditional_inputs)
                    pretrain_pred = g_pretrain_pred + self.config['model']['guidance_scale'] * (m_pretrain_pred - g_pretrain_pred)
                else:
                    pretrain_pred = m_pretrain_pred
                
                dxt_dt = sigma_data * -pretrain_pred

            # Calculate current model prediction
            with self.accelerator.autocast():
                def model_wrapper(scaled_x_t, t):
                    if cond_img is not None:
                        scaled_x_t = torch.cat([scaled_x_t, cond_img], dim=1)
                    pred, logvar = self.model(scaled_x_t, noise_labels=t.flatten(), conditional_inputs=conditional_inputs, return_logvar=True)
                    return -pred, logvar
                
                v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
                v_t = torch.cos(t) * torch.sin(t)

                F_theta, F_theta_grad, logvar = torch.func.jvp(
                    model_wrapper, 
                    (x_t / sigma_data, t),
                    (v_x, v_t),
                    has_aux=True
                )
                F_theta_grad = F_theta_grad.detach()
                F_theta_minus = F_theta.detach()
                
                max_f_theta_grad_norm = torch.max(torch.sqrt(torch.mean(F_theta_grad**2, dim=(1,2,3))))
            
            # Warmup ratio
            r = min(1.0, (state['step'] + 1) / self.config['training'].get('warmup_steps', 10000) / self.accelerator.gradient_accumulation_steps)
            
            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            second_term = -r * torch.cos(t) * torch.sin(t) * x_t - r * sigma_data * F_theta_grad
            g = g + second_term
            
            # Tangent normalization
            if 'loss_groups' not in self.config['training']:
                g_norm = torch.sqrt(torch.mean(g**2, dim=(1,2,3), keepdim=True))
            else:
                g_norm_groups = []
                c = 0
                for group_channels in self.config['training']['loss_groups']:
                    g_norm_groups.append(torch.sqrt(torch.mean(g[:, c:c+group_channels]**2, dim=(1,2,3), keepdim=True)))
                    c += group_channels
                g_norm = torch.stack(g_norm_groups, dim=1).mean(dim=1, keepdim=True)
            
            g = g / (g_norm + self.config['training'].get('const_c', 0.1))
            max_g_norm = torch.max(g_norm)
            
            # Calculate loss with adaptive weighting
            weight = 1
            if self.config['training'].get('use_logvar', True):
                loss = weight * (1 / logvar.exp()) * torch.square(F_theta - F_theta_minus - g) + logvar
            else:
                loss = weight * torch.square(F_theta - F_theta_minus - g)
            
            if 'loss_groups' not in self.config['training']:
                loss = loss.mean()
            else:
                loss_groups = []
                c = 0
                for group_channels in self.config['training']['loss_groups']:
                    loss_groups.append(loss[:, c:c+group_channels].mean())
                    c += group_channels
                loss = torch.stack(loss_groups).mean()

            # Update state
            state['seen'] += images.shape[0]
            state['step'] += 1
            
            lr = self.resolved['lr_sched'].get(state['seen'])
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config['training'].get('gradient_clip_val', self.config['training'].get('gradient_clip_val', 100.0))
                )
            else:
                grad_norm = torch.tensor(0.0)
            self.optimizer.step()
            self.model.norm_weights()
            
            if self.accelerator.is_main_process:
                self.ema.update()

        return {
            'loss': loss.item(),
            'lr': lr,
            'grad_norm': grad_norm.item(),
            'max_f_theta_grad_norm': max_f_theta_grad_norm.item(),
            'max_g_norm': max_g_norm.item()
        }
    
    def _normalize_and_process_terrain(self, terrain):
        """Normalize single-channel terrain to [0, 255] uint8 3-channel for KID."""
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

    def _consistency_two_step(self, images, cond_img, conditional_inputs, generator):
        """2-step consistency sampling producing latents/samples matching decoder outputs."""
        sigma_data = self.scheduler.config.sigma_data
        samples = torch.zeros_like(images)
        t_values = [
            torch.arctan(self.scheduler.sigmas[0].to(images.device) / sigma_data)
        ]
        inter_t = self.config['evaluation'].get('inter_t', 1.1)
        if inter_t is not None:
            t_values += [torch.tensor(inter_t, device=images.device)]
        for t_scalar in t_values:
            t = t_scalar.view(1, 1, 1, 1).expand(images.shape[0], 1, 1, 1).to(images.device)
            z = torch.randn(images.shape, generator=generator, device=images.device) * sigma_data
            x_t = torch.cos(t) * samples + torch.sin(t) * z
            model_input = x_t / sigma_data
            if cond_img is not None:
                model_input = torch.cat([model_input, cond_img], dim=1)
            with torch.no_grad(), self.accelerator.autocast():
                pred = -self.model(model_input, noise_labels=t.flatten(), conditional_inputs=conditional_inputs)
            samples = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
        return samples / sigma_data

    def _calculate_decoder_kid(self, data_iter, generator):
        """KID for decoder models using 2-step consistency sampling."""
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

                # 2-step consistency sampling
                samples = self._consistency_two_step(images, cond_img, conditional_inputs, generator)

                # Single-channel decoder evaluation
                samples = samples[:, :1] / scheduler.config.sigma_data
                real_samples = images[:, :1] / scheduler.config.sigma_data
            
                residual_std = self.val_dataset.base_dataset.residual_std.to(images.device)
                residual_mean = self.val_dataset.base_dataset.residual_mean.to(images.device)
                output_full = laplacian_decode(samples * residual_std + residual_mean, lowfreq, extrapolate=True)
                images_full = laplacian_decode(images * residual_std + residual_mean, lowfreq, extrapolate=True)
                
                output_full = torch.sign(output_full) * torch.square(output_full)
                images_full = torch.sign(images_full) * torch.square(images_full)

                kid.update(self._normalize_and_process_terrain(samples), real=False)
                kid.update(self._normalize_and_process_terrain(real_samples), real=True)

                samples_generated += images.shape[0]
                pbar.update(images.shape[0])
            pbar.close()
            kid_mean, kid_std = kid.compute()
            print(f"Final Decoder KID Score (consistency 2-step): {kid_mean.item():.6f} ± {kid_std.item():.6f}")
            return {
                'val/kid_mean': kid_mean.item(),
                'val/kid_std': kid_std.item(),
            }

    def _calculate_base_kid(self, data_iter, generator):
        """KID for base models using 2-step consistency sampling and autoencoder decoding."""
        assert self.autoencoder is not None, "Autoencoder required for base KID evaluation"
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

                # 2-step consistency sampling
                samples = self._consistency_two_step(images, cond_img, conditional_inputs, generator)

                # Decode latents to terrain
                terrain = self._decode_latents_to_terrain(samples[:, :4], samples[:, 4:5], autoencoder, scheduler)
                terrain = torch.sign(terrain) * torch.square(terrain)
                
                real_terrain = batch['ground_truth']
                real_terrain = torch.sign(real_terrain) * torch.square(real_terrain)
                
                kid.update(self._normalize_and_process_terrain(terrain), real=False)
                kid.update(self._normalize_and_process_terrain(real_terrain), real=True)

                samples_generated += images.shape[0]
                pbar.update(images.shape[0])
            pbar.close()
            autoencoder = autoencoder.to('cpu')
            kid_mean, kid_std = kid.compute()
            print(f"Final KID Score (consistency 2-step): {kid_mean.item():.6f} ± {kid_std.item():.6f}")
            kid = kid.to('cpu')
            del kid
            return {
                'val/kid_mean': kid_mean.item(),
                'val/kid_std': kid_std.item(),
            }

    def evaluate(self):
        """Perform evaluation and return metrics using 2-step consistency sampling for KID."""
        # Build validation dataloader
        self.val_dataset.set_seed(self.config['training']['seed'] + 638)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            **self.resolved['dataloader_kwargs']
        )

        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config['training']['seed'] + 548)
        val_dataloader_iter = iter(self.val_dataloader)

        # Evaluate with EMA weights
        with temporary_ema_to_model(self.ema.ema_models[0]):
            output = {}
            mode = self.config.get('evaluation', {}).get('mode')
            if mode == 'base':
                output.update(self._calculate_base_kid(val_dataloader_iter, generator))
            elif mode == 'decoder':
                output.update(self._calculate_decoder_kid(val_dataloader_iter, generator))
            else:
                print('No evaluation mode specified, skipping KID calculation')
            torch.cuda.empty_cache()
            return output
        

