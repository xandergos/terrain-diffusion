import numpy as np
import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from tqdm import tqdm
from ema_pytorch import PostHocEMA

from terrain_diffusion.training.datasets.long_dataset import LongDataset
from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.training.utils import temporary_ema_to_model


def linear_warmup(start_value, end_value, current_step, total_steps):
    """Perform linear warmup from start_value to end_value."""
    if current_step >= total_steps:
        return end_value
    return start_value + (end_value - start_value) * (current_step / total_steps)


def calculate_fid(generator, val_dataset, config, device, n_samples=50000):
    """Calculate FID score between generated samples and validation dataset."""
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Constants for denormalization and renormalization
    MEAN = -2607
    STD = 2435
    MIN_ELEVATION = -10000
    MAX_ELEVATION = 9000
    ELEVATION_RANGE = MAX_ELEVATION - MIN_ELEVATION
    
    def process_images(images):
        images = images * STD + MEAN
        images = torch.clamp(images, MIN_ELEVATION, MAX_ELEVATION)
        images = ((images - MIN_ELEVATION) * 255 / ELEVATION_RANGE).to(torch.uint8)
        images = images.repeat(1, 3, 1, 1)
        return images
    
    pbar = tqdm(total=n_samples, desc="Calculating FID")
    
    latent_size = config['training']['latent_size']
    latent_channels = generator.config.latent_channels

    # Use the same real images to compute both real and fake statistics
    processed = 0
    val_loader = DataLoader(val_dataset, batch_size=64)
    generator.eval()
    with torch.no_grad():
        while processed < n_samples:
            for batch in val_loader:
                if processed >= n_samples:
                    break
                images = batch['image'].to(device)
                
                # Limit to remaining
                take_n = min(images.shape[0], n_samples - processed)
                images = images[:take_n]

                # Sample random per-channel t in [0, pi/2]
                num_channels = images.shape[1]
                t = torch.rand(images.shape[0], num_channels, device=device) * (torch.pi / 2)
                
                # Pure noise input
                t = torch.full((images.shape[0], num_channels), np.arctan(160), device=device)
                
                # Form mixed inputs using the same real images and fresh noise
                latent = torch.randn(images.shape[0], latent_channels, latent_size, latent_size, device=device)
                z_img = torch.randn_like(images)
                mixed_input = torch.cos(t)[..., None, None] * images + torch.sin(t)[..., None, None] * z_img
                fake_images, _ = generator(latent, mixed_input, t)
                fake_images = fake_images[:, :1]

                # Crop real images to match the current fake images size if needed
                h_diff = images.shape[-2] - fake_images.shape[-2]
                w_diff = images.shape[-1] - fake_images.shape[-1]
                h_start = h_diff // 2
                w_start = w_diff // 2
                images = images[:, :, h_start:h_start+fake_images.shape[-2], w_start:w_start+fake_images.shape[-1]]

                fid.update(process_images(images[:, :1]), real=True)
                fid.update(process_images(fake_images), real=False)
                pbar.update(images.shape[0])
                processed += images.shape[0]
    return float(fid.compute())


class InjectionGANTrainer(Trainer):
    """Trainer for GAN models."""
    
    def __init__(self, config, resolved, accelerator, state):
        self.config = config
        self.resolved = resolved
        self.accelerator = accelerator
        self.state = state
        
        # Load models and datasets
        self.generator = resolved['generator']
        self.discriminator = resolved['discriminator']
        self.train_dataset = resolved['train_dataset']
        self.val_dataset = LongDataset(resolved['val_dataset'], shuffle=True)
        
        # Setup optimizers
        self.g_optimizer = self._get_optimizer(self.generator, config['g_optimizer'])
        self.d_optimizer = self._get_optimizer(self.discriminator, config['d_optimizer'])
        
        # Warmup parameters
        self.burnin_steps = config['training'].get('burnin_steps', 0)
        self.initial_r_gamma = config['training'].get('r_gamma', 0) * config['training'].get('r_warmup_factor', 10)
        self.final_r_gamma = config['training'].get('r_gamma', 0)
        self.initial_beta_2 = 1 - 10 * (1 - config['g_optimizer']['kwargs']['betas'][1])
        self.final_beta_2 = config['g_optimizer']['kwargs']['betas'][1]
        
        self.printed_size = False
        
        # Initialize EMA
        resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
        self.ema = PostHocEMA(self.generator, **resolved['ema'])
        
        print("Warming beta_2 from", self.initial_beta_2, "to", self.final_beta_2)
        
    def _get_optimizer(self, model, config):
        """Get optimizer based on config settings."""
        if config['type'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), **config['kwargs'])
        else:
            raise ValueError(f"Invalid optimizer type: {config['type']}")
        return optimizer
        
    def get_accelerate_modules(self):
        """Returns modules that need to be passed to accelerator.prepare."""
        return (self.generator, self.discriminator, self.g_optimizer, self.d_optimizer)
    
    def set_prepared_modules(self, prepared_modules):
        """Set the modules returned from accelerator.prepare back into the trainer."""
        self.generator, self.discriminator, self.g_optimizer, self.d_optimizer = prepared_modules
        # Move EMA to device after models are prepared
        self.ema = self.ema.to(self.accelerator.device)
    
    def get_checkpoint_modules(self):
        """Returns a list of modules that need to be registered for checkpointing."""
        return [self.ema]
    
    def load_model_checkpoint(self, model_ckpt_path):
        """Load model weights from a checkpoint file."""
        # GAN models typically don't load from pretrained checkpoints
        # but we keep this for interface consistency
        pass
    
    def get_model_for_saving(self):
        """Returns the main model to save config for."""
        return self.generator
    
    def train_step(self, state, batch):
        """Perform one training step."""
        # Warmup r_gamma and beta_2 during burnin steps
        if state['step'] < self.burnin_steps:
            current_r_gamma = linear_warmup(
                self.initial_r_gamma, 
                self.final_r_gamma, 
                state['step'], 
                self.burnin_steps
            )
            current_beta_2 = linear_warmup(
                self.initial_beta_2, 
                self.final_beta_2, 
                state['step'], 
                self.burnin_steps
            )
            
            # Update beta_2 for both optimizers
            for optimizer in [self.g_optimizer, self.d_optimizer]:
                for group in optimizer.param_groups:
                    group['betas'] = (group['betas'][0], current_beta_2)
        else:
            current_r_gamma = self.final_r_gamma

        pct_fixed = linear_warmup(
            self.config['training'].get('warmup_pct_fixed', 0.95),
            0.5,
            state['step'],
            self.config['training'].get('burnin_steps', 1)
        )
                
        def sample_t(bs, channels):
            t = torch.rand(bs, channels, device=self.accelerator.device)
            t = torch.atan(2 * torch.exp(10 * t - 3))
            
            # Randomly make half the batch = arctan(160)
            mask = torch.rand(bs, device=self.accelerator.device) < pct_fixed
            t[mask] = torch.atan(torch.tensor(160.0, device=self.accelerator.device))
            
            return t

        latent_size = self.config['training']['latent_size']
        latent_channels = self.generator.config.latent_channels
        # Train discriminator
        with self.accelerator.accumulate(self.discriminator):
            real_images = batch['image']
            batch_size = real_images.shape[0]
            
            with self.accelerator.autocast():
                self.discriminator.train()
                
                t = sample_t(batch_size, real_images.shape[1])
                latent = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=self.accelerator.device)
                z_img = torch.randn_like(real_images)
                mixed_real = torch.cos(t)[..., None, None] * real_images + torch.sin(t)[..., None, None] * z_img

                # Generate fake using mixed input and the same t
                with torch.no_grad():
                    fake_images, _ = self.generator(latent, mixed_real, t)
                    
                if not self.printed_size:
                    print("Fake image size:", fake_images.shape)
                    print("Real image size:", real_images.shape)
                    self.printed_size = True
                    
                cond_cat = torch.cat([mixed_real, mixed_real], dim=0)
                
                # Central crop real_images to match fake_images size
                real_images_uncropped = real_images
                
                h_diff = real_images.shape[-2] - fake_images.shape[-2]
                w_diff = real_images.shape[-1] - fake_images.shape[-1]
                h_start = h_diff // 2
                w_start = w_diff // 2
                real_images = real_images[:, :, h_start:h_start+fake_images.shape[-2], w_start:w_start+fake_images.shape[-1]]
                
                img_cat = torch.cat([real_images, fake_images.detach()], dim=0).detach().requires_grad_(True)
                pred_cat = self.discriminator(img_cat)
                real_pred, fake_pred = pred_cat[:batch_size], pred_cat[batch_size:]
                
                d_loss = torch.nn.functional.softplus(fake_pred - real_pred).mean()
                
                if self.config['training'].get('r_gamma', 0) > 0 and state['step'] % self.config['training'].get('r_interval', 16) == 0:
                    grad_real = torch.autograd.grad(
                        outputs=pred_cat.sum(), inputs=img_cat,
                        create_graph=True, retain_graph=True
                    )[0]
                    r_reg = current_r_gamma * 0.5 * grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
                    total_d_loss = d_loss + r_reg
                else:
                    r_reg = 0
                    total_d_loss = d_loss

            self.d_optimizer.zero_grad()
            self.accelerator.backward(total_d_loss)
            if self.accelerator.sync_gradients:
                discriminator_grad_norm = self.accelerator.clip_grad_norm_(self.discriminator.parameters(), 100.0)
            self.d_optimizer.step()

        # Train generator
        with self.accelerator.accumulate(self.generator):
            with self.accelerator.autocast():
                self.discriminator.eval()
                
                t_g = sample_t(batch_size, real_images_uncropped.shape[1])
                latent_g = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=self.accelerator.device)
                z_img_g = torch.randn_like(real_images_uncropped)
                mixed_real_g = torch.cos(t_g)[..., None, None] * real_images_uncropped + torch.sin(t_g)[..., None, None] * z_img_g
                fake_images, v_t_pred = self.generator(latent_g, mixed_real_g, t_g)

                fake_pred = self.discriminator(fake_images)
                g_loss = torch.nn.functional.softplus(real_pred.detach() - fake_pred).mean()
                
                mean = fake_images.mean(dim=(0, 2, 3))
                std = fake_images.std(dim=(0, 2, 3))
                kl_loss = (
                    torch.log(1 / (std + 1e-8)) +
                    (std**2 + mean**2) / 2 - 0.5
                ).mean()
                total_g_loss = g_loss + kl_loss * self.config['training'].get('kl_weight', 0.0) * linear_warmup(
                    self.config['training'].get('kl_warmup_factor', 1.0),
                    1.0,
                    state['step'],
                    self.config['training'].get('burnin_steps', 1)
                )
                
                h_diff = z_img.shape[-2] - fake_images.shape[-2]
                w_diff = z_img.shape[-1] - fake_images.shape[-1]
                h_start = h_diff // 2
                w_start = w_diff // 2
                z_img_cropped = z_img[:, :, h_start:h_start+fake_images.shape[-2], w_start:w_start+fake_images.shape[-1]]
                v_t = torch.cos(t[..., None, None]) * z_img_cropped - torch.sin(t[..., None, None]) * real_images
                pred_v_t = v_t_pred
                diffusion_error = (pred_v_t - v_t).pow(2).mean()
                total_g_loss = total_g_loss + diffusion_error * self.config['training'].get('diffusion_error_weight', 0.0)
                
            self.g_optimizer.zero_grad()
            self.accelerator.backward(total_g_loss)
            if self.accelerator.sync_gradients:
                generator_grad_norm = self.accelerator.clip_grad_norm_(self.generator.parameters(), 10.0)
            self.g_optimizer.step()
            
        if self.accelerator.is_main_process:
            self.ema.update()
        
        # Update learning rates
        # Update state
        state['seen'] += batch_size
        state['step'] += 1
        
        lr_warmup = linear_warmup(
            self.config['training'].get('lr_warmup_factor', 1.0), 
            1.0, 
            state['step'], 
            self.config['training'].get('burnin_steps', 1)
        )
        lr = self.resolved['lr_sched'].get(state['seen']) * lr_warmup
        for g in self.g_optimizer.param_groups:
            g['lr'] = lr
        for g in self.d_optimizer.param_groups:
            g['lr'] = lr * self.config['training'].get('disc_lr_mult', 1.0)
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'kl_loss': kl_loss.item(),
            'diffusion_error': diffusion_error.item(),
            'r_loss': (r_reg.item() / current_r_gamma) if current_r_gamma > 0 else 0,
            'd_grad_norm': discriminator_grad_norm.item(),
            'g_grad_norm': generator_grad_norm.item(),
            'lr': lr
        }
    
    def evaluate(self):
        """Perform evaluation and return metrics."""
        self.generator.eval()
        with temporary_ema_to_model(self.ema.ema_models[0]):
            self.val_dataset.set_seed(self.config['training']['seed'] + 123)
            fid = calculate_fid(self.generator, self.val_dataset, self.config, self.accelerator.device)
            print(f"FID: {fid}")
        self.generator.train()
        return {'val/fid': fid}

