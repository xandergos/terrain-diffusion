import numpy as np
import os
import torch
from torchvision.transforms.v2.functional import gaussian_blur
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

def random_crop(images, crop_size):
    """Randomly crop images to a square of size crop_size."""
    h_diff = images.shape[-2] - crop_size
    w_diff = images.shape[-1] - crop_size
    assert h_diff >= 0 and w_diff >= 0
    if h_diff == 0 and w_diff == 0:
        return images
    
    batch_size = images.shape[0]
    h_starts = torch.randint(0, h_diff + 1, (batch_size,))
    w_starts = torch.randint(0, w_diff + 1, (batch_size,))
    
    # Crop each image independently
    cropped = []
    for i in range(batch_size):
        h_start = h_starts[i].item()
        w_start = w_starts[i].item()
        cropped.append(images[i:i+1, :, h_start:h_start + crop_size, w_start:w_start + crop_size])
    
    return torch.cat(cropped, dim=0)

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
    
    # Process real and fake samples simultaneously
    val_loader = DataLoader(val_dataset, batch_size=64)
    generator.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if pbar.n >= pbar.total:
                break
            
            real_images = batch['image'].to(device)
            batch_size = real_images.shape[0]
            real_images = random_crop(real_images, config['training']['crop_size'])
            real_images = process_images(real_images[:, :1])
            
            z = torch.randn(batch_size, config['generator']['latent_channels'],
                          config['training']['latent_size'], 
                          config['training']['latent_size'],
                          device=device)
            if config['training'].get('mode') == 'inject':
                t = torch.atan(torch.tensor(160.0, device=device, dtype=torch.float32)).view(1, 1, 1, 1).repeat(batch_size, real_images.shape[1])
                fake_images = generator(z, torch.randn(real_images.shape, device=device), t)[:, :1]
            else:
                fake_images = generator(z)[:, :1]
            
            # Crop to match real image size if needed
            fake_images = random_crop(fake_images, config['training']['crop_size'])
            fake_images = process_images(fake_images)
            
            fid.update(fake_images, real=False)
            fid.update(real_images, real=True)
            pbar.update(real_images.shape[0])
    
    fid = float(fid.compute())
    print(f"FID: {fid}")
    return fid


class GANTrainer(Trainer):
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
            self.config['training'].get('warmup_pct_fixed', 0.5),
            self.config['training'].get('pct_fixed', 0.5),
            state['step'],
            self.config['training'].get('burnin_steps', 1)
        )
        def sample_t(bs, channels):
            t = torch.rand(bs, channels, device=self.accelerator.device)
            t = torch.atan(2 * torch.exp(8 * t - 3))
            
            # Randomly make half the batch = arctan(160)
            mask = torch.rand(bs, device=self.accelerator.device) < pct_fixed
            t[mask] = torch.atan(torch.tensor(160.0, device=self.accelerator.device))
            
            return t
        
        # Train discriminator
        with self.accelerator.accumulate(self.discriminator):
            real_images = batch['image']
            real_images_uncropped = real_images
            batch_size = real_images.shape[0]
            
            with self.accelerator.autocast():
                self.discriminator.train()
                
                with torch.no_grad():
                    z = torch.randn(batch_size, self.config['generator']['latent_channels'],
                                self.config['training']['latent_size'], 
                                self.config['training']['latent_size'],
                                device=self.accelerator.device)
                    if self.config['training'].get('mode') == 'inject':
                        t = sample_t(batch_size, real_images.shape[1])
                        z_img = torch.randn_like(real_images_uncropped)
                        mixed_real = torch.cos(t)[..., None, None] * real_images_uncropped + torch.sin(t)[..., None, None] * z_img
                        fake_images = self.generator(z, mixed_real, t)
                    else:
                        fake_images = self.generator(z)
                
                if not self.printed_size:
                    print("Fake image size:", fake_images.shape)
                    print("Real image size:", real_images.shape)
                    self.printed_size = True
                    
                real_images = random_crop(real_images, self.config['training']['crop_size'])
                fake_images = random_crop(fake_images, self.config['training']['crop_size'])
                all_images = torch.cat([real_images, fake_images], dim=0).detach().requires_grad_(True)
                pred = self.discriminator(all_images)
                real_pred = pred[:batch_size]
                fake_pred = pred[batch_size:]
                
                d_loss = torch.nn.functional.softplus(fake_pred - real_pred).mean()
                
                if self.config['training'].get('r_gamma', 0) > 0 and state['step'] % self.config['training'].get('r_interval', 16) == 0:
                    grad_real = torch.autograd.grad(
                        outputs=pred.sum(), inputs=all_images,
                        create_graph=True)[0]
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
                z = torch.randn(batch_size, self.config['generator']['latent_channels'],
                            self.config['training']['latent_size'], 
                            self.config['training']['latent_size'],
                            device=self.accelerator.device)
                
                if self.config['training'].get('mode') == 'inject':
                    t = sample_t(batch_size, real_images.shape[1])
                    z_img = torch.randn_like(real_images_uncropped)
                    mixed_real = torch.cos(t)[..., None, None] * real_images_uncropped + torch.sin(t)[..., None, None] * z_img
                    fake_images = self.generator(z, mixed_real, t)
                else:
                    fake_images = self.generator(z)
                
                fake_pred = self.discriminator(fake_images)
                g_loss = torch.nn.functional.softplus(real_pred.detach() - fake_pred).mean()
                
                real_mean = 0
                real_std = 1

                mean = fake_images.mean(dim=(0, 2, 3))
                std = fake_images.std(dim=(0, 2, 3))
                kl_loss = (
                    torch.log(real_std / (std + 1e-8)) +
                    (std**2 + (mean - real_mean)**2) / (2 * (real_std**2 + 1e-8)) - 0.5
                ).mean()
                total_g_loss = g_loss + kl_loss * self.config['training'].get('kl_weight', 0.0)
                
                # Range loss
                below_min = torch.nn.functional.relu(-2 - fake_images)
                above_max = torch.nn.functional.relu(fake_images - 3.2)
                range_loss = (below_min**2 + above_max**2).mean()
                total_g_loss = total_g_loss + range_loss * self.config['training'].get('range_weight', 1.0)
            
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
            'range_loss': range_loss.item(),
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
        self.generator.train()
        return {'val/fid': fid}

