from collections import defaultdict
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from ema_pytorch import PostHocEMA

from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.models.edm_autoencoder import EDMAutoencoder


class AutoencoderTrainer(Trainer):
    """Trainer for autoencoder models."""
    
    def __init__(self, config, resolved, accelerator, state):
        self.config = config
        self.resolved = resolved
        self.accelerator = accelerator
        self.state = state
        
        # Load model and datasets
        self.model = resolved['model']
        assert isinstance(self.model, EDMAutoencoder), "Currently only EDMAutoencoder is supported."
        
        self.train_dataset = resolved['train_dataset']
        self.val_dataset = resolved['val_dataset']
        
        # Setup optimizer
        self.optimizer = self._get_optimizer(self.model, config)
        
        # Setup dataloaders
        self.train_dataloader = DataLoader(
            LongDataset(self.train_dataset, shuffle=True), 
            batch_size=config['training']['train_batch_size'],
            **resolved['dataloader_kwargs'], 
            drop_last=True
        )
        self.val_dataloader = DataLoader(
            LongDataset(self.val_dataset, shuffle=True), 
            batch_size=config['training']['train_batch_size'],
            **resolved['dataloader_kwargs'], 
            drop_last=True
        )
        
        # Perceptual loss
        self.perceptual_loss = lpips.LPIPS(net='alex', spatial=True)
        
        # Loss weights
        self.mse_weight = config['training'].get('mse_weight', 1.0)
        self.perceptual_weight = config['training'].get('perceptual_weight', 1.0)
        
        # Initialize EMA
        resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
        self.ema = PostHocEMA(self.model, **resolved['ema'])
        
        print(f"Training model with {self.model.count_parameters()} parameters.")
    
    def _get_optimizer(self, model, config):
        """Get optimizer based on config settings."""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found for optimizer.")
        if config['optimizer']['type'] == 'adam':
            optimizer = torch.optim.Adam(trainable_params, **config['optimizer']['kwargs'])
        else:
            raise ValueError(f"Unknown optimizer type: {config['optimizer']['type']}")
        return optimizer
    
    def get_accelerate_modules(self):
        """Returns modules that need to be passed to accelerator.prepare."""
        return (self.model, self.train_dataloader, self.optimizer, 
                self.val_dataloader, self.perceptual_loss)
    
    def set_prepared_modules(self, prepared_modules):
        """Set the modules returned from accelerator.prepare back into the trainer."""
        self.model, self.train_dataloader, self.optimizer, \
            self.val_dataloader, self.perceptual_loss = prepared_modules
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
    
    def get_model_for_saving(self):
        """Returns the main model to save config for."""
        return self.model
    
    def _calculate_loss(self, reconstruction, reference):
        """Calculate MSE + perceptual loss."""
        # MSE loss
        mse_loss = torch.nn.functional.mse_loss(reconstruction, reference)
        
        # Perceptual loss with normalization
        ref_min = torch.amin(reference, dim=(1, 2, 3), keepdim=True)
        ref_max = torch.amax(reference, dim=(1, 2, 3), keepdim=True)
        eps = 0.1
        
        ref_range = torch.maximum((ref_max - ref_min) * 1.1, torch.tensor(eps))
        ref_center = (ref_min + ref_max) / 2
        
        normalized_ref = ((reference - ref_center) / ref_range * 2)
        normalized_rec = ((reconstruction - ref_center) / ref_range * 2)
        normalized_rec = normalized_rec.clamp(-1, 1)
        
        perceptual_loss = self.perceptual_loss(
            normalized_ref.repeat(1, 3, 1, 1), 
            normalized_rec.repeat(1, 3, 1, 1)
        ).mean()
        
        # Weighted combination
        total_loss = self.mse_weight * mse_loss + self.perceptual_weight * perceptual_loss
        
        return total_loss, mse_loss, perceptual_loss
    
    def train_step(self, state):
        """Perform one training step."""
        stats = {}
        
        # Fetch batch for autoencoder training
        batch = next(self.train_iter)
        images = batch['image']
        cond_img = batch.get('cond_img')
        conditional_inputs = batch.get('cond_inputs')

        # Train autoencoder
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                scaled_clean_images = images
                if cond_img is not None:
                    scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
                
                z_means, z_logvars = self.model.preencode(scaled_clean_images, conditional_inputs)
                z = self.model.postencode(z_means, z_logvars)
                decoded_x, logvar = self.model.decode(z, include_logvar=True)

                # Calculate reconstruction losses
                recon_loss, mse_loss, perceptual_loss = self._calculate_loss(decoded_x, scaled_clean_images)
                
                # KL loss
                ndz_logvars = z_logvars[:, :self.model.config.latent_channels]
                ndz_means = z_means[:, :self.model.config.latent_channels]
                kl_loss = -0.5 * torch.mean(1 + ndz_logvars - ndz_means**2 - ndz_logvars.exp())
                
                total_loss = recon_loss + kl_loss * self.config['training']['kl_weight']
            
            self.optimizer.zero_grad()
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip_val'])
            self.optimizer.step()

        if self.accelerator.is_main_process:
            self.ema.update()
        
        # Update learning rate
        lr = self.resolved['lr_sched'].get(state['seen'])
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        
        stats['loss'] = total_loss.item()
        stats['recon_loss'] = recon_loss.item()
        stats['mse_loss'] = mse_loss.item()
        stats['perceptual_loss'] = perceptual_loss.item()
        stats['kl_loss'] = kl_loss.item()
        stats['lr'] = lr
        stats['grad_norm'] = grad_norm
        
        return stats
    
    def evaluate(self):
        """Perform evaluation and return metrics."""
        validation_stats = defaultdict(list)
        pbar = tqdm(total=self.config['evaluation']['validation_steps'], desc="Validation")
        val_dataloader_iter = iter(self.val_dataloader)
        
        fid_metric = None
        if self.config['evaluation'].get('eval_fid', False):
            fid_metric = FrechetInceptionDistance(feature=2048).to(self.accelerator.device)
        
        while pbar.n < pbar.total:
            batch = next(val_dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            
            self.model.eval()
            with torch.no_grad(), self.accelerator.autocast():
                scaled_clean_images = images
                if cond_img is not None:
                    scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
                
                z_means, z_logvars = self.model.preencode(scaled_clean_images, conditional_inputs)
                z = self.model.postencode(z_means, z_logvars)
                decoded_x, logvar = self.model.decode(z, include_logvar=True)

                # Calculate reconstruction losses
                recon_loss, mse_loss, perceptual_loss = self._calculate_loss(decoded_x, scaled_clean_images)
                
                ndz_logvars = z_logvars[:, :self.model.config.latent_channels]
                ndz_means = z_means[:, :self.model.config.latent_channels]
                kl_loss = -0.5 * torch.mean(1 + ndz_logvars - ndz_means**2 - ndz_logvars.exp())
                total_loss = recon_loss + kl_loss * self.config['training']['kl_weight']

                # Update FID metric if enabled
                if fid_metric is not None:
                    try:
                        pred_residual = decoded_x[:, :1]
                        real_residual = scaled_clean_images[:, :1]
                        real_min = torch.amin(real_residual, dim=(1, 2, 3), keepdim=True)
                        real_max = torch.amax(real_residual, dim=(1, 2, 3), keepdim=True)
                        value_range = torch.maximum(real_max - real_min, torch.tensor(1.0, device=real_residual.device))
                        value_mid = (real_min + real_max) / 2
                        samples_norm = torch.clamp(((pred_residual - value_mid) / value_range + 0.5) * 255, 0, 255)
                        samples_norm = samples_norm.repeat(1, 3, 1, 1).to(torch.uint8)
                        real_norm = torch.clamp(((real_residual - value_mid) / value_range + 0.5) * 255, 0, 255)
                        real_norm = real_norm.repeat(1, 3, 1, 1).to(torch.uint8)
                        fid_metric.update(samples_norm, real=False)
                        fid_metric.update(real_norm, real=True)
                    except Exception:
                        pass

            validation_stats['loss'].append(total_loss.item())
            validation_stats['recon_loss'].append(recon_loss.item())
            validation_stats['mse_loss'].append(mse_loss.item())
            validation_stats['perceptual_loss'].append(perceptual_loss.item())
            validation_stats['kl_loss'].append(kl_loss.item())
            
            pbar.update(images.shape[0])
            pbar.set_postfix({k: f"{np.mean(v):.3f}" for k, v in validation_stats.items()})
        
        # Return average losses
        out_stats = {f'val/{k}': np.mean(v) for k, v in validation_stats.items()}
        if fid_metric is not None:
            out_stats['val/fid'] = fid_metric.compute().item()
        
        return out_stats

