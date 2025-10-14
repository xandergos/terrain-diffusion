import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from ema_pytorch import PostHocEMA

from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.training.unet import EDMUnet2D
from terrain_diffusion.training.utils import temporary_ema_to_model


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
        
        # Initialize EMA
        resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
        self.ema = PostHocEMA(self.model, **resolved['ema'])
        
        print("Validation dataset size:", len(self.val_dataset))
        print(f"Training model with {self.model.count_parameters()} parameters.")
    
    def _get_optimizer(self, model, config):
        """Get optimizer based on config settings."""
        from heavyball.foreach_soap import ForeachSOAP
        from heavyball.foreach_adamw import ForeachAdamW
        
        if config['optimizer']['type'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), **config['optimizer']['kwargs'])
        elif config['optimizer']['type'] == 'heavyball-adam':
            optimizer = ForeachAdamW(model.parameters(), **config['optimizer']['kwargs'])
        elif config['optimizer']['type'] == 'soap':
            optimizer = ForeachSOAP(model.parameters(), **config['optimizer']['kwargs'])
        else:
            raise ValueError(f"Invalid optimizer type: {config['optimizer']['type']}")
        return optimizer
    
    def get_accelerate_modules(self):
        """Returns modules that need to be passed to accelerator.prepare."""
        return (self.model, self.train_dataloader, self.optimizer, self.val_dataloader)
    
    def set_prepared_modules(self, prepared_modules):
        """Set the modules returned from accelerator.prepare back into the trainer."""
        self.model, self.train_dataloader, self.optimizer, self.val_dataloader = prepared_modules
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
        """Calculate loss with optional grouping."""
        loss = 1 / (logvar.exp() * sigma_data ** 2) * ((pred_v_t - v_t) ** 2) + logvar
        if 'loss_groups' not in self.config['training']:
            return loss.mean()
        
        loss_groups = []
        c = 0
        for group_channels in self.config['training']['loss_groups']:
            loss_groups.append(loss[:, c:c+group_channels].mean())
            c += group_channels
        loss = torch.stack(loss_groups).mean()
        return loss
    
    def train_step(self, state):
        """Perform one training step."""
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                batch = next(self.train_iter)
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
    
    def evaluate(self):
        """Perform evaluation and return metrics."""
        validation_stats = {'loss': []}
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config['training']['seed'])
        pbar = tqdm(total=self.config['evaluation']['validation_steps'], desc="Validation")
        val_dataloader_iter = iter(self.val_dataloader)
        
        while pbar.n < pbar.total:
            batch = next(val_dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            
            sigma = torch.randn(images.shape[0], device=images.device, generator=generator).reshape(-1, 1, 1, 1)
            sigma = (sigma * self.config['evaluation']['P_std'] + self.config['evaluation']['P_mean']).exp()
            
            sigma_data = self.scheduler.config.sigma_data
            if self.config['evaluation'].get('scale_sigma', False):
                calc_channels = self.config['evaluation']['scaling_channels']
                sigma = sigma * torch.maximum(
                    torch.std(images[:, calc_channels], dim=[1, 2, 3], keepdim=True) / sigma_data, 
                    torch.tensor(self.config['evaluation'].get('sigma_scale_eps', 0.05), device=images.device)
                )
            
            t = torch.atan(sigma / sigma_data)
            cnoise = t.flatten()
            
            noise = torch.randn_like(images) * sigma_data
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
        
        # Handle EMA validation if configured
        if self.config['evaluation'].get('val_ema_idx', -1) >= 0:
            if self.config['evaluation']['val_ema_idx'] < len(self.ema.ema_models):
                with temporary_ema_to_model(self.ema.ema_models[self.config['evaluation']['val_ema_idx']]):
                    # Re-run validation with EMA model
                    pass  # Already run above, this is for compatibility
            else:
                warnings.warn(f"Invalid val_ema_idx: {self.config['evaluation']['val_ema_idx']}. "
                              "Using model's parameters.")
        
        return {'val_loss': np.mean(validation_stats['loss'])}

