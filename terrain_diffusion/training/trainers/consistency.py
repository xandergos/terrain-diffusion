import os
import torch
from ema_pytorch import PostHocEMA

from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.models.edm_unet import EDMUnet2D


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
        self.optimizer = self._get_optimizer(self.model, config)
        
        # Initialize EMA
        resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
        self.ema = PostHocEMA(self.model, **resolved['ema'])
        
        print(f"Training model with {self.model.count_parameters()} parameters.")
    
    def _get_optimizer(self, model, config):
        """Get optimizer based on config settings."""
        from heavyball.foreach_soap import ForeachSOAP
        from heavyball.foreach_adamw import ForeachAdamW
        
        opt_config = config['optimizer']
        if opt_config['type'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), **opt_config['kwargs'])
        elif opt_config['type'] == 'heavyball-adam':
            optimizer = ForeachAdamW(model.parameters(), **opt_config['kwargs'])
        elif opt_config['type'] == 'soap':
            optimizer = ForeachSOAP(model.parameters(), **opt_config['kwargs'])
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
    
    def evaluate(self):
        """Consistency models typically don't have separate evaluation."""
        return {}

