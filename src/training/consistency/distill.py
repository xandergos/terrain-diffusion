from functools import partial
import click
import torch
import torch.nn as nn
import numpy as np
import json
import catalogue
import click
from datetime import datetime
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from ema_pytorch import PostHocEMA
from diffusion.datasets.datasets import LongDataset
from src.data.laplacian_encoder import *
from diffusion.registry import build_registry
from diffusion.samplers.tiled import TiledSampler
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from schedulefree import ScheduleFreeWrapper, AdamWScheduleFree
from src.training.utils import SerializableEasyDict as EasyDict
from diffusion.unet import EDMUnet2D

@click.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False)
@click.option("--debug-run", "debug_run", is_flag=True, default=False)
def distill(config_path, ckpt_path, debug_run):
    build_registry()
    
    config = Config().from_disk(config_path) if config_path else None
    resolved = registry.resolve(config, validate=False)

    # Load anything that needs to be used for training.
    model_pretrained = EDMUnet2D.from_pretrained(resolved['model']['path'])
    model = EDMUnet2D.from_pretrained(resolved['model']['path'])
    
    # Reset logvar weights
    model.logvar_linear.weight.data.copy_(torch.randn_like(model.logvar_linear.weight.data))
    model_pretrained.eval()
    model.eval()
    
    lr_scheduler = resolved['lr_sched']
    train_dataset = resolved['train_dataset']
    scheduler = resolved['scheduler']
    if resolved['optimizer']['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **resolved['optimizer']['kwargs'])
        sf_optim = False
    elif resolved['optimizer']['type'] == 'sf-adam':
        optimizer = AdamWScheduleFree(model.parameters(), **resolved['optimizer']['kwargs'])
        optimizer.eval()
        sf_optim = True
    resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
    ema = PostHocEMA(model, **resolved['ema'])
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    dataloader = DataLoader(LongDataset(train_dataset), batch_size=config['training']['train_batch_size'],
                            **resolved['dataloader_kwargs'])
        
    print(f"Training model with {model.count_parameters()} parameters.")

    # Setup accelerate
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=resolved['training']['gradient_accumulation_steps'],
        log_with=None
    )
    ema = ema.to(accelerator.device)
    model, model_pretrained, dataloader, optimizer = accelerator.prepare(model, model_pretrained, dataloader, optimizer)
    accelerator.register_for_checkpointing(state)
    accelerator.register_for_checkpointing(ema)
    
    # Load from checkpoint if needed
    if ckpt_path:
        accelerator.load_state(ckpt_path)
    
    # Save full train config
    if not debug_run:
        os.makedirs(os.path.join(resolved['logging']['save_dir'], 'configs'), exist_ok=True)
        with open(os.path.join(resolved['logging']['save_dir'], 'configs', f'config_{state.seen//1000}kimg.json'), 'w') as f:
            json.dump(config, f)
        with open(os.path.join(resolved['logging']['save_dir'], 'configs', f'config_latest.json'), 'w') as f:
            json.dump(config, f)
            
        # Save model config
        os.makedirs(os.path.join(resolved['logging']['save_dir'], 'configs'), exist_ok=True)
        model.save_config(os.path.join(resolved['logging']['save_dir'], 'configs', f'model_config_{state.seen//1000}kimg'))
        model.save_config(os.path.join(resolved['logging']['save_dir'], 'configs', f'model_config_latest'))
        
    if accelerator.is_main_process:
        if debug_run:
            config['wandb']['mode'] = 'disabled'
        wandb.init(
            **config['wandb'],
            config=config
        )
        
    def safe_rmtree(path):
        """Removes a tree but only checkpoint files."""
        for fp in os.listdir(path):
            if os.path.isdir(os.path.join(path, fp)):
                safe_rmtree(os.path.join(path, fp))
            else:
                legal_extensions = ['.bin', '.safetensors', '.pkl', '.pt', '.json', '.md']
                for ext in legal_extensions:
                    if fp.endswith(ext):
                        os.remove(os.path.join(path, fp))
                        break
        os.rmdir(path)

    def save_checkpoint(base_folder_path, overwrite=False):
        if os.path.exists(base_folder_path + '_checkpoint') and not overwrite:
            strtime = datetime.now().strftime("_%Y%m%d_%H%M%S")
            base_folder_path = f"{base_folder_path}{strtime}"
        elif os.path.exists(base_folder_path + '_checkpoint'):
            safe_rmtree(base_folder_path + '_checkpoint')
        os.makedirs(base_folder_path + '_checkpoint', exist_ok=False)
        accelerator.save_state(base_folder_path + '_checkpoint')
        
        torch.save(ema.state_dict(), os.path.join(base_folder_path + '_checkpoint', 'phema.pt'))

    grad_norm = torch.tensor(0.0, device=accelerator.device)
    dataloader_iter = iter(dataloader)
    while state.epoch < config['training']['epochs']:
        stats_hist = {'loss': [], 'importance_loss': []}
        progress_bar = tqdm(dataloader_iter, desc=f"Epoch {state.epoch}", total=config['training']['epoch_steps'])
        while progress_bar.n < config['training']['epoch_steps']:
            batch = next(dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')

            with accelerator.accumulate(model):
                sigma_data = scheduler.config.sigma_data
                sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
                sigma = (sigma * config['training']['P_std'] + config['training']['P_mean']).exp()  # Sample τ from proposal distribution
                t = torch.arctan(sigma / sigma_data)  # Convert to t using arctan
                t.requires_grad_(True)
                
                # Generate z and x_t
                z = torch.randn_like(images) * sigma_data
                x_t = torch.cos(t) * images + torch.sin(t) * z
                
                # Replace 0.1% of sigmas with uniform random values to ensure very large sigmas are reached
                num_to_replace = np.random.poisson(config['training']['random_replace_rate'] * sigma.numel())
                max_train_sigma = scheduler.config.sigma_max / scheduler.config.scaling_t
                uniform_sigma = torch.rand(num_to_replace, device=sigma.device) * (max_train_sigma - scheduler.config.sigma_min) + scheduler.config.sigma_min
                sigma.view(-1)[:num_to_replace] = uniform_sigma
                
                # Calculate dx_t/dt using pretrained model
                with torch.no_grad():
                    scaled_x_t = x_t / sigma_data
                    if cond_img is not None:
                        scaled_x_t = torch.cat([scaled_x_t, cond_img], dim=1)
                    pretrain_pred = model_pretrained(scaled_x_t, noise_labels=t.flatten(), conditional_inputs=conditional_inputs)
                    dxt_dt = sigma_data * -pretrain_pred

                # Calculate current model prediction
                if sf_optim:
                    optimizer.train()
                with accelerator.autocast():
                    def model_wrapper(scaled_x_t, t):
                        if cond_img is not None:
                            scaled_x_t = torch.cat([scaled_x_t, cond_img], dim=1)
                        pred, logvar = model(scaled_x_t, noise_labels=t.flatten(), conditional_inputs=conditional_inputs, return_logvar=True)
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
                
                # Warmup ratio
                r = min(1.0, state.step / config['training'].get('warmup_steps', 10000))
                # Calculate gradient g using JVP rearrangement
                g = -torch.cos(t)**2 * (sigma_data * F_theta_minus - dxt_dt)
                second_term = -r * torch.cos(t) * torch.sin(t) * x_t - r * sigma_data * F_theta_grad
                g = g + second_term
                g = g.detach()
                
                # Tangent normalization
                g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
                g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())
                g = g / (g_norm + config['training'].get('const_c', 0.1))
                
                # Tangent clipping (Only use this OR normalization)
                # g = torch.clamp(g, min=-1, max=1)
                
                # Calculate loss with adaptive weighting
                weight = 1 / sigma
                loss = (weight / torch.exp(logvar)) * torch.square(F_theta - F_theta_minus - g) + logvar
                loss = loss.mean()

                state.seen += images.shape[0]
                state.step += 1
                lr = lr_scheduler.get(state.seen)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=config['training'].get('gradient_clip_val', 1.0))
                optimizer.step()
                model.norm_weights()
                if accelerator.is_main_process:
                    if sf_optim:
                        optimizer.eval()
                    ema.update()

                stats_hist['loss'].append(loss.item())
                progress_bar.set_postfix({'loss': np.mean(stats_hist['loss']), 
                                        "lr": lr,
                                        "grad_norm": grad_norm.item()})
                progress_bar.update(1)

        state.epoch += 1
        if accelerator.is_main_process:
            wandb.log({
                "avg_loss": np.mean(stats_hist['loss']),
                "loss": np.median(stats_hist['loss']),
                "lr": lr,
                "step": state.step,
                "epoch": state.epoch,
                "seen": state.seen
            }, step=state.epoch)
            if sf_optim:
                optimizer.eval()
            if state.epoch % config['logging']['temp_save_epochs'] == 0:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0:
                save_checkpoint(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")

if __name__ == '__main__':
    distill()