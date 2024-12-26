from contextlib import contextmanager
import json
import click
from datetime import datetime
import hamiltorch.samplers
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import ETOPODataset, LongDataset
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from terrain_diffusion.training.gan.discriminator import MPDiscriminator
from terrain_diffusion.training.gan.generator import MPGenerator
from terrain_diffusion.training.utils import *
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import SerializableEasyDict as EasyDict
from torchvision.transforms.v2.functional import gaussian_blur
import torch.distributions as distributions
import hamiltorch

def get_optimizer(model, config):
    """Get optimizer based on config settings."""
    if config['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **config['kwargs'])
    else:
        raise ValueError(f"Invalid optimizer type: {config['type']}")
    return optimizer

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False)
@click.option("--debug-run", is_flag=True, default=False)
@click.option("--resume", "resume_id", type=str, required=False)
@click.pass_context
def main(ctx, config_path, ckpt_path, debug_run, resume_id):
    """Main training function."""
    build_registry()
    config = Config().from_disk(config_path)
    
    if debug_run:
        config['wandb']['mode'] = 'disabled'
    if resume_id:
        config['wandb']['id'] = resume_id
        config['wandb']['resume'] = 'must'
    wandb.init(**config['wandb'], config=config)
    
    # Resolve configuration using registry
    resolved = registry.resolve(config, validate=False)
    
    # Load components from resolved config
    generator = resolved['generator']
    discriminator = resolved['discriminator']
    train_dataset = resolved['train_dataset']
    lr_scheduler = resolved['lr_sched']
    resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
    ema = PostHocEMA(generator, **resolved['ema'])

    # Setup optimizers
    g_optimizer = get_optimizer(generator, config['g_optimizer'])
    d_optimizer = get_optimizer(discriminator, config['d_optimizer'])
    
    train_dataloader = DataLoader(
        LongDataset(train_dataset, shuffle=True),
        batch_size=config['training']['batch_size'],
        **resolved['dataloader_kwargs']
    )

    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=resolved['training']['gradient_accumulation_steps']
    )
    ema = ema.to(accelerator.device)
    
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    generator, discriminator, train_dataloader, g_optimizer, d_optimizer = \
        accelerator.prepare(generator, discriminator, train_dataloader, g_optimizer, d_optimizer)
    accelerator.register_for_checkpointing(state)
    accelerator.register_for_checkpointing(ema)

    if ckpt_path:
        accelerator.load_state(ckpt_path)

    def save_checkpoint(base_folder_path, overwrite=False):
        """Save training checkpoint."""
        if os.path.exists(base_folder_path + '_checkpoint') and not overwrite:
            strtime = datetime.now().strftime("_%Y%m%d_%H%M%S")
            base_folder_path = f"{base_folder_path}{strtime}"
        elif os.path.exists(base_folder_path + '_checkpoint'):
            safe_rmtree(base_folder_path + '_checkpoint')
        os.makedirs(base_folder_path + '_checkpoint', exist_ok=False)
        
        accelerator.save_state(base_folder_path + '_checkpoint')
        torch.save(ema.state_dict(), os.path.join(base_folder_path + '_checkpoint', 'phema.pt'))
        
        with open(os.path.join(base_folder_path + '_checkpoint', 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    # Training loop
    train_iter = iter(train_dataloader)
    while state['epoch'] < config['training']['epochs']:
        stats_hist = {'g_loss': [], 'd_loss': [], 'kl_loss': []}
        progress_bar = tqdm(train_iter, desc=f"Epoch {state['epoch']}", 
                          total=config['training']['epoch_steps'])
        
        while progress_bar.n < config['training']['epoch_steps']:
            # Train discriminator
            with accelerator.accumulate(discriminator):
                batch = next(train_iter)
                real_images = batch['image']
                if config['training'].get('r1_gamma', 0) > 0 and state['step'] % config['training'].get('r1_interval', 16) == 0:
                    real_images.requires_grad_(True)
                batch_size = real_images.shape[0]
                
                z = torch.randn(batch_size, config['generator']['latent_channels'],
                              config['training']['latent_size'], config['training']['latent_size'],
                              device=accelerator.device)
                
                with accelerator.autocast():
                    discriminator.train()
                    fake_images = generator(z)
                    if config['training'].get('blur_sigma', 0) > 0:
                        sigma = int(config['training']['blur_sigma'])
                        fake_images = gaussian_blur(fake_images, (1+2*sigma, 1+2*sigma), sigma)
                        
                    # crop fake images to match real image size
                    h_diff = fake_images.shape[-2] - real_images.shape[-2]
                    w_diff = fake_images.shape[-1] - real_images.shape[-1]
                    h_starts = torch.randint(0, h_diff + 1, (batch_size,), device=fake_images.device)
                    w_starts = torch.randint(0, w_diff + 1, (batch_size,), device=fake_images.device)
                    fake_images = torch.stack([
                        fake_images[i, :, h_starts[i]:h_starts[i]+real_images.shape[-2], 
                                w_starts[i]:w_starts[i]+real_images.shape[-1]]
                        for i in range(batch_size)
                    ])
                    
                    all_images = torch.cat([real_images, fake_images.detach()], dim=0)
                    pred = discriminator(all_images)
                    real_pred = pred[:batch_size]
                    fake_pred = pred[batch_size:]
                    
                    d_loss = (torch.nn.functional.softplus(-real_pred) + torch.nn.functional.softplus(fake_pred)).mean()
                    
                    if config['training'].get('r1_gamma', 0) > 0 and state['step'] % config['training'].get('r1_interval', 16) == 0:
                        grad_real = torch.autograd.grad(
                            outputs=real_pred.sum(), inputs=real_images,
                            create_graph=True, only_inputs=True)[0]
                        r1_reg = config['training']['r1_gamma'] * 0.5 * grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
                        d_loss = d_loss + r1_reg

                d_optimizer.zero_grad()
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    discriminator_grad_norm = accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)
                d_optimizer.step()

            # Train generator
            with accelerator.accumulate(generator):
                with accelerator.autocast():
                    discriminator.eval()
                    fake_pred = discriminator(fake_images)
                    g_loss = torch.nn.functional.softplus(-fake_pred).mean()
                
                    # KL Divergence loss to encourage fake_pred to follow a standard normal distribution
                    fake_pred_flat = fake_pred.flatten()
                    pred_mean = fake_pred_flat.mean()
                    pred_std = fake_pred_flat.std()
                    
                    # KL Divergence for Normal distribution: 
                    # KL(N(μ,σ²) || N(0,1)) = log(1/σ) + (σ² + μ²)/2 - 1/2
                    kl_loss = -torch.log(pred_std) + (pred_std**2 + pred_mean**2)/2 - 0.5
                    
                    full_g_loss = g_loss + kl_loss * config['training'].get('kl_weight', 0.0)
                
                g_optimizer.zero_grad()
                accelerator.backward(full_g_loss)
                if accelerator.sync_gradients:
                    generator_grad_norm = accelerator.clip_grad_norm_(generator.parameters(), 1.0)
                g_optimizer.step()
                
            if accelerator.is_main_process:
                ema.update()  

            stats_hist['d_loss'].append(d_loss.item())
            stats_hist['g_loss'].append(g_loss.item())
            stats_hist['kl_loss'].append(kl_loss.item())
            
            state['seen'] += batch_size
            state['step'] += 1
            
            lr = lr_scheduler.get(state.seen)
            for g in g_optimizer.param_groups:
                g['lr'] = lr
            for g in d_optimizer.param_groups:
                g['lr'] = lr * config['training'].get('disc_lr_mult', 1.0)
            
            progress_bar.set_postfix({
                'd_loss': f"{np.mean(stats_hist['d_loss'][-10:]):.4f}",
                'g_loss': f"{np.mean(stats_hist['g_loss'][-10:]):.4f}",
                'kl_loss': f"{np.mean(stats_hist['kl_loss'][-10:]):.4f}",
                'lr': lr,
                'd_grad_norm': f"{discriminator_grad_norm:.4f}",
                'g_grad_norm': f"{generator_grad_norm:.4f}"
            })
            progress_bar.update(1)

        progress_bar.close()
            
        # Logging
        if accelerator.is_main_process:
            log_values = {
                'train/d_loss': np.mean(stats_hist['d_loss']),
                'train/g_loss': np.mean(stats_hist['g_loss']),
                'train/kl_loss': np.mean(stats_hist['kl_loss']),
                'epoch': state['epoch'],
                'step': state['step'],
                'seen': state['seen'],
                'lr': lr
            }
            wandb.log(log_values, step=state['epoch'])
            
            # Save checkpoints
            if state['epoch'] % config['logging']['temp_save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state['epoch'] % config['logging']['save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/{state['seen']//1000}kimg")
        
        state['epoch'] += 1

if __name__ == '__main__':
    main()
