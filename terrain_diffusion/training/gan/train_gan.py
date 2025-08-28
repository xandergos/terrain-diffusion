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
from terrain_diffusion.training.datasets.datasets import LongDataset
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from terrain_diffusion.training.utils import *
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import SerializableEasyDict as EasyDict
from torchvision.transforms.v2.functional import gaussian_blur
from torchmetrics.image.fid import FrechetInceptionDistance

def get_optimizer(model, config):
    """Get optimizer based on config settings."""
    if config['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **config['kwargs'])
    else:
        raise ValueError(f"Invalid optimizer type: {config['type']}")
    return optimizer

def linear_warmup(start_value, end_value, current_step, total_steps):
    """
    Perform linear warmup from start_value to end_value.
    
    Args:
        start_value (float): Initial value at the start of warmup.
        end_value (float): Final value at the end of warmup.
        current_step (int): Current step in the warmup process.
        total_steps (int): Total number of warmup steps.
    
    Returns:
        float: Interpolated value during warmup.
    """
    if current_step >= total_steps:
        return end_value
    return start_value + (end_value - start_value) * (current_step / total_steps)

def calculate_fid(generator, val_dataset, config, device, n_samples=50000):
    """
    Calculate FID score between generated samples and validation dataset using torchmetrics.
    
    Args:
        generator: The generator model
        val_dataset: Validation dataset
        config: Training configuration
        device: Device to run calculation on
        n_samples: Number of samples to generate
    """
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Constants for denormalization and renormalization
    MEAN = -2607
    STD = 2435
    MIN_ELEVATION = -10000
    MAX_ELEVATION = 9000
    ELEVATION_RANGE = MAX_ELEVATION - MIN_ELEVATION
    
    def process_images(images):
        # Denormalize from standard normal back to elevation values
        images = images * STD + MEAN
        # Clip to valid elevation range
        images = torch.clamp(images, MIN_ELEVATION, MAX_ELEVATION)
        # Normalize to [0, 255] for FID calculation
        images = ((images - MIN_ELEVATION) * 255 / ELEVATION_RANGE).to(torch.uint8)
        # FID expects 3 channels
        images = images.repeat(1, 3, 1, 1)
        return images
    
    pbar = tqdm(total=n_samples*2, desc="Calculating FID")
    
    # Process real samples
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    for i, batch in enumerate(val_loader):
        if i * 64 >= n_samples:
            break
        images = batch['image'].to(device)
        images = process_images(images[:, :1])
        fid.update(images, real=True)
        pbar.update(images.shape[0])
        
    # Generate and process fake samples
    generator.eval()
    with torch.no_grad():
        for i in range(0, n_samples, 64):
            batch_size = min(64, n_samples - i)
            z = torch.randn(batch_size, config['generator']['latent_channels'],
                          config['training']['latent_size'], 
                          config['training']['latent_size'],
                          device=device)
            fake_images = generator(z)[:, :1]
            
            # Crop to match real image size if needed
            if fake_images.shape[-2:] != val_dataset[0]['image'].shape[-2:]:
                h_diff = fake_images.shape[-2] - val_dataset[0]['image'].shape[-2]
                w_diff = fake_images.shape[-1] - val_dataset[0]['image'].shape[-1]
                # Randomly select crop position
                h_start = torch.randint(0, h_diff + 1, (1,)).item()
                w_start = torch.randint(0, w_diff + 1, (1,)).item()
                fake_images = fake_images[:, :,
                                       h_start:h_start + val_dataset[0]['image'].shape[-2],
                                       w_start:w_start + val_dataset[0]['image'].shape[-1]]
            
            fake_images = process_images(fake_images)
            fid.update(fake_images, real=False)
            pbar.update(batch_size)
    return float(fid.compute())

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
    
    if os.path.exists(f"{config['logging']['save_dir']}/latest_checkpoint") and not ckpt_path:
        print("The save_dir directory already exists. Would you like to resume training from the latest checkpoint? (y/n)")
        resp = input().strip().lower()
        if resp == "y":
            ckpt_path = f"{config['logging']['save_dir']}/latest_checkpoint"
        elif resp == "n":
            print("Beginning new training run...")
        else:
            print("Unexpected input. Exiting...")
            return
        
    if debug_run:
        config['wandb']['mode'] = 'disabled'
    # Auto-resume W&B run from checkpoint metadata if available (unless explicitly provided)
    if ckpt_path and not resume_id and not debug_run:
        try:
            with open(os.path.join(ckpt_path, 'wandb_run.json'), 'r') as f:
                run_meta = json.load(f)
            if 'id' in run_meta and run_meta['id']:
                config['wandb']['id'] = run_meta['id']
                config['wandb']['resume'] = 'must'
        except Exception:
            pass
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
    val_dataset = LongDataset(resolved['val_dataset'], length=50000)
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
        
    print(state)

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
        generator.save_config(os.path.join(base_folder_path + '_checkpoint', f'model_config'))
        # Persist W&B run id for seamless resumption
        try:
            with open(os.path.join(base_folder_path + '_checkpoint', 'wandb_run.json'), 'w') as f:
                json.dump({'id': wandb.run.id if wandb.run else None}, f)
        except Exception:
            pass

    printed_size = False

    # Training loop
    train_iter = iter(train_dataloader)
    burnin_steps = config['training'].get('burnin_steps', 0)
    initial_r_gamma = config['training'].get('r_gamma', 0) * config['training'].get('r_warmup_factor', 10)
    final_r_gamma = config['training'].get('r_gamma', 0)
    
    # Warmup parameters for beta_2 in optimizers
    initial_beta_2 = 1 - 10 * (1 - config['g_optimizer']['kwargs']['betas'][1])
    final_beta_2 = config['g_optimizer']['kwargs']['betas'][1]
    
    print("Warming beta_2 from", initial_beta_2, "to", final_beta_2)

    while state['epoch'] < config['training']['epochs']:
        stats_hist = {'g_loss': [], 'd_loss': [], 'kl_loss': [], 'range_loss': [], 'r_loss': []}
        progress_bar = tqdm(train_iter, desc=f"Epoch {state['epoch']}", 
                          total=config['training']['epoch_steps'])
        
        while progress_bar.n < config['training']['epoch_steps']:
            # Warmup r_gamma and beta_2 during burnin steps
            if state['step'] < burnin_steps:
                current_r_gamma = linear_warmup(
                    initial_r_gamma, 
                    final_r_gamma, 
                    state['step'], 
                    burnin_steps
                )
                current_beta_2 = linear_warmup(
                    initial_beta_2, 
                    final_beta_2, 
                    state['step'], 
                    burnin_steps
                )
                
                # Update beta_2 for both optimizers
                for optimizer in [g_optimizer, d_optimizer]:
                    for group in optimizer.param_groups:
                        group['betas'] = (group['betas'][0], current_beta_2)
            else:
                current_r_gamma = final_r_gamma

            # Train discriminator
            with accelerator.accumulate(discriminator):
                batch = next(train_iter)
                real_images = batch['image']
                batch_size = real_images.shape[0]
                
                
                with accelerator.autocast():
                    discriminator.train()
                    
                    with torch.no_grad():
                        z = torch.randn(batch_size, config['generator']['latent_channels'],
                                    config['training']['latent_size'], config['training']['latent_size'],
                                    device=accelerator.device)
                        fake_images = generator(z)
                    
                    if config['training'].get('blur_sigma', 0) > 0:
                        sigma = int(config['training']['blur_sigma'])
                        fake_images = gaussian_blur(fake_images, (1+2*sigma, 1+2*sigma), sigma)
                        
                    if not printed_size:
                        print("Fake image size:", fake_images.shape)
                        print("Real image size:", real_images.shape)
                        printed_size = True
                        
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
                    
                    all_images = torch.cat([real_images, fake_images.detach()], dim=0).detach().requires_grad_(True)
                    pred = discriminator(all_images)
                    real_pred = pred[:batch_size]
                    fake_pred = pred[batch_size:]
                    
                    d_loss = torch.nn.functional.softplus(fake_pred - real_pred).mean()
                    
                    if config['training'].get('r_gamma', 0) > 0 and state['step'] % config['training'].get('r_interval', 16) == 0:
                        # Compute gradient penalty separately before the main backward pass
                        grad_real = torch.autograd.grad(
                            outputs=pred.sum(), inputs=all_images,
                            create_graph=True)[0]
                        r_reg = current_r_gamma * 0.5 * grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
                        # Combine losses after gradient computation
                        total_d_loss = d_loss + r_reg
                    else:
                        r_reg = 0
                        total_d_loss = d_loss

                d_optimizer.zero_grad()
                accelerator.backward(total_d_loss)
                if accelerator.sync_gradients:
                    discriminator_grad_norm = accelerator.clip_grad_norm_(discriminator.parameters(), 100.0)
                d_optimizer.step()

            # Train generator
            with accelerator.accumulate(generator):
                with accelerator.autocast():
                    discriminator.eval()
                    z = torch.randn(batch_size, config['generator']['latent_channels'],
                                config['training']['latent_size'], config['training']['latent_size'],
                                device=accelerator.device)
                    fake_images = generator(z)
                    
                    fake_pred = discriminator(fake_images)
                    g_loss = torch.nn.functional.softplus(real_pred.detach() - fake_pred).mean()
                    
                    real_mean = real_images.mean(dim=(0, 2, 3))
                    real_std = real_images.std(dim=(0, 2, 3))
    
                    mean = fake_images.mean(dim=(0, 2, 3))
                    std = fake_images.std(dim=(0, 2, 3))
                    kl_loss = (
                        torch.log(real_std / (std + 1e-8)) +     # log std ratio
                        (std**2 + (mean - real_mean)**2) / (2 * (real_std**2 + 1e-8)) - 0.5  # variance and mean difference
                    ).mean()  # Take mean across channels
                    total_g_loss = g_loss + kl_loss * config['training'].get('kl_weight', 0.0)
                    
                    # Range loss to encourage values between -2 and 3.2
                    below_min = torch.nn.functional.relu(-2 - fake_images)
                    above_max = torch.nn.functional.relu(fake_images - 3.2)
                    range_loss = (below_min**2 + above_max**2).mean()
                    total_g_loss = total_g_loss + range_loss * config['training'].get('range_weight', 1.0)
                
                g_optimizer.zero_grad()
                accelerator.backward(total_g_loss)
                if accelerator.sync_gradients:
                    generator_grad_norm = accelerator.clip_grad_norm_(generator.parameters(), 10.0)
                g_optimizer.step()
                
            if accelerator.is_main_process:
                ema.update()  

            stats_hist['d_loss'].append(d_loss.item())
            stats_hist['g_loss'].append(g_loss.item())
            stats_hist['kl_loss'].append(kl_loss.item())
            stats_hist['range_loss'].append(range_loss.item())
            stats_hist['r_loss'].append((r_reg.item() / current_r_gamma) if current_r_gamma > 0 else 0)
            state['seen'] += batch_size
            state['step'] += 1
            
            lr_warmup = linear_warmup(
                config['training'].get('lr_warmup_factor', 1.0), 
                1.0, 
                state['step'], 
                config['training'].get('burnin_steps', 1)
            )
            lr = lr_scheduler.get(state.seen) * lr_warmup
            for g in g_optimizer.param_groups:
                g['lr'] = lr
            for g in d_optimizer.param_groups:
                g['lr'] = lr * config['training'].get('disc_lr_mult', 1.0)
            
            progress_bar.set_postfix({
                'd_loss': f"{np.mean(stats_hist['d_loss'][-10:]):.4f}",
                'g_loss': f"{np.mean(stats_hist['g_loss'][-10:]):.4f}",
                'kl_loss': f"{np.mean(stats_hist['kl_loss'][-10:]):.4f}",
                'range_loss': f"{np.mean(stats_hist['range_loss'][-10:]):.4f}",
                "r_loss": f"{r_reg / current_r_gamma:.4f}" if current_r_gamma > 0 else 0,
                'lr': lr,
                'd_grad_norm': f"{discriminator_grad_norm:.4f}",
                'g_grad_norm': f"{generator_grad_norm:.4f}"
            })
            progress_bar.update(1)

        progress_bar.close()
            
        # Evaluation
        if (state['epoch'] % config['training'].get('eval_epochs', 5) == 0 and 
            accelerator.is_main_process and not debug_run):
            generator.eval()
            with temporary_ema_to_model(ema.ema_models[0]):
                fid = calculate_fid(generator, val_dataset, config, accelerator.device)
            print(f"FID: {fid}")
            wandb.log({'eval/fid': fid}, step=state['epoch'])
            generator.train()
            
        # Logging
        state.epoch += 1
        if accelerator.is_main_process:
            log_values = {
                'train/d_loss': np.mean(stats_hist['d_loss']),
                'train/g_loss': np.mean(stats_hist['g_loss']),
                'train/kl_loss': np.mean(stats_hist['kl_loss']),
                'train/r_loss': np.mean(stats_hist['r_loss']),
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

if __name__ == '__main__':
    main()
