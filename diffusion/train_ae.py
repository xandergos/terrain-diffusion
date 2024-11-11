from collections import defaultdict
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
from diffusion.encoder import *
from diffusion.registry import build_registry
from diffusion.samplers.tiled import TiledSampler
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from diffusion.unet import DiffusionAutoencoder
from utils import SerializableEasyDict as EasyDict
from safetensors.torch import load_model
import lpips
import torch.nn.functional as F


    

@click.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False)
@click.option("--model-ckpt", "model_ckpt_path", type=click.Path(exists=True), required=False)
@click.option("--debug-run", "debug_run", is_flag=True, default=False)
@click.option("--reset-state", "reset_state", is_flag=True, default=False)
def main(config_path, ckpt_path, model_ckpt_path, debug_run, reset_state):
    build_registry()
    
    config = Config().from_disk(config_path) if config_path else None
    
    # Resolve this later
    sampler_config = config.get('sampler', None)
    if sampler_config:
        del config['sampler']
            
    resolved = registry.resolve(config, validate=False)

    # Load anything that needs to be used for training.
    model = resolved['model']
    lr_scheduler = resolved['lr_sched']
    train_dataset = resolved['train_dataset']
    optimizer = torch.optim.Adam(model.parameters(), betas=tuple(config['training'].get('adam_betas', (0.9, 0.999))))
    resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
    ema = PostHocEMA(model, **resolved['ema'])
    if config['training']['disc_weight'] > 0:
        discriminator = resolved['discriminator']
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), betas=tuple(config['training'].get('disc_adam_betas', (0.9, 0.999))))
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    dataloader = DataLoader(LongDataset(train_dataset), batch_size=config['training']['train_batch_size'],
                            **resolved['dataloader_kwargs'])
    perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=True)
    
    if model_ckpt_path:
        temp_model_statedict = type(model).from_pretrained(model_ckpt_path).state_dict()
        filtered_state_dict = {}
        for name, param in temp_model_statedict.items():
            if name in model.state_dict():
                if param.shape == model.state_dict()[name].shape:
                    filtered_state_dict[name] = param
                else:
                    print(f"Skipping parameter {name} due to shape mismatch. "
                          f"Loaded shape: {param.shape}, "
                          f"Model shape: {model.state_dict()[name].shape}")
            else:
                print(f"Skipping parameter {name} as it doesn't exist in the current model.")
        temp_model_statedict = filtered_state_dict
        try:
            model.load_state_dict(temp_model_statedict)
        except Exception as e:
            print("Loading model with strict=False")
            model.load_state_dict(temp_model_statedict, strict=False)
        del temp_model_statedict
        
    print(f"Training model with {model.count_parameters()} parameters.")

    # Setup accelerate
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=resolved['training']['gradient_accumulation_steps'],
        log_with=None
    )
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
    perceptual_loss_fn = accelerator.prepare(perceptual_loss_fn)
    if config['training']['disc_weight'] > 0:
        discriminator, optimizer_d = accelerator.prepare(discriminator, optimizer_d)
    accelerator.register_for_checkpointing(state)
    accelerator.register_for_checkpointing(ema)
    
    # Load from checkpoint if needed
    if ckpt_path:
        accelerator.load_state(ckpt_path)
    if reset_state:
        state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    
    
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

    dataloader_iter = iter(dataloader)
    while state.epoch < config['training']['epochs']:
        stats_hist = defaultdict(list)
        progress_bar = tqdm(dataloader_iter, desc=f"Epoch {state.epoch}", total=config['training']['epoch_steps'])
        while progress_bar.n < config['training']['epoch_steps']:
            batch = next(dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')

            sigma_data = config['training']['sigma_data']
            
            with accelerator.autocast(), accelerator.accumulate(model):
                scaled_clean_images = images / sigma_data
                if cond_img is not None:
                    scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
                
                if state.step % 2 != 0 and config['training']['disc_weight'] > 0:
                    with torch.no_grad():
                        z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
                        z = model.postencode(z_means, z_logvars)
                        decoded_x = model.decode(z)
                else:
                    z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
                    z = model.postencode(z_means, z_logvars)
                    decoded_x = model.decode(z)
                
                def percep_loss_fn(reconstruction, reference):
                    ref_min = torch.amin(reference, dim=(1, 2, 3), keepdim=True)
                    ref_max = torch.amax(reference, dim=(1, 2, 3), keepdim=True)
                    eps = 1e-4
                    
                    normalized_ref = ((reference - ref_min) / (ref_max - ref_min + eps) * 2 - 1) * 0.9
                    normalized_rec = ((reconstruction - ref_min) / (ref_max - ref_min + eps) * 2 - 1) * 0.9
                    normalized_rec = normalized_rec.clamp(-1, 1)
                    
                    rec_perceptual_loss = perceptual_loss_fn(normalized_ref.repeat(1, 3, 1, 1), normalized_rec.repeat(1, 3, 1, 1))
                    return rec_perceptual_loss.mean()

                if state.step % 2 != 0 and config['training']['disc_weight'] > 0:  # Update discriminator
                    if config['training']['lambda_gp'] > 0:
                        scaled_clean_images.requires_grad_(True)
                    real_output = torch.mean(discriminator(scaled_clean_images).view(scaled_clean_images.size(0), -1), dim=1)
                    fake_output = torch.mean(discriminator(decoded_x.detach()).view(decoded_x.size(0), -1), dim=1)
                    
                    # Compute main gradients
                    d_loss = torch.mean(F.relu(1 - real_output) + F.relu(1 + fake_output))
                    optimizer_d.zero_grad()
                    accelerator.backward(d_loss, retain_graph=True)
                    
                    # R1 gradient penalty for real inputs
                    if config['training']['lambda_gp'] > 0:
                        batch_size = scaled_clean_images.size(0)
                        gradients = torch.autograd.grad(outputs=real_output.sum(), inputs=scaled_clean_images,
                                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                        gradient_penalty = gradients.pow(2).view(batch_size, -1).sum(1)
                        lambda_gp = config['training']['lambda_gp']
                        accelerator.backward(lambda_gp * gradient_penalty.mean())
                    
                    optimizer_d.step()
                else:
                    rec_mse_loss = (1 / sigma_data ** 2) * (decoded_x - scaled_clean_images) ** 2
                    rec_mse_loss = rec_mse_loss.mean()
                    mse_weight = config['training']['mse_weight']
                    rec_percep_loss = percep_loss_fn(decoded_x, scaled_clean_images)
                    percep_weight = config['training']['percep_weight']
                    kl_loss = -0.5 * torch.mean(1 + z_logvars - z_means**2 - z_logvars.exp())
                    kl_weight = config['training']['kl_weight']
                    
                    if config['training']['disc_weight'] > 0:
                        disc_output = torch.mean(discriminator(decoded_x).view(decoded_x.size(0), -1), dim=1)
                        g_loss = torch.mean(F.relu(1 - disc_output))
                        g_weight = config['training']['disc_weight']
                    else:
                        g_loss = 0
                        g_weight = 0
                    
                    loss = mse_weight * rec_mse_loss + percep_weight * rec_percep_loss + kl_weight * kl_loss + g_weight * g_loss
                    
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training'].get('gradient_clip_val', 1.0))
                    optimizer.step()

                state.seen += images.shape[0]
                state.step += 1
                lr = lr_scheduler.get(state.seen)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                if config['training']['disc_weight'] > 0:
                    for g in optimizer_d.param_groups:
                        g['lr'] = lr * config['training']['disc_lr_mult']

            if accelerator.is_main_process:
                ema.update()

            if state.step % 2 == 0 or config['training']['disc_weight'] == 0:
                stats_hist['loss'].append(loss.item())
                stats_hist['kl_loss'].append(kl_loss.item())
                stats_hist['rec_mse_loss'].append(rec_mse_loss.item())
                stats_hist['rec_percep_loss'].append(rec_percep_loss.item())
                stats_hist['d_loss'].append(d_loss.item() if config['training']['disc_weight'] > 0 else float('nan'))
                stats_hist['g_loss'].append(g_loss.item() if config['training']['disc_weight'] > 0 else float('nan'))
                progress_bar.set_postfix({'loss': np.mean(stats_hist['loss']), 
                                        'rec_mse_loss': np.mean(stats_hist['rec_mse_loss']),
                                        'rec_percep_loss': np.mean(stats_hist['rec_percep_loss']),
                                        'kl_loss': np.mean(stats_hist['kl_loss']),
                                        'd_loss': np.mean(stats_hist['d_loss']),
                                        'g_loss': np.mean(stats_hist['g_loss']),
                                        "lr": lr})
            progress_bar.update(1)

        state.epoch += 1
        if accelerator.is_main_process:
            wandb.log({
                "loss": np.mean(stats_hist['loss']),
                "kl_loss": np.mean(stats_hist['kl_loss']),
                "rec_mse_loss": np.mean(stats_hist['rec_mse_loss']),
                "rec_percep_loss": np.mean(stats_hist['rec_percep_loss']),
                "lr": lr,
                "step": state.step,
                "epoch": state.epoch,
                "seen": state.seen
            }, step=state.epoch)
            if state.epoch % config['logging']['temp_save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")

if __name__ == '__main__':
    main()
