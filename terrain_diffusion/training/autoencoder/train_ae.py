from collections import defaultdict
import json
import click
from datetime import datetime
from heavyball import ForeachAdamW, ForeachSOAP
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.data.laplacian_encoder import *
from terrain_diffusion.training.registry import build_registry
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
import lpips
from terrain_diffusion.training.unet import EDMAutoencoder
from terrain_diffusion.training.utils import *
from terrain_diffusion.training.utils import SerializableEasyDict as EasyDict

def get_optimizer(model, config):
    """Get optimizer based on config settings."""
    if config['optimizer']['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **config['optimizer']['kwargs'])
    elif config['optimizer']['type'] == 'heavyball-adam':
        optimizer = ForeachAdamW(model.parameters(), **config['optimizer']['kwargs'])
    elif config['optimizer']['type'] == 'soap':
        optimizer = ForeachSOAP(model.parameters(), **config['optimizer']['kwargs'])
    else:
        raise ValueError(f"Unknown optimizer type: {config['optimizer']['type']}")
    return optimizer

def variance_adjusted_loss(reconstruction, reference, eps=0.25):
    ref_min = torch.amin(reference, dim=(1, 2, 3), keepdim=True)
    ref_max = torch.amax(reference, dim=(1, 2, 3), keepdim=True)
    
    ref_range = torch.maximum(ref_max - ref_min, torch.tensor(eps))
    ref_center = (ref_min + ref_max) / 2
    
    normalized_ref = ((reference - ref_center) / ref_range * 2)
    normalized_rec = ((reconstruction - ref_center) / ref_range * 2)
    
    return (normalized_rec - normalized_ref).abs().mean()

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True, help="Path to the configuration file")
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False, help="Path to a checkpoint to resume training from")
@click.option("--model-ckpt", "model_ckpt_path", type=click.Path(exists=True), required=False, help="Path to a HuggingFace model to initialize weights from")
@click.option("--debug-run", "debug_run", is_flag=True, default=False, help="Run in debug mode which disables wandb and all file saving")
@click.option("--resume", "resume_id", type=str, required=False, help="Wandb run ID to resume")
@click.option("--override", "-o", multiple=True, help="Override config values (format: key.subkey=value)")
@click.pass_context
def main(ctx, config_path, ckpt_path, model_ckpt_path, debug_run, resume_id, override):
    build_registry()
    
    config = Config().from_disk(config_path) if config_path else None
    
    # Handle both explicit overrides and wandb sweep parameters
    all_overrides = list(override)
    for param in ctx.args:
        if param.startswith('--'):
            key, value = param.lstrip('-').split('=', 1)
            all_overrides.append(f"{key}={value}")
    
    # Apply all config overrides
    for o in all_overrides:
        key_path, value = o.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        set_nested_value(config, key_path, value, o)

    if debug_run:
        config['wandb']['mode'] = 'disabled'
    if resume_id:
        config['wandb']['id'] = resume_id
        config['wandb']['resume'] = 'must'
    wandb.init(
        **config['wandb'],
        config=config
    )
    print("Run ID:", wandb.run.id)
    
    resolved = registry.resolve(config, validate=False)

    # Load anything that needs to be used for training.
    model = resolved['model']
    assert isinstance(model, EDMAutoencoder), "Currently only EDMAutoencoder is supported for autoencoder training."
    lr_scheduler = resolved['lr_sched']
    
    train_dataset = resolved['train_dataset']
    val_dataset = resolved['val_dataset']
    
    optimizer = get_optimizer(model, config)
    resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
    ema = PostHocEMA(model, **resolved['ema'])
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    train_dataloader = DataLoader(LongDataset(train_dataset, shuffle=True), batch_size=config['training']['train_batch_size'],
                            **resolved['dataloader_kwargs'], drop_last=True)
    val_dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), batch_size=config['training']['train_batch_size'],
                                **resolved['dataloader_kwargs'], drop_last=True)
    perceptual_loss = lpips.LPIPS(net='alex', spatial=True)
    
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
    
    ema = ema.to(accelerator.device)
    model, train_dataloader, optimizer, val_dataloader = accelerator.prepare(model, train_dataloader, optimizer, val_dataloader)
    perceptual_loss = accelerator.prepare(perceptual_loss)
    accelerator.register_for_checkpointing(state)
    accelerator.register_for_checkpointing(ema)
    
    # Load from checkpoint if needed
    if ckpt_path:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        accelerator.load_state(ckpt_path)
    
    if accelerator.is_main_process:
        if debug_run:
            config['wandb']['mode'] = 'disabled'
        wandb.init(
            **config['wandb'],
            config=config
        )

    def percep_loss_fn(reconstruction, reference):
        ref_min = torch.amin(reference, dim=(1, 2, 3), keepdim=True)
        ref_max = torch.amax(reference, dim=(1, 2, 3), keepdim=True)
        eps = 0.1
        
        ref_range = torch.maximum((ref_max - ref_min) * 1.1, torch.tensor(eps))
        ref_center = (ref_min + ref_max) / 2
        
        normalized_ref = ((reference - ref_center) / ref_range * 2)
        normalized_rec = ((reconstruction - ref_center) / ref_range * 2)
        normalized_rec = normalized_rec.clamp(-1, 1)
        
        rec_perceptual_loss = perceptual_loss(normalized_ref.repeat(1, 3, 1, 1), normalized_rec.repeat(1, 3, 1, 1))
        return rec_perceptual_loss.mean()

    def validate(repeats, dataloader, pbar_title):
        validation_stats = {
            'loss': [], 
            'kl_loss': [], 
            'rec_direct_loss': [], 
            'rec_percep_loss': []
        }
        pbar = tqdm(total=repeats * len(val_dataset), desc=pbar_title)
        val_dataloader_iter = iter(dataloader)
        
        while pbar.n < pbar.total:
            batch = next(val_dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')

            sigma_data = config['training']['sigma_data']
            
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                scaled_clean_images = images / sigma_data
                if cond_img is not None:
                    scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
                
                # Encode and decode
                z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
                z = model.postencode(z_means, z_logvars)
                decoded_x = model.decode(z)

                # Calculate losses
                rec_direct_loss = variance_adjusted_loss(decoded_x, scaled_clean_images)
                
                rec_percep_loss = percep_loss_fn(decoded_x, scaled_clean_images)
                kl_loss = -0.5 * torch.mean(1 + z_logvars - z_means**2 - z_logvars.exp())
                
                # Combine losses with weights
                loss = (config['training']['direct_weight'] * rec_direct_loss + 
                       config['training']['percep_weight'] * rec_percep_loss + 
                       config['training']['kl_weight'] * kl_loss)

            # Record statistics
            validation_stats['loss'].append(loss.item())
            validation_stats['kl_loss'].append(kl_loss.item())
            validation_stats['rec_direct_loss'].append(rec_direct_loss.item())
            validation_stats['rec_percep_loss'].append(rec_percep_loss.item())
            
            pbar.update(images.shape[0])
            pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in validation_stats.items()})
        
        # Return average losses
        return {k: np.mean(v) for k, v in validation_stats.items()}
    
    def save_checkpoint(base_folder_path, overwrite=False):
        if os.path.exists(base_folder_path + '_checkpoint') and not overwrite:
            strtime = datetime.now().strftime("_%Y%m%d_%H%M%S")
            base_folder_path = f"{base_folder_path}{strtime}"
        elif os.path.exists(base_folder_path + '_checkpoint'):
            safe_rmtree(base_folder_path + '_checkpoint')
        os.makedirs(base_folder_path + '_checkpoint', exist_ok=False)
        
        accelerator.save_state(base_folder_path + '_checkpoint')
        torch.save(ema.state_dict(), os.path.join(base_folder_path + '_checkpoint', 'phema.pt'))
        
        # Save full train config and model config
        with open(os.path.join(base_folder_path + '_checkpoint', f'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        model.save_config(os.path.join(base_folder_path + '_checkpoint', f'model_config'))

    dataloader_iter = iter(train_dataloader)
    grad_norm = 0.0
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
                
                z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
                z = model.postencode(z_means, z_logvars)
                decoded_x = model.decode(z)
                
                # Scaling MSE loss so large scale images don't dominate
                rec_direct_loss = variance_adjusted_loss(decoded_x, scaled_clean_images)
                
                rec_percep_loss = percep_loss_fn(decoded_x, scaled_clean_images)
                
                kl_loss = -0.5 * torch.mean(1 + z_logvars - z_means**2 - z_logvars.exp())
                
                percep_weight = config['training']['percep_weight']
                direct_weight = config['training']['direct_weight']
                kl_weight = config['training']['kl_weight'] * min(1, state.step / config['training'].get('warmup_steps', 1))
                
                loss = direct_weight * rec_direct_loss + percep_weight * rec_percep_loss + kl_weight * kl_loss
                
                state.seen += images.shape[0]
                state.step += 1
                lr = lr_scheduler.get(state.seen)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                    
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=config['training'].get('gradient_clip_val', 100.0)).item()
                optimizer.step()

            if accelerator.is_main_process:
                ema.update()

            stats_hist['loss'].append(loss.item())
            stats_hist['kl_loss'].append(kl_loss.item())
            stats_hist['rec_direct_loss'].append(rec_direct_loss.item())
            stats_hist['rec_percep_loss'].append(rec_percep_loss.item())
            stats_hist['grad_norm'].append(grad_norm)
            progress_bar.set_postfix({'loss': np.mean(stats_hist['loss']), 
                                    'rec_direct_loss': np.mean(stats_hist['rec_direct_loss']),
                                    'rec_percep_loss': np.mean(stats_hist['rec_percep_loss']),
                                    'kl_loss': np.mean(stats_hist['kl_loss']),
                                    "lr": lr,
                                    "grad_norm": grad_norm})
            progress_bar.update(1)
            
        progress_bar.close()

        val_losses = None
        eval_losses = None
        if config['evaluation']['validate_epochs'] > 0 and (state.epoch + 1) % config['evaluation']['validate_epochs'] == 0:
            if config['evaluation'].get('val_ema_idx', -1) >= 0 and config['evaluation']['val_ema_idx'] < len(ema.ema_models):
                with temporary_ema_to_model(ema.ema_models[config['evaluation']['val_ema_idx']]):
                    val_losses = validate(config['evaluation']['validation_repeats'], val_dataloader, "Validation Loss")
                    if config['evaluation'].get('training_eval', False):
                        eval_losses = validate(config['evaluation']['validation_repeats'], train_dataloader, "Eval Loss")
            else:
                if config['evaluation'].get('val_ema_idx', -1) >= 0:
                    warnings.warn(f"Invalid val_ema_idx: {config['evaluation']['val_ema_idx']}. "
                                  "Falling back to using the model's parameters.")
                val_losses = validate(config['evaluation']['validation_repeats'], val_dataloader, "Validation Loss")
                if config['evaluation'].get('training_eval', False):
                    eval_losses = validate(config['evaluation']['validation_repeats'], train_dataloader, "Eval Loss")
                    
        state.epoch += 1
        if accelerator.is_main_process:
            wandb_logs = {
                "loss": np.mean(stats_hist['loss']),
                "kl_loss": np.mean(stats_hist['kl_loss']),
                "rec_direct_loss": np.mean(stats_hist['rec_direct_loss']),
                "rec_percep_loss": np.mean(stats_hist['rec_percep_loss']),
                "lr": lr,
                "step": state.step,
                "epoch": state.epoch,
                "seen": state.seen
            }
            if val_losses:
                val_losses = {f"val_{k}": v for k, v in val_losses.items()}
                wandb_logs.update(val_losses)
            if eval_losses:
                eval_losses = {f"eval_{k}": v for k, v in eval_losses.items()}
                wandb_logs.update(eval_losses)
            wandb.log(wandb_logs, step=state.epoch, commit=True)
            if state.epoch % config['logging']['temp_save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")

if __name__ == '__main__':
    main()
