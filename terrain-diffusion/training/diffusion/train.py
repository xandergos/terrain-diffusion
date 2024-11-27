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
from training.datasets.datasets import LongDataset, MultiDataset
from data.laplacian_encoder import *
from training.diffusion.registry import build_registry
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from training.utils import SerializableEasyDict as EasyDict
from schedulefree import AdamWScheduleFree
from training.diffusion.unet import EDMUnet2D

from PIL import Image
import warnings

def set_nested_value(config, key_path, value, original_override):
    """Set a value in nested config dict, warning if key path doesn't exist."""
    keys = key_path.split('.')
    current = config
    
    # Check if the full path exists before modifying
    try:
        for key in keys[:-1]:
            if key not in current:
                warnings.warn(f"Creating new config section '{key}' from override: {original_override}")
                current[key] = {}
            current = current[key]
        
        if keys[-1] not in current:
            warnings.warn(f"Creating new config value '{key_path}' from override: {original_override}")
        current[keys[-1]] = value
    except (KeyError, TypeError) as e:
        warnings.warn(f"Failed to apply override '{original_override}': {str(e)}")

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True, help="Path to the configuration file")
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False, help="Path to a checkpoint (folder) to resume training from")
@click.option("--model-ckpt", "model_ckpt_path", type=click.Path(exists=True), required=False, help="Path to a HuggingFace model to initialize weights from")
@click.option("--debug-run", "debug_run", is_flag=True, default=False, help="Run in debug mode which disables wandb and all file saving")
@click.option("--resume", "resume_id", type=str, required=False, help="Wandb run ID to resume")
@click.option("--override", "-o", multiple=True, help="Override config values (format: key.subkey=value)")
@click.pass_context
def main(ctx, config_path, ckpt_path, model_ckpt_path, debug_run, resume_id, override):
    build_registry()
    
    config = Config().from_disk(config_path) if config_path else None
    
    # Handle both explicit overrides (-o flag) and wandb sweep parameters
    all_overrides = list(override)
    
    # Process any additional wandb sweep parameters
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
    assert isinstance(model, EDMUnet2D), "Currently only supports EDMUnet2D for diffusion training."
    lr_scheduler = resolved['lr_sched']
    dataset = resolved['dataset']
    if not isinstance(dataset, MultiDataset):
        dataset = MultiDataset(dataset)  # Has no effect but can now use .split()
    train_dataset, val_dataset = dataset.split(config['training']['val_pct'], generator=torch.Generator().manual_seed(68197))
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
    val_dataloader = DataLoader(LongDataset(val_dataset, shuffle=False), batch_size=config['training']['train_batch_size'],
                                **resolved['dataloader_kwargs'])
    print("Validation dataset size:", len(val_dataset))
    
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
    model, dataloader, optimizer, val_dataloader = accelerator.prepare(model, dataloader, optimizer, val_dataloader)
    accelerator.register_for_checkpointing(state)
    accelerator.register_for_checkpointing(ema)
    
    # Load from checkpoint if needed
    if ckpt_path:
        accelerator.load_state(ckpt_path)
        
    def validate(repeats, dataloader, pbar_title):
        validation_stats = {'loss': []}
        generator = torch.Generator(device=accelerator.device).manual_seed(config['training']['seed'])
        pbar = tqdm(total=repeats * len(val_dataset), desc=pbar_title)
        val_dataloader_iter = iter(dataloader)
        while pbar.n < pbar.total:
            batch = next(val_dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            
            sigma = torch.randn(images.shape[0], device=images.device, generator=generator).reshape(-1, 1, 1, 1)
            sigma = (sigma * config['evaluation']['P_std'] + config['evaluation']['P_mean']).exp()
            sigma_data = scheduler.config.sigma_data
            t = torch.atan(sigma / sigma_data)
            cnoise = t.flatten()
            
            noise = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * noise

            x = x_t / sigma_data
            if cond_img is not None:
                x = torch.cat([x, cond_img], dim=1)
                
            if sf_optim:
                optimizer.eval()
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                model_output, logvar = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs, return_logvar=True)
                pred_v_t = -sigma_data * model_output
                
            v_t = torch.cos(t) * noise - torch.sin(t) * images

            loss = 1 / (logvar.exp() * sigma_data ** 2) * ((pred_v_t - v_t) ** 2) + logvar
            loss = loss.mean()
            validation_stats['loss'].append(loss.item())
            
            pbar.update(images.shape[0])
            pbar.set_postfix(loss=np.mean(validation_stats['loss']))
                
        return np.mean(validation_stats['loss'])
                
        
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
        
        # Save full train config and model config
        with open(os.path.join(base_folder_path + '_checkpoint', f'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        model.save_config(os.path.join(base_folder_path + '_checkpoint', f'model_config_latest'))

    dataloader_iter = iter(dataloader)
    grad_norm = torch.tensor(0.0, device=accelerator.device)
    while state.epoch < config['training']['epochs']:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        stats_hist = {'loss': [], 'importance_loss': []}
        progress_bar = tqdm(dataloader_iter, desc=f"Epoch {state.epoch}", total=config['training']['epoch_steps'])
        while progress_bar.n < config['training']['epoch_steps']:            
            batch = next(dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')

            sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * config['training']['P_std'] + config['training']['P_mean']).exp()
            sigma_data = scheduler.config.sigma_data
            t = torch.atan(sigma / sigma_data)
            cnoise = t.flatten()
        
            noise = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * noise

            x = x_t / sigma_data
            if cond_img is not None:
                x = torch.cat([x, cond_img], dim=1)
                
            if sf_optim:
                optimizer.train()
            model.train()
            with accelerator.autocast(), accelerator.accumulate(model):
                model_output, logvar = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs, return_logvar=True)
                pred_v_t = -sigma_data * model_output
                
            v_t = torch.cos(t) * noise - torch.sin(t) * images

            loss = 1 / (logvar.exp() * sigma_data ** 2) * ((pred_v_t - v_t) ** 2) + logvar
            loss = loss.mean()

            state.seen += images.shape[0]
            state.step += 1
            lr = lr_scheduler.get(state.seen)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=config['training'].get('gradient_clip_val', 10.0))
            optimizer.step()

            if accelerator.is_main_process:
                if sf_optim:
                    optimizer.eval()
                ema.update()

            stats_hist['loss'].append(loss.item())
            progress_bar.set_postfix({'loss': np.mean(stats_hist['loss']),
                                      "lr": lr,
                                      "grad_norm": grad_norm.item()})
            progress_bar.update(1)
            
        progress_bar.close()
        val_loss = None
        eval_loss = None
        if config['training']['validate_epochs'] > 0 and (state.epoch + 1) % config['training']['validate_epochs'] == 0:
            val_loss = validate(config['training']['validation_repeats'], val_dataloader, "Validation Loss")
            if sf_optim:
                eval_loss = validate(config['training']['validation_repeats'], dataloader, "Eval Loss")

        state.epoch += 1
        if accelerator.is_main_process:
            log_values = {
                "loss": np.mean(stats_hist['loss']),
                "lr": lr,
                "step": state.step,
                "epoch": state.epoch,
                "seen": state.seen
            }
            if val_loss is not None:
                log_values['val_loss'] = val_loss
                log_values['eval_loss'] = eval_loss
            wandb.log(log_values, step=state.epoch, commit=True)
            if sf_optim:
                optimizer.eval()
            if state.epoch % config['logging']['temp_save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")

if __name__ == '__main__':
    main()