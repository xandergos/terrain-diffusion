from contextlib import contextmanager
import json
import click
from datetime import datetime
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import LongDataset
from terrain_diffusion.data.laplacian_encoder import *
from terrain_diffusion.training.diffusion.registry import build_registry
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree
from heavyball.foreach_soap import ForeachSOAP
from heavyball.foreach_adamw import ForeachAdamW
from terrain_diffusion.training.diffusion.unet import EDMUnet2D
import torch._dynamo.config
import torch._inductor.config
from terrain_diffusion.training.utils import *
from terrain_diffusion.training.utils import SerializableEasyDict as EasyDict

from PIL import Image
import warnings

        
def get_optimizer(model, config):
    if config['optimizer']['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **config['optimizer']['kwargs'])
    elif config['optimizer']['type'] == 'heavyball-adam':
        optimizer = ForeachAdamW(model.parameters(), **config['optimizer']['kwargs'])
    elif config['optimizer']['type'] == 'soap':
        optimizer = ForeachSOAP(model.parameters(), **config['optimizer']['kwargs'])
    else:
        raise ValueError(f"Invalid optimizer type: {config['optimizer']['type']}. Options are 'adam', 'heavyball-adam', and 'soap'.")
    return optimizer

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
    assert isinstance(model, EDMUnet2D), "Currently only EDMUnet2D is supported for diffusion training."
    lr_scheduler = resolved['lr_sched']
    
    train_dataset = resolved['train_dataset']
    val_dataset = resolved['val_dataset']
        
    scheduler = resolved['scheduler']
    optimizer = get_optimizer(model, config)
    resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
    ema = PostHocEMA(model, **resolved['ema'])
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    dataloader = DataLoader(LongDataset(train_dataset, shuffle=True), batch_size=config['training']['train_batch_size'],
                            **resolved['dataloader_kwargs'], drop_last=True)
    val_dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), batch_size=config['training']['train_batch_size'],
                                **resolved['dataloader_kwargs'], drop_last=True)
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
        with torch.no_grad():
            model.logvar_linear.weight.copy_(torch.randn_like(model.logvar_linear.weight))
        
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
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
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
            
            if config['evaluation'].get('scale_sigma', False):
                sigma = sigma * torch.maximum(torch.std(images, dim=[1, 2, 3], keepdim=True) / sigma_data, config['evaluation'].get('sigma_scale_eps', 1e-2))
            
            sigma_data = scheduler.config.sigma_data
            t = torch.atan(sigma / sigma_data)
            cnoise = t.flatten()
            
            noise = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * noise

            x = x_t / sigma_data
            if cond_img is not None:
                x = torch.cat([x, cond_img], dim=1)
                
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                model_output, logvar = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs, return_logvar=True)
                pred_v_t = -sigma_data * model_output
                
            v_t = torch.cos(t) * noise - torch.sin(t) * images

            loss = 1 / (logvar.exp() * sigma_data ** 2) * ((pred_v_t - v_t) ** 2) + logvar
            loss = loss.mean()
            validation_stats['loss'].append(loss.item())
            
            pbar.update(images.shape[0])
            pbar.set_postfix(loss=f"{np.mean(validation_stats['loss']):.4f}")
                
        return np.mean(validation_stats['loss'])

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

    dataloader_iter = iter(dataloader)
    grad_norm = 0.0
    while state.epoch < config['training']['epochs']:
        stats_hist = {'loss': []}
        progress_bar = tqdm(dataloader_iter, desc=f"Epoch {state.epoch}", total=config['training']['epoch_steps'])
        while progress_bar.n < config['training']['epoch_steps']:  
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    batch = next(dataloader_iter)
                    images = batch['image']
                    cond_img = batch.get('cond_img')
                    conditional_inputs = batch.get('cond_inputs')
                    
                    sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
                    sigma = (sigma * config['training']['P_std'] + config['training']['P_mean']).exp()
                    if config['training'].get('scale_sigma', False):
                        sigma = sigma * torch.maximum(torch.std(images, dim=[1, 2, 3], keepdim=True) / sigma_data, config['training'].get('sigma_scale_eps', 1e-2))
                
                    sigma_data = scheduler.config.sigma_data
                    t = torch.atan(sigma / sigma_data)
                    cnoise = t.flatten()
                
                    noise = torch.randn_like(images) * sigma_data
                    x_t = torch.cos(t) * images + torch.sin(t) * noise

                    x = x_t / sigma_data
                    if cond_img is not None:
                        x = torch.cat([x, cond_img], dim=1)
                    
                    if isinstance(optimizer, AdamWScheduleFree):
                        optimizer.train()
                    model.train()
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
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=config['training'].get('gradient_clip_val', 10.0)).item()
                optimizer.step()

            if accelerator.is_main_process:
                ema.update()

            stats_hist['loss'].append(loss.item())
            progress_bar.set_postfix({'loss': f'{np.mean(stats_hist["loss"]):.4f}',
                                      "lr": lr,
                                      "grad_norm": grad_norm})
            progress_bar.update(1)
            
        progress_bar.close()
        
        val_loss = None
        eval_loss = None
        if config['evaluation']['validate_epochs'] > 0 and (state.epoch + 1) % config['evaluation']['validate_epochs'] == 0:
            if config['evaluation'].get('val_ema_idx', -1) >= 0 and config['evaluation']['val_ema_idx'] < len(ema.ema_models):
                with temporary_ema_to_model(ema.ema_models[config['evaluation']['val_ema_idx']]):
                    val_loss = validate(config['evaluation']['validation_repeats'], val_dataloader, "Validation Loss")
                    if config['evaluation'].get('training_eval', False):
                        eval_loss = validate(config['evaluation']['validation_repeats'], dataloader, "Eval Loss")
            else:
                if config['evaluation'].get('val_ema_idx', -1) >= 0:
                    warnings.warn(f"Invalid val_ema_idx: {config['evaluation']['val_ema_idx']}. "
                                  "Falling back to using the model's parameters.")
                val_loss = validate(config['evaluation']['validation_repeats'], val_dataloader, "Validation Loss")
                if config['evaluation'].get('training_eval', False):
                    eval_loss = validate(config['evaluation']['validation_repeats'], dataloader, "Eval Loss")

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
            if eval_loss is not None:
                log_values['eval_loss'] = eval_loss
            wandb.log(log_values, step=state.epoch, commit=True)
            if state.epoch % config['logging']['temp_save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")

if __name__ == '__main__':
    main()