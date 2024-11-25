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
from utils import SerializableEasyDict as EasyDict
from schedulefree import ScheduleFreeWrapper, AdamWScheduleFree

from PIL import Image
import warnings

def log_samples(images, config, state):
    import numpy as np
    
    # Convert samples to numpy array and move to CPU
    samples_np = images.cpu().numpy()
    
    # Normalize to 0-1 range
    samples_np = (samples_np - np.min(samples_np, axis=(1, 2, 3), keepdims=True)) / (np.max(samples_np, axis=(1, 2, 3), keepdims=True) - np.min(samples_np, axis=(1, 2, 3), keepdims=True))
    
    # Convert to PIL images
    pil_images = [Image.fromarray((img * 255).astype(np.uint8)[0]) for img in samples_np]
    
    # Create grid
    num_images = len(pil_images)
    grid_width = int(np.sqrt(num_images))
    grid_height = grid_width
    
    # Remove extra images if there are more than grid_width * grid_height
    max_images = grid_width * grid_height
    if num_images > max_images:
        pil_images = pil_images[:max_images]
        num_images = max_images
    
    img_width, img_height = pil_images[0].size
    grid_img = Image.new('RGB', (grid_width * img_width, grid_height * img_height))
    
    for i, img in enumerate(pil_images):
        x = (i % grid_width) * img_width
        y = (i // grid_width) * img_height
        grid_img.paste(img, (x, y))
    
    # Save the grid image
    os.makedirs(f"{config['logging']['save_dir']}/samples", exist_ok=True)
    save_path = f"{config['logging']['save_dir']}/samples/terrain_samples_epoch_{state.epoch}.png"
    grid_img.save(save_path)
    
    wandb.log({"samples": [wandb.Image(img) for img in pil_images]},
              commit=False, step=state.epoch)

def plot_tensor_channels(x):
    """
    Plot the channels of a tensor side by side.
    
    Args:
        x (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W).
    """
    import matplotlib.pyplot as plt
    import torch
    
    # Ensure the tensor is on CPU and convert to numpy
    x = x.detach().cpu().numpy()
    
    # If the input is a batch, take the first item
    if x.ndim == 4:
        x = x[0]
    
    # Get the number of channels
    num_channels = x.shape[0]
    
    # Create a figure with subplots for each channel
    fig, axes = plt.subplots(1, num_channels, figsize=(4*num_channels, 4))
    
    # If there's only one channel, axes will not be an array
    if num_channels == 1:
        axes = [axes]
    
    # Plot each channel
    for i, ax in enumerate(axes):
        im = ax.imshow(x[i], cmap='viridis')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    plt.close()

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
    lr_scheduler = resolved['lr_sched']
    dataset = resolved['dataset']
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
        
    registry.utils.register("get_object", func=lambda object: {'model': model, 'scheduler': scheduler}[object])
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

    dataloader_iter = iter(dataloader)
    grad_norm = torch.tensor(0.0, device=accelerator.device)
    while state.epoch < config['training']['epochs']:
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
        if (state.epoch + 1) % config['training']['validate_epochs'] == 0:
            val_loss = validate(config['training']['validation_repeats'], val_dataloader, "Validation Loss")
            eval_loss = validate(config['training']['validation_repeats'], dataloader, "Eval Loss")
        else:
            val_loss = None
            eval_loss = None

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