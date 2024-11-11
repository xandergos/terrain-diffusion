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
from soap import SOAP
from schedulefree import ScheduleFreeWrapper, AdamWScheduleFree

from PIL import Image

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

@click.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False)
@click.option("--model-ckpt", "model_ckpt_path", type=click.Path(exists=True), required=False)
@click.option("--debug-run", "debug_run", is_flag=True, default=False)
def main(config_path, ckpt_path, model_ckpt_path, debug_run):
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
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
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

    dataloader_iter = iter(dataloader)
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
            
            # Calculate likelihood ratio of P_mean = -0.4, P_std = 1.0 to current distribution
            # for consistent comparison between runs
            sigma_importance_ratio = config['training']['P_std'] / 1.4 * torch.exp(
                (torch.log(sigma) - config['training']['P_mean'])**2 / (2 * config['training']['P_std']**2)
                - (torch.log(sigma) + 0.4)**2 / (2 * 1.4**2)
            )
            
            # Replace 0.1% of sigmas with uniform random values to ensure very large sigmas are reached
            num_to_replace = np.random.poisson(config['training']['random_replace_rate'] * sigma.numel())
            max_train_sigma = scheduler.config.sigma_max / scheduler.config.scaling_t
            uniform_sigma = torch.rand(num_to_replace, device=sigma.device) * (max_train_sigma - scheduler.config.sigma_min) + scheduler.config.sigma_min
            sigma.view(-1)[:num_to_replace] = uniform_sigma
        
            noise = torch.randn_like(images) * sigma
            noisy_images = images + noise

            cnoise = scheduler.trigflow_precondition_noise(sigma).flatten()
            scaled_noisy_images = scheduler.precondition_inputs(noisy_images, sigma)

            x = scaled_noisy_images
            if cond_img is not None:
                x = torch.cat([x, cond_img], dim=1)
                
            if sf_optim:
                optimizer.train()
            with accelerator.autocast(), accelerator.accumulate(model):
                pred_noise, logvar = model(x, noise_labels=cnoise, conditional_inputs=conditional_inputs, return_logvar=True)
                
            denoised = scheduler.precondition_outputs(noisy_images, pred_noise, sigma)

            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
            weight = weight.repeat(1, pred_noise.shape[1], 1, 1)
            max_weight = torch.tensor(config['training']['channel_max_weight'], device=weight.device, dtype=weight.dtype).reshape([1, pred_noise.shape[1], 1, 1])
            weight = torch.minimum(weight, max_weight)
        
            loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
            loss = loss.mean()
                
            with torch.no_grad():
                importance_loss = loss * sigma_importance_ratio
            importance_loss = importance_loss.mean()

            state.seen += images.shape[0]
            state.step += 1
            lr = lr_scheduler.get(state.seen)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training'].get('gradient_clip_val', 1.0))
            optimizer.step()

            if accelerator.is_main_process:
                if sf_optim:
                    optimizer.eval()
                ema.update()

            stats_hist['loss'].append(loss.item())
            stats_hist['importance_loss'].append(importance_loss.item())
            progress_bar.set_postfix({'loss': np.mean(stats_hist['loss']), 
                                      "importance_loss": np.mean(stats_hist['importance_loss']), 
                                      "lr": lr})
            progress_bar.update(1)

        state.epoch += 1
        if accelerator.is_main_process:
            wandb.log({
                "avg_loss": np.mean(stats_hist['loss']),
                "loss": np.median(stats_hist['loss']),
                "importance_loss": np.mean(stats_hist['importance_loss']),
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
            if sampler_config is not None and state.epoch % config['logging']['plot_epochs'] == 0:
                sampler_resolved = registry.resolve(sampler_config, validate=False)
                sampler = sampler_resolved['init']
                images = sampler.get_region(*sampler_resolved['region'])
                    
                log_samples(images, config, state)
                # Delete sampler to free memory in case models are used
                del sampler, sampler_resolved

if __name__ == '__main__':
    main()