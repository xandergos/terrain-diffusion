import catalogue
import click
import datetime
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from diffusers.training_utils import EMAModel
from diffusion.datasets.datasets import LongDataset
from diffusion.encoder import *
from diffusion.registry import build_registry
from diffusion.samplers.tiled import TiledSampler
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from utils import SerializableEasyDict as EasyDict

def log_samples(images, config, state, encoder):
    import math
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np

    # Convert samples to numpy array and move to CPU
    samples_np = images.cpu().numpy()

    # Define custom normalization
    class SignedLogNorm(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, clip=False):
            super().__init__(vmin, vmax, clip)

        def __call__(self, value, clip=None):
            cbrt_min = np.sign(self.vmin) * np.sqrt(np.abs(self.vmin))
            cbrt_max = np.sign(self.vmax) * np.sqrt(np.abs(self.vmax))
            normalized = (np.sign(value) * np.sqrt(np.abs(value)) - cbrt_min) / (cbrt_max - cbrt_min)
            return normalized

    # Calculate grid size
    bs = samples_np.shape[0]
    grid_size = int(math.sqrt(bs))
    
    # Create a figure with subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f'Generated Terrain Samples at Epoch {state.epoch}')
    
    # Plot each sample
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < bs:
                norm = SignedLogNorm(vmin=samples_np[idx, 0].min(), vmax=samples_np[idx, 0].max())
                im = axs[i, j].imshow(samples_np[idx, 0], cmap='terrain', norm=norm)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].axis('off')
                cbar = fig.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)
                
                # Set colorbar ticks to show original values
                min_val, max_val = samples_np[idx, 0].min(), samples_np[idx, 0].max()
                cbar.set_ticks([min_val, 0, max_val])
                cbar.set_ticklabels([f'{min_val:.2f}', '0', f'{max_val:.2f}'])
            else:
                axs[i, j].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{config['logging']['save_dir']}/terrain_samples_epoch_{state.epoch}.png")
    plt.close(fig)
    
    # Log the entire figure to wandb
    wandb.log({"samples": wandb.Image(f"{config['logging']['save_dir']}/terrain_samples_epoch_{state.epoch}.png")},
              commit=False, step=state.epoch)


@click.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False)
def main(config_path, ckpt_path):
    build_registry()
    
    config = Config().from_disk(config_path) if config_path else None
            
    resolved = registry.resolve(config, validate=False)

    # Load anything that needs to be used for training.
    model = resolved['model']
    lr_scheduler = resolved['lr_sched']
    train_dataset = resolved['train_dataset']
    scheduler = resolved['scheduler']
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99))
    ema = EMAModel(model.to('cuda').parameters(), **resolved['ema'])
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    dataloader = DataLoader(LongDataset(train_dataset), batch_size=config['training']['train_batch_size'],
                            **resolved['dataloader_kwargs'])

    # Setup accelerate
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=resolved['training']['gradient_accumulation_steps'],
        log_with=None
    )
    model, dataloader, optimizer, ema = accelerator.prepare(model, dataloader, optimizer, ema)
    accelerator.register_for_checkpointing(state)
    
    # Temporary fix for old checkpoints
    # path = 'checkpoints/64_128x3/latest.pt'
    # ckpt = torch.load(path)
    # model.load_state_dict(ckpt['model'])
    # optimizer.load_state_dict(ckpt['optimizer'])
    # ema.load_state_dict(ckpt['ema'])
    # state = EasyDict(ckpt['state'])
    
    # Load from checkpoint if needed
    if ckpt_path:
        accelerator.load_state(ckpt_path)
    
    if accelerator.is_main_process:
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
        
    def save_model(base_folder_path, overwrite=False):
        if os.path.exists(base_folder_path + '_model') and not overwrite:
            strtime = datetime.now().strftime("_%Y%m%d_%H%M%S")
            base_folder_path = f"{base_folder_path}{strtime}"
        elif os.path.exists(base_folder_path + '_model'):
            safe_rmtree(base_folder_path + '_model')
        os.makedirs(base_folder_path + '_model', exist_ok=False)
        
        # Saving the EMA model
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        model.save_pretrained(base_folder_path + '_model')
        ema.restore(model.parameters())

    dataloader_iter = iter(dataloader)
    while state.epoch < config['training']['epochs']:
        loss_hist = []
        progress_bar = tqdm(dataloader_iter, desc=f"Epoch {state.epoch}", total=config['training']['epoch_steps'])
        while progress_bar.n < config['training']['epoch_steps']:
            batch = next(dataloader_iter)
            images = batch['image']
            label = batch.get('label')
            cond_img = batch.get('cond_img')
            context = batch.get('context')

            sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * config['training']['P_std'] + config['training']['P_mean']).exp()
            sigma_data = scheduler.config.sigma_data
            
            # Replace 0.1% of sigmas with uniform random values to ensure very large sigmas are reached
            num_to_replace = np.random.poisson(config['training']['random_replace_rate'] * sigma.numel())
            max_train_sigma = scheduler.config.sigma_max / scheduler.config.scaling_t
            uniform_sigma = torch.rand(num_to_replace, device=sigma.device) * (max_train_sigma - scheduler.config.sigma_min) + scheduler.config.sigma_min
            sigma.view(-1)[:num_to_replace] = uniform_sigma
        
            noise = torch.randn_like(images) * sigma
            noisy_images = images + noise

            cnoise = scheduler.precondition_noise(sigma).flatten()
            scaled_noisy_images = scheduler.precondition_inputs(noisy_images, sigma)

            x = scaled_noisy_images
            if cond_img is not None:
                x = torch.cat([x, cond_img], dim=1)
            with accelerator.autocast(), accelerator.accumulate(model):
                pred_noise, logvar = model(x, noise_labels=cnoise, label_index=label, context=context, return_logvar=True)
                denoised = scheduler.precondition_outputs(noisy_images, pred_noise, sigma)

                weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
                weight = weight.repeat(1, pred_noise.shape[1], 1, 1)
                max_weight = torch.tensor(config['training']['channel_max_weight'], device=weight.device, dtype=weight.dtype).reshape([1, pred_noise.shape[1], 1, 1])
                weight = torch.minimum(weight, max_weight)
                
                loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
                loss = loss.mean()

                state.seen += images.shape[0]
                state.step += 1
                lr = lr_scheduler.get(state.seen)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                optimizer.zero_grad()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            ema.step(model.parameters())

            loss_hist.append(loss.item())
            progress_bar.set_postfix({'loss': np.mean(loss_hist), "lr": lr})
            progress_bar.update(1)

        state.epoch += 1
        if accelerator.is_main_process:
            wandb.log({
                "loss": np.mean(loss_hist),
                "lr": lr,
                "step": state.step,
                "epoch": state.epoch,
                "seen": state.seen
            }, step=state.epoch)
            if state.epoch % config['logging']['temp_save_epochs'] == 0:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
                save_model(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0:
                save_checkpoint(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")
                save_model(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")
            if state.epoch % config['logging']['plot_epochs'] == 0:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                
                sampler = TiledSampler(model=model, 
                                       scheduler=scheduler,
                                       **resolved['sampler']['init'])
                images = sampler.get_region(*resolved['sampler']['region'])
                log_samples(images, config, state, train_dataset.sub_datasets[0].encoder)
                
                ema.restore(model.parameters())

if __name__ == '__main__':
    main()