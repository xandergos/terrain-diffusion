import datetime
import json
import os
import random
import click
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
import catalogue
from confection import registry, Config
import numpy as np
import torch
from tqdm import tqdm
import wandb
from easydict import EasyDict

from diffusion.encoder import denoise_pyramid_layer

from diffusion.datasets.datasets import BaseTerrainDataset, LongDataset, MultiDataset, UpsamplingTerrainDataset
from diffusion.loss import SqrtLRScheduler
from diffusion.unet import EDMUnet2D

from diffusers.training_utils import EMAModel

from torch.utils.data import DataLoader
from accelerate import Accelerator

@torch.no_grad()
def generate_images(image_shape, model, scheduler, timesteps, device, generator=None,
                    cond_image=None, **net_inputs):
    scheduler.set_timesteps(timesteps)

    batch_size = image_shape[0]
    sample = torch.randn(*image_shape, generator=generator, device=device) * scheduler.sigmas[0]
    for t, sigma in tqdm(zip(scheduler.timesteps, scheduler.sigmas), desc="Generating images"):
        t = t.to(device)
        x = scheduler.precondition_inputs(sample, sigma)
        if cond_image is not None:
            x = torch.cat([sample, cond_image], dim=1)
        pred_noise = model(x, t.repeat(batch_size).flatten(), **net_inputs)
        
        sample = scheduler.step(pred_noise, t, sample).prev_sample

    return sample

def log_samples(samples, config, state, encoder):
    import math
    import matplotlib.pyplot as plt 
    # Convert samples to numpy array and move to CPU
    samples_np = samples.cpu().numpy()
    
    for i in range(samples_np.shape[0]):
        samples_np[i, 1] = denoise_pyramid_layer(samples_np[i, [1]], encoder, depth=1, maxiter=3)[0]
        
    samples_np = samples_np[:, 0, ...] * 155 * 2 + samples_np[:, 1, ...] * 2420 * 2 - 2651  
    
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
                im = axs[i, j].imshow(samples_np[idx], cmap='terrain')
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].axis('off')
                cbar = fig.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)
                cbar.set_ticks([samples_np[idx].min(), samples_np[idx].max()])
                cbar.set_ticklabels([f'{samples_np[idx].min():.2f}', f'{samples_np[idx].max():.2f}'])
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
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=False)
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False)
@click.option("--not-strict", "not_strict_load", is_flag=True, type=bool, default=False, required=False)
def main(config_path, ckpt_path, not_strict_load):
    if not config_path and not ckpt_path:
        click.echo("--config OR --ckpt must be provided.", err=True)
        raise click.Abort()

    registry.scheduler = catalogue.create("confection", "schedulers", entry_points=False)
    registry.scheduler.register("edm_dpm", func=EDMDPMSolverMultistepScheduler)

    registry.model = catalogue.create("confection", "models", entry_points=False)
    registry.model.register("unet", func=EDMUnet2D)

    registry.lr_sched = catalogue.create("confection", "lr_sched", entry_points=False)
    registry.lr_sched.register("sqrt", func=SqrtLRScheduler)

    registry.dataset = catalogue.create("confection", "datasets", entry_points=False)
    registry.dataset.register("base_terrain", func=BaseTerrainDataset)
    registry.dataset.register("upsampling_terrain", func=UpsamplingTerrainDataset)
    registry.dataset.register("multi_dataset", func=MultiDataset)

    config = Config().from_disk(config_path) if config_path else None
    ckpt = None

    if ckpt_path:
        ckpt = torch.load(ckpt_path, weights_only=False)
        if 'config' in ckpt:
            ckpt_config = Config().from_str(ckpt['config'])
            
            if config:
                if config['training']['override_checkpoint']:
                    ckpt_config.update(config)
                    config = ckpt_config
                else:
                    config.update(ckpt_config)
            else:
                config = ckpt_config
        elif config:
            click.echo("Warning: No config found in checkpoint. Using config from command line arguments.")
        else:
            click.echo("No config found in checkpoint or command line arguments.", err=True)
            raise click.Abort()
            

    resolved = registry.resolve(config, validate=False)

    # Load anything that needs to be used for training.
    model = resolved['model']
    lr_scheduler = resolved['lr_sched']
    train_dataset = resolved['train_dataset']
    scheduler = resolved['scheduler']
    optimizer = torch.optim.Adam(model.parameters())
    ema = EMAModel(model.to('cuda').parameters(), **resolved['ema'])
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    dataloader = DataLoader(LongDataset(train_dataset), batch_size=config['training']['train_batch_size'],
                            **resolved['dataloader_kwargs'])
    
    # Loading evaluation data
    random.seed(config['logging']['seed'])
    if model.config['label_dim'] > 0:
        assert len(config['logging']['label_weights']) == model.config['label_dim']
        labels = random.choices(list(range(model.config['label_dim'])), k=config['training']['eval_batch_size'], 
                                weights=config['logging']['label_weights'])
    if 'eval_dataset' in resolved:
        eval_dataloader = DataLoader(resolved['eval_dataset'], batch_size=config['training']['eval_batch_size'])
        eval_data = next(iter(eval_dataloader))
    else:
        eval_data = None

    # Load from checkpoint if needed
    if ckpt:
        click.echo("Loading from checkpoint...")
        
        for key in config['checkpoint']['ignore_keys']:
            if key in ckpt:
                del ckpt[key]
        #ckpt['model'] = ckpt['net']
        #for key in list(ckpt['model'].keys()):
        #    ckpt['model'][key.replace('unet.', '')] = ckpt['model'][key]
        #    del ckpt['model'][key]
        #    
        #ckpt['model']['noise_fourier.freqs'] = ckpt['model']['emb_fourier.freqs']
        #ckpt['model']['noise_fourier.phases'] = ckpt['model']['emb_fourier.phases']
        #ckpt['model']['noise_linear.weight'] = ckpt['model']['emb_noise.weight']
        #ckpt['model']['label_embeds.weight'] = torch.transpose(ckpt['model']['emb_label.weight'], 0, 1)
        
        model.load_state_dict(ckpt['model'], strict=not not_strict_load)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            click.echo("Warning: No optimizer found in checkpoint.")
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        else:
            click.echo("Warning: No EMA found in checkpoint, initializing new EMA model from loaded model.")
            ema = EMAModel(model.to('cuda').parameters(), **resolved['ema'])
        if 'state' in ckpt:
            state = EasyDict(ckpt['state'])
        else:
            click.echo("Warning: No state found in checkpoint.")

    # Setup accelerate
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=resolved['training']['gradient_accumulation_steps'],
        log_with=None
    )
    model, dataloader, optimizer, ema = accelerator.prepare(model, dataloader, optimizer, ema)

    
    if accelerator.is_main_process:
        wandb.init(
            **config['wandb'],
            config=config
        )

    def save_state(file_path, overwrite=False):
        if os.path.exists(file_path) and not overwrite:
            ext = os.path.extsep + os.path.basename(file_path).split(os.path.extsep)[-1]
            base = file_path[:-len(ext)]
            strtime = datetime.now().strftime("_%Y%m%d_%H%M%S")
            file_path = f"{base}{strtime}{ext}"
        assert not os.path.exists(file_path) or overwrite
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            "config": config.to_str(),
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": accelerator.unwrap_model(ema).state_dict(),
            'state': dict(state)
        }, file_path)

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
                pred_noise, logvar = model(x, cnoise, label, context, return_logvar=True)
                denoised = scheduler.precondition_outputs(noisy_images, pred_noise, sigma)

                weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
                weight = weight.repeat(1, x.shape[1], 1, 1)
                max_weight = torch.tensor(config['training']['channel_max_weight'], device=weight.device, dtype=weight.dtype).reshape([1, weight.shape[1], 1, 1])
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
                save_state(f"{config['logging']['save_dir']}/latest.pt", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0:
                save_state(f"{config['logging']['save_dir']}/{state.seen//1000}kimg.pt")
            if state.epoch % config['logging']['plot_epochs'] == 0:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                
                bs = config['training']['eval_batch_size']
                image_size = (bs, model.config['out_channels'], model.config['image_size'], model.config['image_size'])
                if model.config['label_dim'] > 0:
                    labels_tensor = torch.tensor(labels, device=accelerator.device)
                else:
                    labels_tensor = None
                if eval_data is not None:
                    cond_eval_data = dict.copy(eval_data)
                    if 'image' in cond_eval_data:
                        del cond_eval_data['image']
                    samples = generate_images(image_size, model, scheduler, config['logging']['generation_steps'], accelerator.device,
                                              generator=torch.Generator(device=accelerator.device).manual_seed(43),
                                              label_index=labels_tensor, **cond_eval_data)
                else:
                    samples = generate_images(image_size, model, scheduler, config['logging']['generation_steps'], accelerator.device,
                                              generator=torch.Generator(device=accelerator.device).manual_seed(43),
                                              label_index=labels_tensor)
                log_samples(samples, config, state, train_dataset.sub_datasets[0].encoder)
                
                ema.restore(model.parameters())

if __name__ == '__main__':
    main()