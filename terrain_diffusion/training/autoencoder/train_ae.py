from collections import defaultdict
import json
import click
from datetime import datetime
 
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
from ema_pytorch import PostHocEMA
import yaml
from terrain_diffusion.training.autoencoder.resnet_autoencoder import ResNetAutoencoder
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
from torchmetrics.image.fid import FrechetInceptionDistance

def get_optimizer(model, config, optimizer_key='optimizer'):
    """Get optimizer based on config settings."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found for optimizer. Check freezing settings.")
    if config[optimizer_key]['type'] == 'adam':
        optimizer = torch.optim.Adam(trainable_params, **config[optimizer_key]['kwargs'])
    else:
        raise ValueError(f"Unknown optimizer type: {config[optimizer_key]['type']}")
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
@click.option("--donot-resume-wandb", "donot_resume_wandb", is_flag=True, default=False, help="Don't resume W&B run")
@click.pass_context
def main(ctx, config_path, ckpt_path, model_ckpt_path, debug_run, resume_id, override, donot_resume_wandb):
    build_registry()
    
    with open(config_path, 'r') as f:
        config = Config(yaml.safe_load(f))
    
    if os.path.exists(f"{config['logging']['save_dir']}/latest_checkpoint") and not ckpt_path:
        print("The save_dir directory already exists. Would you like to resume training from the latest checkpoint? (y/n)")
        output = input()
        if output.lower() == "y":
            ckpt_path = f"{config['logging']['save_dir']}/latest_checkpoint"
        elif output.lower() == "n":
            print("Beginning new training run...")
        else:
            print("Unexpected input. Exiting...")
            return
        
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
    # Auto-resume W&B run from checkpoint metadata if available (unless explicitly provided)
    if ckpt_path and not resume_id and not debug_run and not donot_resume_wandb:
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
    wandb.init(
        **config['wandb'],
        config=config
    )
    print("Run ID:", wandb.run.id)
    
    resolved = registry.resolve(config, validate=False)
    tasks = resolved['training']['tasks']

    # Load anything that needs to be used for training.
    model = resolved['model']
    assert isinstance(model, EDMAutoencoder) or isinstance(model, ResNetAutoencoder), "Currently only EDMAutoencoder and ResNetAutoencoder are supported for autoencoder training."
    lr_scheduler = resolved['lr_sched']
    # Optionally train encoder only (freeze encoder layers)
    if config['training'].get('decoder_only', False):
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in model.parameters())
        print(f"Decoder-only mode: {num_trainable}/{num_total} parameters trainable.")
    
    train_dataset = resolved['train_dataset']
    val_dataset = resolved['val_dataset']
    
    optimizer = get_optimizer(model, config, optimizer_key='optimizer')
    resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
    ema = PostHocEMA(model, **resolved['ema'])
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    train_dataloader = DataLoader(LongDataset(train_dataset, shuffle=True), batch_size=config['training']['train_batch_size'],
                            **resolved['dataloader_kwargs'], drop_last=True)
    val_dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), batch_size=config['training']['train_batch_size'],
                                **resolved['dataloader_kwargs'], drop_last=True)
    perceptual_loss = lpips.LPIPS(net='alex', spatial=True)
    
    discriminator = None
    d_optimizer = None
    if config['training']['discriminator_weight'] > 0:
        discriminator = resolved['discriminator']
        d_optimizer = get_optimizer(discriminator, config, optimizer_key='optimizer_d')
    
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
    if discriminator is not None:
        model, discriminator, train_dataloader, optimizer, d_optimizer, val_dataloader = accelerator.prepare(
            model, discriminator, train_dataloader, optimizer, d_optimizer, val_dataloader)
    else:
        model, train_dataloader, optimizer, val_dataloader = accelerator.prepare(model, train_dataloader, optimizer, val_dataloader)
    perceptual_loss = accelerator.prepare(perceptual_loss)
    accelerator.register_for_checkpointing(state)
    accelerator.register_for_checkpointing(ema)
    
    # Load from checkpoint if needed
    if ckpt_path:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        accelerator.load_state(ckpt_path)

    def percep_loss_fn(reconstruction, reference):
        assert 1 == reconstruction.shape[1] == reference.shape[1]
        
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
    
    def validate(total_steps, dataloader, pbar_title):
        validation_stats = defaultdict(list)
        pbar = tqdm(total=total_steps, desc=pbar_title)
        val_dataloader_iter = iter(dataloader)
        fid_metric = None
        if config['evaluation'].get('eval_fid', False):
            fid_metric = FrechetInceptionDistance(feature=2048).to(accelerator.device)
        
        while pbar.n < pbar.total:
            batch = next(val_dataloader_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                scaled_clean_images = images
                if cond_img is not None:
                    scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
                
                # Encode and decode
                if config['training']['decoder_only']:
                    with torch.no_grad():
                        z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
                        z = model.postencode(z_means, z_logvars)
                else:
                    z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
                    z = model.postencode(z_means, z_logvars)
                decoded_x, logvar = model.decode(z, include_logvar=True)

                # Calculate losses
                loss = 0
                cidx = 0
                for tidx, task in enumerate(tasks):
                    task_loss = 0
                    channels = task['channels']
                    pred = decoded_x[:, cidx:cidx+channels]
                    target = scaled_clean_images[:, cidx:cidx+channels]
                    for loss_type, loss_weight in zip(task['losses'], task['weights']):
                        if loss_type == 'va_mae':
                            task_loss = task_loss + variance_adjusted_loss(pred, target) * loss_weight
                        elif loss_type == 'mse':
                            task_loss = task_loss + torch.nn.functional.mse_loss(pred, target) * loss_weight
                        elif loss_type == 'mae':
                            task_loss = task_loss + torch.nn.functional.l1_loss(pred, target) * loss_weight
                        elif loss_type == 'huber':
                            task_loss = task_loss + torch.nn.functional.smooth_l1_loss(pred, target, beta=task['beta']) * loss_weight
                        elif loss_type == 'percep':
                            task_loss = task_loss + percep_loss_fn(pred, target) * loss_weight
                        elif loss_type == 'bce':
                            unnormalized_target = target * task['data_std'] + task['data_mean']
                            task_loss = task_loss + torch.nn.functional.binary_cross_entropy_with_logits(pred, unnormalized_target) * loss_weight
                        else:
                            raise ValueError(f"Unknown loss type: {loss_type}")
                        validation_stats[f"{task['name']}_{loss_type}"].append(task_loss.item() / loss_weight)
                    cidx += channels
                    task_loss = task_loss * task['task_weight']
                    loss = loss + task_loss
                
                ndz_logvars = z_logvars[:, :model.config.latent_channels]
                ndz_means = z_means[:, :model.config.latent_channels]
                kl_loss = -0.5 * torch.mean(1 + ndz_logvars - ndz_means**2 - ndz_logvars.exp())
                
                # Combine losses with weights
                loss = loss + kl_loss * config['training']['kl_weight']

                # Update FID metric on first channel if available
                if fid_metric is not None:
                    try:
                        pred_residual = decoded_x[:, :1]
                        real_residual = scaled_clean_images[:, :1]
                        real_min = torch.amin(real_residual, dim=(1, 2, 3), keepdim=True)
                        real_max = torch.amax(real_residual, dim=(1, 2, 3), keepdim=True)
                        value_range = torch.maximum(real_max - real_min, torch.tensor(1.0, device=real_residual.device))
                        value_mid = (real_min + real_max) / 2
                        samples_norm = torch.clamp(((pred_residual - value_mid) / value_range + 0.5) * 255, 0, 255)
                        samples_norm = samples_norm.repeat(1, 3, 1, 1).to(torch.uint8)
                        real_norm = torch.clamp(((real_residual - value_mid) / value_range + 0.5) * 255, 0, 255)
                        real_norm = real_norm.repeat(1, 3, 1, 1).to(torch.uint8)
                        fid_metric.update(samples_norm, real=False)
                        fid_metric.update(real_norm, real=True)
                    except Exception:
                        pass

            # Record statistics
            validation_stats['loss'].append(loss.item())
            validation_stats['kl_loss'].append(kl_loss.item())
            
            pbar.update(images.shape[0])
            pbar.set_postfix({k: f"{np.mean(v):.3f}" for k, v in validation_stats.items()})
        
        # Return average losses
        out_stats = {k: np.mean(v) for k, v in validation_stats.items()}
        if fid_metric is not None:
            out_stats['fid'] = fid_metric.compute().item()
        return out_stats
    
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
        # Persist W&B run id for seamless resumption
        try:
            with open(os.path.join(base_folder_path + '_checkpoint', 'wandb_run.json'), 'w') as f:
                json.dump({'id': wandb.run.id if wandb.run else None}, f)
        except Exception:
            pass

    # Training loop
    train_iter = iter(train_dataloader)
    burnin_steps = config['training'].get('burnin_steps', 0)
    initial_r_gamma = config['training'].get('r_gamma', 0) * config['training'].get('r_warmup_factor', 10)
    final_r_gamma = config['training'].get('r_gamma', 0)
    
    # Warmup parameters for beta_2 in optimizers
    if discriminator is not None:
        initial_beta_2 = 1 - 10 * (1 - config['optimizer']['kwargs']['betas'][1])
        final_beta_2 = config['optimizer']['kwargs']['betas'][1]
        print("Warming beta_2 from", initial_beta_2, "to", final_beta_2)

    printed_size = False
    while state.epoch < config['training']['epochs']:
        stats_hist = defaultdict(list)
        progress_bar = tqdm(train_iter, desc=f"Epoch {state.epoch}", 
                          total=config['training']['epoch_steps'])
        
        while progress_bar.n < config['training']['epoch_steps']:
            # Warmup r_gamma and beta_2 during burnin steps
            if discriminator is not None and state['step'] < burnin_steps:
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
                for opt in [optimizer, d_optimizer]:
                    if opt is not None:
                        for group in opt.param_groups:
                            group['betas'] = (group['betas'][0], current_beta_2)
            else:
                current_r_gamma = final_r_gamma if discriminator is not None else 0

            # Train discriminator first (like in GAN training)
            real_pred = None
            if config['training']['discriminator_weight'] > 0:
                with accelerator.accumulate(discriminator):
                    # Use two different batches: one for real images, one for reconstruction
                    batch_real = next(train_iter)
                    real_images = batch_real['image']
                    real_batch_size = real_images.shape[0]

                    batch_fake_src = next(train_iter)
                    images_for_recon = batch_fake_src['image']
                    cond_img_fake = batch_fake_src.get('cond_img')
                    conditional_inputs_fake = batch_fake_src.get('cond_inputs')
                    
                    with accelerator.autocast():
                        discriminator.train()
                        
                        # Generate fake images (autoencoder reconstruction) from a different batch
                        with torch.no_grad():
                            scaled_fake_inputs = images_for_recon
                            if cond_img_fake is not None:
                                scaled_fake_inputs = torch.cat([scaled_fake_inputs, cond_img_fake], dim=1)
                            
                            z_means, z_logvars = model.preencode(scaled_fake_inputs, conditional_inputs_fake)
                            z = model.postencode(z_means, z_logvars)
                            decoded_x, logvar = model.decode(z, include_logvar=True)
                        
                        if not printed_size:
                            print("Decoded image size:", decoded_x.shape)
                            print("Real image size:", real_images.shape)
                            printed_size = True
                        
                        # Optimal pattern: single discriminator call on concatenated images
                        if len(config['training']['sigmoid_channels']) > 0:
                            x_list = []
                            prev_i = -1
                            for i in config['training']['sigmoid_channels']:
                                x_list.append(decoded_x[:, prev_i+1:i])
                                x_list.append(torch.sigmoid(decoded_x[:, i:i+1]))
                                prev_i = i
                            x_list.append(decoded_x[:, config['training']['sigmoid_channels'][-1]+1:])
                            sigmoided_decoded_x = torch.cat(x_list, dim=1)
                        else:
                            sigmoided_decoded_x = decoded_x
                        all_images = torch.cat([real_images, sigmoided_decoded_x.detach()], dim=0).detach().requires_grad_(True)
                        pred = discriminator(all_images)
                        real_pred = pred[:real_batch_size]
                        fake_pred = pred[real_batch_size:]
                        
                        # Discriminator loss (real should be high, fake should be low)
                        d_loss = torch.nn.functional.softplus(fake_pred - real_pred).mean()
                        
                        # Add gradient penalty
                        if current_r_gamma > 0 and state['step'] % config['training'].get('r_interval', 1) == 0:
                            # Compute gradient penalty
                            grad_real = torch.autograd.grad(
                                outputs=pred.sum(), inputs=all_images,
                                create_graph=True)[0]
                            r_reg = current_r_gamma * 0.5 * grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
                            total_d_loss = d_loss + r_reg
                        else:
                            r_reg = torch.tensor(0.0, device=d_loss.device)
                            total_d_loss = d_loss
                        
                    d_optimizer.zero_grad()
                    accelerator.backward(total_d_loss)
                    if accelerator.sync_gradients:
                        discriminator_grad_norm = accelerator.clip_grad_norm_(discriminator.parameters(), config['training'].get('grad_clip_val', 10.0))
                    d_optimizer.step()
                        
                    # Print gradient norms for each discriminator parameter
                    #for name, param in discriminator.named_parameters():
                    #    if param.grad is not None:
                    #        grad_norm = param.grad.norm().item()
                    #        print(f"{name}: {grad_norm:.6f}")
                    
                    stats_hist['d_loss'].append(d_loss.item())
                    stats_hist['r_loss'].append((r_reg.item() / current_r_gamma) if current_r_gamma > 0 else 0)
            
            # Fetch batch for autoencoder training
            batch = next(train_iter)
            images = batch['image']
            cond_img = batch.get('cond_img')
            conditional_inputs = batch.get('cond_inputs')
            batch_size = images.shape[0]
            
            # New real images for discriminator
            batch_real = next(train_iter)
            real_images = batch_real['image']
            real_batch_size = real_images.shape[0]

            # Train autoencoder (like generator in GAN training)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    if discriminator is not None:
                        discriminator.eval()
                    
                    scaled_clean_images = images
                    if cond_img is not None:
                        scaled_clean_images = torch.cat([scaled_clean_images, cond_img], dim=1)
                    
                    z_means, z_logvars = model.preencode(scaled_clean_images, conditional_inputs)
                    z = model.postencode(z_means, z_logvars)
                    decoded_x, logvar = model.decode(z, include_logvar=True)

                    # Calculate autoencoder losses
                    loss = 0
                    cidx = 0
                    for tidx, task in enumerate(tasks):
                        task_loss = 0
                        channels = task['channels']
                        pred = decoded_x[:, cidx:cidx+channels]
                        target = scaled_clean_images[:, cidx:cidx+channels]
                        for loss_type, loss_weight in zip(task['losses'], task['weights']):
                            if loss_type == 'va_mae':
                                task_loss = task_loss + variance_adjusted_loss(pred, target) * loss_weight
                            elif loss_type == 'mse':
                                task_loss = task_loss + torch.nn.functional.mse_loss(pred, target) * loss_weight
                            elif loss_type == 'mae':
                                task_loss = task_loss + torch.nn.functional.l1_loss(pred, target) * loss_weight
                            elif loss_type == 'huber':
                                task_loss = task_loss + torch.nn.functional.smooth_l1_loss(pred, target, beta=task['beta']) * loss_weight
                            elif loss_type == 'percep':
                                task_loss = task_loss + percep_loss_fn(pred, target) * loss_weight
                            elif loss_type == 'bce':
                                unnormalized_target = target * task['data_std'] + task['data_mean']
                                task_loss = task_loss + torch.nn.functional.binary_cross_entropy_with_logits(pred, unnormalized_target) * loss_weight
                            else:
                                raise ValueError(f"Unknown loss type: {loss_type}")
                            stats_hist[f"{task['name']}_{loss_type}"].append(task_loss.item() / loss_weight)
                        cidx += channels
                        task_loss = task_loss * task['task_weight']
                        loss = loss + task_loss
                    
                    loss = loss * linear_warmup(
                        config['training'].get('task_weight_warmup_factor', 1.0), 
                        1.0, 
                        state['step'], 
                        config['training'].get('burnin_steps', 1)
                    )
                    
                    # KL loss
                    ndz_logvars = z_logvars[:, :model.config.latent_channels]
                    ndz_means = z_means[:, :model.config.latent_channels]
                    kl_loss = -0.5 * torch.mean(1 + ndz_logvars - ndz_means**2 - ndz_logvars.exp())
                    loss = loss + kl_loss * config['training']['kl_weight']
                    
                    # Add adversarial loss if discriminator enabled
                    if config['training']['discriminator_weight'] > 0:
                        # Apply sigmoid to some channels if specified
                        if len(config['training']['sigmoid_channels']) > 0:
                            x_list = []
                            prev_i = 0
                            for i in config['training']['sigmoid_channels']:
                                x_list.append(decoded_x[:, prev_i:i])
                                x_list.append(torch.sigmoid(decoded_x[:, i:i+1]))
                                prev_i = i
                            x_list.append(decoded_x[:, config['training']['sigmoid_channels'][-1]+1:])
                            sigmoided_decoded_x = torch.cat(x_list, dim=1)
                        else:
                            sigmoided_decoded_x = decoded_x
                            
                        if True:
                            with torch.no_grad():
                                real_pred = discriminator(real_images)
                        
                        fake_pred = discriminator(sigmoided_decoded_x)
                        adv_loss = torch.nn.functional.softplus(real_pred.detach() - fake_pred).mean()
                        loss = loss + adv_loss * config['training']['discriminator_weight'] * linear_warmup(
                            config['training'].get('disc_weight_warmup_factor', 1.0), 
                            1.0, 
                            state['step'], 
                            config['training'].get('burnin_steps', 1)
                        )
                        stats_hist['adv_loss'].append(adv_loss.item())
                
                #loss = loss * 0
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    generator_grad_norm = accelerator.clip_grad_norm_(model.parameters(), config['training'].get('grad_clip_val', 10.0))
                optimizer.step()

            if accelerator.is_main_process:
                ema.update()  

            state['seen'] += batch_size
            state['step'] += 1
            
            # Update learning rates (like in GAN training)
            lr_warmup = linear_warmup(
                config['training'].get('lr_warmup_factor', 1.0), 
                1.0, 
                state['step'], 
                config['training'].get('burnin_steps', 1)
            )
            disc_lr_warmup = linear_warmup(
                config['training'].get('disc_lr_warmup_factor', 1.0), 
                1.0, 
                state['step'], 
                config['training'].get('burnin_steps', 1)
            )
            lr = lr_scheduler.get(state.seen) * lr_warmup
            for g in optimizer.param_groups:
                g['lr'] = lr
            if discriminator is not None:
                for g in d_optimizer.param_groups:
                    g['lr'] = lr * config['training'].get('disc_lr_mult', 1.0) * disc_lr_warmup

            stats_hist['loss'].append(loss.item())
            stats_hist['kl_loss'].append(kl_loss.item())
            
            progress_bar.set_postfix({
                'loss': f"{np.mean(stats_hist['loss'][-10:]):.4f}",
                'kl_loss': f"{np.mean(stats_hist['kl_loss'][-10:]):.4f}",
                'd_loss': f"{np.mean(stats_hist['d_loss'][-10:]):.4f}" if discriminator else "N/A",
                "r_loss": f"{np.mean(stats_hist['r_loss'][-10:]):.4f}" if discriminator else "N/A",
                'adv_loss': f"{np.mean(stats_hist['adv_loss'][-10:]):.4f}" if discriminator else "N/A",
                'lr': lr,
                'd_grad_norm': f"{discriminator_grad_norm:.4f}" if discriminator else "N/A",
                'g_grad_norm': f"{generator_grad_norm:.4f}"
            })
            progress_bar.update(1)
            
        progress_bar.close()

        val_losses = None
        eval_losses = None
        if config['evaluation']['validate_epochs'] > 0 and (state.epoch + 1) % config['evaluation']['validate_epochs'] == 0:
            if config['evaluation'].get('val_ema_idx', -1) >= 0 and config['evaluation']['val_ema_idx'] < len(ema.ema_models):
                with temporary_ema_to_model(ema.ema_models[config['evaluation']['val_ema_idx']]):
                    val_losses = validate(config['evaluation']['validation_steps'], val_dataloader, "Validation Loss")
                    if config['evaluation'].get('training_eval', False):
                        eval_losses = validate(config['evaluation']['validation_steps'], train_dataloader, "Eval Loss")
            else:
                if config['evaluation'].get('val_ema_idx', -1) >= 0:
                    warnings.warn(f"Invalid val_ema_idx: {config['evaluation']['val_ema_idx']}. "
                                  "Falling back to using the model's parameters.")
                val_losses = validate(config['evaluation']['validation_steps'], val_dataloader, "Validation Loss")
                if config['evaluation'].get('training_eval', False):
                    eval_losses = validate(config['evaluation']['validation_steps'], train_dataloader, "Eval Loss")
                    
        state.epoch += 1
        if accelerator.is_main_process:
            wandb_logs = {
                "lr": lr,
                "step": state.step,
                "epoch": state.epoch,
                "seen": state.seen
            }
            wandb_logs.update({k: np.nanmean(v) for k, v in stats_hist.items()})
            if val_losses:
                val_losses = {f"val/{k}": v for k, v in val_losses.items()}
                wandb_logs.update(val_losses)
            if eval_losses:
                eval_losses = {f"eval/{k}": v for k, v in eval_losses.items()}
                wandb_logs.update(eval_losses)
            wandb.log(wandb_logs, step=state.epoch, commit=True)
            if state.epoch % config['logging']['temp_save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state.epoch % config['logging']['save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/{state.seen//1000}kimg")

if __name__ == '__main__':
    main()
