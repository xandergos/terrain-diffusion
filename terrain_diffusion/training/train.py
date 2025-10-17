"""Unified training script for all model types."""
import json
import click
from datetime import datetime
import numpy as np
import os
import torch
from accelerate import Accelerator
from confection import Config, registry
import yaml
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader

from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.utils import SerializableEasyDict as EasyDict, recursive_to
from terrain_diffusion.training.utils import safe_rmtree, set_nested_value
from terrain_diffusion.training.datasets.long_dataset import LongDataset


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True, help="Path to the configuration file")
@click.option("--ckpt", "ckpt_path", type=click.Path(exists=True), required=False, help="Path to a checkpoint to resume training from")
@click.option("--model-ckpt", "model_ckpt_path", type=click.Path(exists=True), required=False, help="Path to a HuggingFace model to initialize weights from")
@click.option("--debug-run", "debug_run", is_flag=True, default=False, help="Run in debug mode which disables wandb and all file saving")
@click.option("--resume", "resume_id", type=str, required=False, help="Wandb run ID to resume")
@click.option("--override", "-o", multiple=True, help="Override config values (format: key.subkey=value)")
@click.pass_context
def main(ctx, config_path, ckpt_path, model_ckpt_path, debug_run, resume_id, override):
    """Main training function for all model types."""
    build_registry()
    
    # Load config (support both .cfg and .yaml)
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = Config(yaml.safe_load(f))
    else:
        config = Config().from_disk(config_path)
    
    # Check for existing checkpoint
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
    
    # Auto-resume W&B run from checkpoint metadata if available
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
    print("Run ID:", wandb.run.id)
    
    # Resolve configuration using registry
    resolved = registry.resolve(config, validate=False)
    
    # Setup state and accelerator
    state = EasyDict({'epoch': 0, 'step': 0, 'seen': 0})
    accelerator = Accelerator(
        mixed_precision=resolved['training']['mixed_precision'],
        gradient_accumulation_steps=resolved['training']['gradient_accumulation_steps'],
        log_with=None
    )
    
    trainer_class = resolved['trainer']
    trainer = trainer_class(config, resolved, accelerator, state)
    
    if model_ckpt_path:
        trainer.load_model_checkpoint(model_ckpt_path)
    
    train_dataset = resolved['train_dataset']
    batch_size = config['training'].get('train_batch_size') or config['training']['batch_size']

    dataloader_kwargs = dict(resolved.get('dataloader_kwargs', {}))
    train_dataloader = DataLoader(
        LongDataset(train_dataset, shuffle=True),
        batch_size=batch_size,
        **dataloader_kwargs
    )

    modules = trainer.get_accelerate_modules()
    prepared_modules = accelerator.prepare(*modules)
    train_dataloader = accelerator.prepare(train_dataloader)
    trainer.set_prepared_modules(prepared_modules)
    
    accelerator.register_for_checkpointing(state)
    for module in trainer.get_checkpoint_modules():
        accelerator.register_for_checkpointing(module)
    
    if ckpt_path:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        accelerator.load_state(ckpt_path)
    
    print(f"Starting training at epoch {state['epoch']}, step {state['step']}")
    
    def save_checkpoint(base_folder_path, overwrite=False):
        """Save training checkpoint."""
        if os.path.exists(base_folder_path + '_checkpoint') and not overwrite:
            strtime = datetime.now().strftime("_%Y%m%d_%H%M%S")
            base_folder_path = f"{base_folder_path}{strtime}"
        elif os.path.exists(base_folder_path + '_checkpoint'):
            safe_rmtree(base_folder_path + '_checkpoint')
        os.makedirs(base_folder_path + '_checkpoint', exist_ok=False)
        
        accelerator.save_state(base_folder_path + '_checkpoint')
        torch.save(trainer.ema.state_dict(), os.path.join(base_folder_path + '_checkpoint', 'phema.pt'))
        
        # Save full train config and model config
        with open(os.path.join(base_folder_path + '_checkpoint', 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save model config
        model = trainer.get_model_for_saving()
        if hasattr(model, 'save_config'):
            model.save_config(os.path.join(base_folder_path + '_checkpoint', 'model_config'))
        
        # Persist W&B run id for seamless resumption
        try:
            with open(os.path.join(base_folder_path + '_checkpoint', 'wandb_run.json'), 'w') as f:
                json.dump({'id': wandb.run.id if wandb.run else None}, f)
        except Exception:
            pass
    
    while state['epoch'] < config['training']['epochs']:
        stats_hist = {}
        progress_bar = tqdm(total=config['training']['epoch_steps'], desc=f"Epoch {state['epoch']}")
        
        train_iter = iter(train_dataloader)
        while progress_bar.n < config['training']['epoch_steps']:
            batch = recursive_to(next(train_iter), device=accelerator.device)
            step_stats = trainer.train_step(state, batch)
            
            # Accumulate statistics
            for k, v in step_stats.items():
                if k not in stats_hist:
                    stats_hist[k] = []
                stats_hist[k].append(v)
            
            # Update progress bar
            postfix = {}
            for k, v in stats_hist.items():
                postfix[k] = f"{np.mean(v[-10:]):.4f}"
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Evaluation
        eval_metrics = {}
        if config.get('evaluation', {}).get('validate_epochs', 0) > 0 and \
           (state['epoch'] + 1) % config['evaluation']['validate_epochs'] == 0:
            eval_metrics = trainer.evaluate()
        elif config['training'].get('eval_epochs', 0) > 0 and \
             (state['epoch'] + 1) % config['training']['eval_epochs'] == 0:
            eval_metrics = trainer.evaluate()
        
        # Logging
        state['epoch'] += 1
        if accelerator.is_main_process:
            log_values = {
                'epoch': state['epoch'],
                'step': state['step'],
                'seen': state['seen']
            }
            # Add training statistics
            for k, v in stats_hist.items():
                if k == 'lr':
                    log_values[f'train/{k}'] = v[-1]
                elif isinstance(v[0], (int, float)):
                    log_values[f'train/{k}'] = np.nanmean(v)
            # Add evaluation metrics
            log_values.update(eval_metrics)
            
            wandb.log(log_values, step=state['epoch'])
            
            # Save checkpoints
            if state['epoch'] % config['logging']['temp_save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/latest", overwrite=True)
            if state['epoch'] % config['logging']['save_epochs'] == 0 and not debug_run:
                save_checkpoint(f"{config['logging']['save_dir']}/{state['seen']//1000}kimg")
    
    print("Training complete!")


if __name__ == '__main__':
    main()
