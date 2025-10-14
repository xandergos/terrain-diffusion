import multiprocessing
import click
import optuna
import torch
import gc
from confection import Config, registry
from terrain_diffusion.inference.evaluation.calc_base_fid import calc_base_fid
from terrain_diffusion.inference.evaluation.calc_decoder_fid import calc_decoder_fid
from terrain_diffusion.inference.evaluation.utils import *
from terrain_diffusion.training.registry import build_registry
from torch.utils.data import DataLoader
from terrain_diffusion.training.datasets import LongDataset

gpu_lock = multiprocessing.Lock()


def objective(config, trial, autoencoder=None):
    """
    Optuna objective function for optimizing FID score.
    
    Args:
        config (dict): Configuration dictionary
        trial (optuna.Trial): Optuna trial object for hyperparameter suggestion
        autoencoder (Autoencoder): Autoencoder model (Optional)
        
    Returns:
        float: FID score (lower is better)
    """
    # Hyperparameters to optimize
    guidance_scale = trial.suggest_float("guidance_scale", 1.0, 2.5)
    main_sigma_rel = trial.suggest_float("main_sigma_rel", 0.015, 0.25)
    guide_sigma_rel = trial.suggest_float("guide_sigma_rel", 0.015, 0.25)
    guide_ema_step = trial.suggest_int("guide_ema_step", 2048, config['max_ema_step'], step=512)
    
    # Fixed configurations
    main_config = config['main_config']
    guide_config = config['guide_config']
    batch_size = config['batch_size']
    device = config['device']
    dtype = config['dtype']
    num_samples = config['num_samples']
    scheduler_steps = config['scheduler_steps']
    
    # Build and resolve configs
    build_registry()
    main_cfg = Config().from_disk(main_config)
    guide_cfg = Config().from_disk(guide_config)
    
    main_resolved = registry.resolve(main_cfg, validate=False)
    guide_resolved = registry.resolve(guide_cfg, validate=False)
    
    # Setup models and dataloader
    # Override dataloader kwargs to prevent file handle accumulation
    dataloader_kwargs = main_resolved['dataloader_kwargs'].copy()
    dataloader_kwargs['num_workers'] = 0  # Use main process only
    dataloader_kwargs['persistent_workers'] = False  # Don't persist workers
    dataloader_kwargs['prefetch_factor'] = None
    
    dataloader = DataLoader(LongDataset(main_resolved['val_dataset'], shuffle=True), 
                           batch_size=batch_size,
                           **dataloader_kwargs)
    model_m, model_g = create_models(
        main_resolved,
        guide_resolved,
        main_sigma_rel=main_sigma_rel,
        guide_sigma_rel=guide_sigma_rel,
        guide_ema_step=guide_ema_step
    )
    
    # Move models to device
    model_m = model_m.to(device)
    model_g = model_g.to(device)
    
    scheduler = main_resolved['scheduler']
    
    # Evaluate FID
    if config['eval_type'] == "decoder":
        calc_fid = calc_decoder_fid
    elif config['eval_type'] == "base":
        calc_fid = calc_base_fid
    else:
        raise ValueError(f"Invalid evaluation type: {config['eval_type']}")

    try:
        fid_score = calc_fid(
            model_m=model_m,
            model_g=model_g,
            scheduler=scheduler,
            dataloader=dataloader,
            num_samples=num_samples,
            guidance_scale=guidance_scale,
            scheduler_steps=scheduler_steps,
            dtype={'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype],
            **config['calc_fid_kwargs']
        )
    finally:
        # Clean up resources to prevent file handle accumulation
        del model_m, model_g, scheduler, dataloader
        del main_resolved, guide_resolved, main_cfg, guide_cfg
        del dataloader_kwargs
        torch.cuda.empty_cache()
        gc.collect()
    
    return fid_score

@click.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True, help="Path to sweep config file")
def sweep_fid(config_path):
    """
    Perform hyperparameter optimization using Optuna to minimize FID score.
    
    The function optimizes the following hyperparameters:
    - guidance_scale: Controls the influence of the guidance model
    - main_sigma_rel: EMA sigma relative value for main model
    - guide_sigma_rel: EMA sigma relative value for guidance model
    - guide_ema_step: Number of training steps for guidance model
    """
    config = Config().from_disk(config_path)
    
    if 'storage_dir' in config['sweep'] and config['sweep']['storage_dir'] is not None:
        study_file = config['sweep']['storage_dir']
        storage = f"sqlite:///{study_file}"
    else:
        study_file = None
        storage = None
        
    # Create new study or load existing one
    if storage is not None:
        try:
            study = optuna.create_study(
                study_name="fid_optimization",
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=storage,
                load_if_exists=True
            )
            print(f"Loaded existing study from {study_file}")
        except Exception as e:
            print(f"Creating new study in {study_file}")
            study = optuna.create_study(
                study_name="fid_optimization", 
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=storage
            )
    else:
        study = optuna.create_study(
            study_name="fid_optimization", 
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
    n_trials = config['sweep']['n_trials']
    study.optimize(lambda trial: objective(config['models'], trial), n_trials=n_trials)

if __name__ == "__main__":
    sweep_fid()