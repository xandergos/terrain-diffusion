from datetime import time
import math
import multiprocessing
import click
import optuna
import torch
from confection import Config, registry
from terrain_diffusion.inference.evaluation.evaluate_fid import evaluate_models_fid
from terrain_diffusion.inference.evaluation.evaluate_fid_consistency import evaluate_models_fid_consistency
from terrain_diffusion.inference.evaluation.utils import create_models_consistency, get_dataloader, create_models
from terrain_diffusion.training.registry import build_registry

gpu_lock = multiprocessing.Lock()

def objective(config, trial):
    """
    Optuna objective function for optimizing FID score.
    
    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter suggestion
        
    Returns:
        float: FID score (lower is better)
    """
    # Hyperparameters to optimize
    main_sigma_rel = trial.suggest_float("main_sigma_rel", 0.015, 0.5)
    intermediate_sigma = trial.suggest_float("intermediate_sigma", 0.01, 20, log=True)
    scale_timesteps = trial.suggest_categorical("scale_timesteps", [True, False])
    
    # Fixed configurations
    main_config = config['config']
    batch_size = config['batch_size']
    device = config['device']
    dtype = config['dtype']
    num_samples = config['num_samples']
    
    # Build and resolve configs
    build_registry()
    main_cfg = Config().from_disk(main_config)
    
    main_resolved = registry.resolve(main_cfg, validate=False)
    
    # Setup models and dataloader
    dataloader = get_dataloader(main_resolved, batch_size)
    model = create_models_consistency(
        main_resolved,
        main_sigma_rel=main_sigma_rel
    )
    
    # Move models to device
    model = model.to(device)
    
    # Evaluate FID
    fid_score = evaluate_models_fid_consistency(
        model=model,
        dataloader=dataloader,
        intermediate_timestep=math.atan(intermediate_sigma / 0.5),
        scale_timesteps=scale_timesteps,
        num_samples=num_samples,
        dtype={'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype]
    )
    
    return fid_score

@click.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), required=True, help="Path to sweep config file")
def sweep_fid(config_path):
    """
    Perform hyperparameter optimization using Optuna to minimize FID score.
    
    The function optimizes the following hyperparameters:
    - main_sigma_rel: EMA sigma relative value for main model
    - intermediate_timestep: Intermediate timestep for consistency evaluation
    - scale_timesteps: Whether to scale timesteps to match input distribution
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