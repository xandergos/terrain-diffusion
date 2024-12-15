from datetime import time
import multiprocessing
import optuna
import torch
from confection import Config, registry
from terrain_diffusion.inference.evaluation.evaluate_fid import (
    get_dataloader, create_models, evaluate_models_fid
)
from terrain_diffusion.training.diffusion.registry import build_registry

gpu_lock = multiprocessing.Lock()

def objective(trial):
    """
    Optuna objective function for optimizing FID score.
    
    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter suggestion
        
    Returns:
        float: FID score (lower is better)
    """
    # Hyperparameters to optimize
    guidance_scale = trial.suggest_float("guidance_scale", 1.0, 2.5)
    main_sigma_rel = trial.suggest_float("main_sigma_rel", 0.015, 0.25)
    guide_sigma_rel = trial.suggest_float("guide_sigma_rel", 0.015, 0.25)
    guide_ema_step = trial.suggest_int("guide_ema_step", 23040, 51712, step=512)
    
    # Fixed configurations
    main_config = "configs/diffusion_x8/diffusion_x8_64-3.cfg"
    guide_config = "configs/diffusion_x8/diffusion_x8_32-3.cfg"
    batch_size = 128
    device = "cuda"
    dtype = torch.float16
    num_samples = 2048  # Minimum required samples for FID
    scheduler_steps = 15
    
    # Build and resolve configs
    build_registry()
    main_cfg = Config().from_disk(main_config)
    guide_cfg = Config().from_disk(guide_config)
    
    main_resolved = registry.resolve(main_cfg, validate=False)
    guide_resolved = registry.resolve(guide_cfg, validate=False)
    
    # Setup models and dataloader
    dataloader = get_dataloader(main_resolved, batch_size)
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
    fid_score = evaluate_models_fid(
        model_m=model_m,
        model_g=model_g,
        scheduler=scheduler,
        dataloader=dataloader,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        scheduler_steps=scheduler_steps,
        dtype=dtype
    )
    
    return fid_score

def sweep_fid():
    """
    Perform hyperparameter optimization using Optuna to minimize FID score.
    
    The function optimizes the following hyperparameters:
    - guidance_scale: Controls the influence of the guidance model
    - main_sigma_rel: EMA sigma relative value for main model
    - guide_sigma_rel: EMA sigma relative value for guidance model
    - scheduler_steps: Number of diffusion steps
    """
    study_file = "fid_optimization.db"
    storage = f"sqlite:///{study_file}"
    
    # Create new study or load existing one
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
        
    n_trials = 100
    study.optimize(objective, n_trials=n_trials)

if __name__ == "__main__":
    sweep_fid()