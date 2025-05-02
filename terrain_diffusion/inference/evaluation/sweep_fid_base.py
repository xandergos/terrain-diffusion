from datetime import time
import multiprocessing
import click
import optuna
import torch
from confection import Config, registry
from terrain_diffusion.inference.evaluation.evaluate_fid_base import evaluate_models_fid, create_models
from terrain_diffusion.inference.evaluation.utils import get_dataloader
from terrain_diffusion.training.registry import build_registry
from terrain_diffusion.training.unet import EDMAutoencoder

gpu_lock = multiprocessing.Lock()

def objective(config, trial, main_resolved, guide_resolved, model_ae):
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
    guide_ema_step = trial.suggest_int("guide_ema_step", 2048, config['max_ema_step'], step=512)
    
    # Fixed configurations
    batch_size = config['batch_size']
    device = config['device']
    dtype = config['dtype']
    num_samples = config['num_samples']
    scheduler_steps = config['scheduler_steps']
    
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
        model_ae=model_ae,
        scheduler=scheduler,
        dataloader=dataloader,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        scheduler_steps=scheduler_steps,
        dtype={'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype]
    )
    
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
    - scheduler_steps: Number of diffusion steps
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
            # Print best trial information if trials exist
            if len(study.trials) > 0:
                best_trial = study.best_trial
                print("\nBest trial so far:")
                print(f"  FID Score: {best_trial.value:.4f}")
                print("  Parameters:")
                for key, value in best_trial.params.items():
                    print(f"    {key}: {value}")
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
    
    print("Loading autoencoder")
    device = config['models']['device']
    ae_path = config['models']['ae_path']
    model_ae = EDMAutoencoder.from_pretrained(ae_path).to(device)
    model_ae = model_ae.to(device)
    print("Autoencoder loaded")
    
    build_registry()
    main_cfg = Config().from_disk(config['models']['main_config'])
    guide_cfg = Config().from_disk(config['models']['guide_config'])
    
    main_resolved = registry.resolve(main_cfg, validate=False)
    guide_resolved = registry.resolve(guide_cfg, validate=False)
    
    study.optimize(lambda trial: objective(config['models'], trial, main_resolved, guide_resolved, model_ae), n_trials=n_trials)

if __name__ == "__main__":
    sweep_fid()