import os
from ema_pytorch import PostHocEMA
import torch
from torch.utils.data import DataLoader
from terrain_diffusion.training.datasets.datasets import LongDataset
from ema_pytorch import PostHocEMA

from terrain_diffusion.training.unet import EDMUnet2D

def get_dataloader(main_resolved_cfg, batch_size):
    val_dataset = main_resolved_cfg['val_dataset']
    dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), 
                            batch_size=batch_size,
                            **main_resolved_cfg['dataloader_kwargs'])
    return dataloader

def create_models(main_resolved, guide_resolved=None, main_sigma_rel=0.05, guide_sigma_rel=0.05, 
                  guide_ema_step=None, save_models=False, is_consistency_model=False):
    """
    Create and initialize the main and guidance models from config files.
    
    Args:
        main_resolved (str): Path to main model config file
        guide_resolved (str, optional): Path to guidance model config file. Defaults to None.
        main_sigma_rel (float, optional): EMA sigma_rel for main model. Defaults to 0.05.
        guide_sigma_rel (float, optional): EMA sigma_rel for guidance model. Defaults to 0.05.
        guide_ema_step (int, optional): EMA step for guidance model. Defaults to None.
        save_models (bool, optional): Save the models after loading. Defaults to False.
        is_consistency_model (bool, optional): Is consistency model (default: False).
    
    Returns:
        tuple: (main_model, guide_model)
    """
    
    if is_consistency_model:
        assert guide_resolved is None
    
    # Initialize models
    if is_consistency_model:
        model_m = EDMUnet2D.from_pretrained(main_resolved['model']['main_path'])
    else:
        model_m = main_resolved['model']
    model_g = guide_resolved['model'] if guide_resolved else None
    
    # Apply EMA for main model
    phema_m_dir = f"{main_resolved['logging']['save_dir']}/phema"
    assert os.path.exists(phema_m_dir), f"Error: The phema directory {phema_m_dir} does not exist."
    main_resolved['ema']['checkpoint_folder'] = phema_m_dir
    ema_m = PostHocEMA(model_m, **main_resolved['ema'])
    ema_m.load_state_dict(torch.load(f"{main_resolved['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
    ema_m.synthesize_ema_model(sigma_rel=main_sigma_rel).copy_params_from_ema_to_model()
    
    # Apply EMA for guidance model if it exists
    if guide_resolved:
        phema_g_dir = f"{guide_resolved['logging']['save_dir']}/phema"
        assert os.path.exists(phema_g_dir), f"Error: The phema directory {phema_g_dir} does not exist."
        guide_resolved['ema']['checkpoint_folder'] = phema_g_dir
        ema_g = PostHocEMA(model_g, **guide_resolved['ema'])
        ema_g.load_state_dict(torch.load(f"{guide_resolved['logging']['save_dir']}/latest_checkpoint/phema.pt", weights_only=True))
        ema_g.synthesize_ema_model(sigma_rel=guide_sigma_rel, step=guide_ema_step).copy_params_from_ema_to_model()
    
    if save_models:
        checkpoint_path = main_resolved['logging']['save_dir']
        save_path = os.path.join(checkpoint_path, 'saved_model')
        model_m.save_pretrained(save_path)
        print(f'Saved main model to {save_path}.')

        if guide_resolved:
            checkpoint_path = guide_resolved['logging']['save_dir']
            save_path = os.path.join(checkpoint_path, 'saved_model')
            model_g.save_pretrained(save_path)
            print(f'Saved guidance model to {save_path}.')
    
    # Move models to device and compile
    model_m = torch.compile(model_m)
    if model_g:
        model_g = torch.compile(model_g)
    
    return model_m, model_g