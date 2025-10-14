import json
import os
import click
from confection import Config
import torch
from tqdm import tqdm
from terrain_diffusion.training.autoencoder.resnet_autoencoder import ResNetAutoencoder
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
from terrain_diffusion.training.gan.generator import MPGenerator
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from diffusers.configuration_utils import ConfigMixin

def load_model_from_checkpoint(checkpoint_path, ema_step=None, sigma_rel=None):
    """
    Load a model from a checkpoint and optionally apply EMA synthesis.
    
    Args:
        checkpoint_path: Path to the checkpoint directory.
        ema_step: EMA step to use.
        sigma_rel: Sigma relative value, can be None.
        
    Returns:
        The loaded model.
    """
    # Load model configuration
    config_path = os.path.join(checkpoint_path, 'model_config')
    with open(os.path.join(config_path, 'config.json'), 'r') as f:
        config = json.load(f)
    if config['_class_name'] == 'EDMUnet2D':
        model = EDMUnet2D.from_config(EDMUnet2D.load_config(config_path))
    elif config['_class_name'] == 'EDMAutoencoder':
        model = EDMAutoencoder.from_config(EDMAutoencoder.load_config(config_path))
    elif config['_class_name'] == 'MPGenerator':
        model = MPGenerator.from_config(MPGenerator.load_config(config_path))
    elif config['_class_name'] == 'ResNetAutoencoder':
        model = ResNetAutoencoder.from_config(ResNetAutoencoder.load_config(config_path))
    else:
        raise ValueError(f'Unknown model class: {config["_class_name"]}')

    if sigma_rel is not None:
        with open(os.path.join(checkpoint_path, 'config.json'), 'r') as f:
            config = json.load(f)
        config['ema']['checkpoint_folder'] = os.path.join(checkpoint_path, '..', 'phema')
        ema = PostHocEMA(model, **config['ema'])
        ema.load_state_dict(torch.load(os.path.join(checkpoint_path, 'phema.pt'), map_location='cpu', weights_only=True))
        ema.gammas = tuple(ema_model.gamma for ema_model in ema.ema_models)
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))

    return model

@click.command()
@click.option('-c', '--checkpoint-path', required=True, help='Path to the checkpoint directory.')
@click.option('-e', '--ema-step', type=int, help='EMA step to use.', default=None)
@click.option('-s', '--sigma-rel', type=float, help='Sigma relative value, can be None.', default=None)
def save_model(checkpoint_path, ema_step, sigma_rel):
    """
    Command-line tool to load a model from a checkpoint and optionally apply EMA synthesis.
    
    Args:
        checkpoint_path: Path to the checkpoint directory.
        ema_step: EMA step to use.
        sigma_rel: Sigma relative value, can be None.
    """
    model = load_model_from_checkpoint(checkpoint_path, ema_step, sigma_rel)
    model.save_pretrained(os.path.join(checkpoint_path, 'saved_model'))
    save_path = os.path.join(checkpoint_path, 'saved_model')
    print(f'Saved model to {save_path}, with EMA step {ema_step} and sigma rel {sigma_rel}.')

if __name__ == '__main__':
    save_model()
