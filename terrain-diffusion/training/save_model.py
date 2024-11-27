import os
import click
import torch
from tqdm import tqdm
from diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA

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
    # Load model configuration
    config_path = os.path.join(checkpoint_path, 'model_config')
    model = EDMUnet2D.from_config(EDMUnet2D.load_config(config_path))

    if sigma_rel is not None:
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], update_every=1, checkpoint_every_num_steps=12800,
                         checkpoint_folder=os.path.join(checkpoint_path, '..', 'phema'))
        ema.load_state_dict(torch.load(os.path.join(checkpoint_path, 'phema.pt'), map_location='cpu', weights_only=True))
        ema.gammas = tuple(ema_model.gamma for ema_model in ema.ema_models)
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))

    model.save_pretrained(os.path.join(checkpoint_path, 'saved_model'))
    save_path = os.path.join(checkpoint_path, 'saved_model')
    print(f'Saved model to {save_path}, with EMA step {ema_step} and sigma rel {sigma_rel}.')

if __name__ == '__main__':
    save_model()
