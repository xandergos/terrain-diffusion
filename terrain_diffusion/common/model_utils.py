import os
from ema_pytorch import PostHocEMA
from safetensors.torch import load_model

def get_model(cls, checkpoint_path, sigma_rel=None, ema_step=None, device='cpu'):
    config_path = os.path.join(checkpoint_path, 'model_config')
    model = cls.from_config(cls.load_config(config_path))
    if sigma_rel is not None:
        # sigma_rels are placeholders since we dont use them
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], checkpoint_folder=os.path.join(checkpoint_path, '..', 'phema')).to(device)
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))
    return model.to(device)