import os
from ema_pytorch import PostHocEMA
from safetensors.torch import load_model

# Local paths -> HuggingFace repo fallbacks
MODEL_PATHS = {
    "coarse": ("checkpoints/models/diffusion_coarse", "xandergos/TerrainDiffusion-Diffusion-Coarse-128A"),
    "base": ("checkpoints/models/consistency_base-192x3", "xandergos/TerrainDiffusion-Consistency-Base-192x3"),
    "decoder": ("checkpoints/models/consistency_decoder-64x3", "xandergos/TerrainDiffusion-Consistency-Decoder-64x3"),
    "diffusion_base": ("checkpoints/models/diffusion_base-192x3", "xandergos/TerrainDiffusion-Diffusion-Base-192x3"),
    "diffusion_base_guide": ("checkpoints/models/diffusion_base-128x3", "xandergos/TerrainDiffusion-Diffusion-Base-128x3"),
}

def resolve_model_path(user_path: str | None, local_default: str, hf_repo: str) -> str:
    """
    Resolve model path with priority: user override -> local default -> HuggingFace.
    
    Args:
        user_path: User-specified path (local or HF repo). If provided, used as-is.
        local_default: Default local checkpoint path.
        hf_repo: HuggingFace repo ID fallback.
    
    Returns:
        Path to use for model loading (local path or HF repo ID).
    """
    if user_path is not None:
        return user_path
    if os.path.exists(local_default):
        return local_default
    return hf_repo

def get_default_model_path(name: str) -> str:
    """Get default model path, preferring local if available."""
    local, hf = MODEL_PATHS[name]
    return local if os.path.exists(local) else hf

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