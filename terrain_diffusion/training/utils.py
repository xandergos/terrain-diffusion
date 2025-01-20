from contextlib import contextmanager
import os
import warnings
from easydict import EasyDict
import torch


class SerializableEasyDict(EasyDict):
    def state_dict(self):
        return {k: v for k, v in self.items() if k not in ['state_dict', 'load_state_dict']}

    def load_state_dict(self, state_dict):
        self.update(state_dict)

def recursive_to(obj, device):
    """Recursive move all tensors in dicts/lists to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(item, device) for item in obj)
    return obj

@contextmanager
def temporary_ema_to_model(ema_model):
    """
    Context manager that temporarily transfers EMA parameters to the model
    and restores the original model parameters afterwards.
    
    Usage:
        with ema.temporary_ema_to_model(ema_model):
            # Model temporarily has EMA parameters here
            result = model(input)
        # Original model parameters are restored here
    """
    # Store original parameters
    model_state = {
        name: param.data.to(device='cpu', copy=True)
        for name, param in ema_model.get_params_iter(ema_model.model)
    }
    model_buffers = {
        name: buffer.data.to(device='cpu', copy=True)
        for name, buffer in ema_model.get_buffers_iter(ema_model.model)
    }

    # Copy EMA parameters to model
    ema_model.copy_params_from_ema_to_model()

    try:
        yield
    finally:
        # Restore original parameters
        for (name, param) in ema_model.get_params_iter(ema_model.model):
            param.data.copy_(model_state[name])
        for (name, buffer) in ema_model.get_buffers_iter(ema_model.model):
            buffer.data.copy_(model_buffers[name])

def safe_rmtree(path):
    """Removes a tree but only checkpoint files."""
    for fp in os.listdir(path):
        if os.path.isdir(os.path.join(path, fp)):
            safe_rmtree(os.path.join(path, fp))
        else:
            legal_extensions = ['.bin', '.safetensors', '.pkl', '.pt', '.json', '.md']
            for ext in legal_extensions:
                if fp.endswith(ext):
                    os.remove(os.path.join(path, fp))
                    break
    os.rmdir(path)

def set_nested_value(config, key_path, value, original_override):
    """Set a value in nested config dict, warning if key path doesn't exist."""
    keys = key_path.split('.')
    current = config
    
    # Check if the full path exists before modifying
    try:
        for key in keys[:-1]:
            if key not in current:
                warnings.warn(f"Creating new config section '{key}' from override: {original_override}")
                current[key] = {}
            current = current[key]
        
        if keys[-1] not in current:
            warnings.warn(f"Creating new config value '{key_path}' from override: {original_override}")
        current[keys[-1]] = value
    except (KeyError, TypeError) as e:
        warnings.warn(f"Failed to apply override '{original_override}': {str(e)}")