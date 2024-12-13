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
    return obj
