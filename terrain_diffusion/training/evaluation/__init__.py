import torch

def _linear_weight_window(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    s = size
    mid = (s - 1) / 2
    y, x = torch.meshgrid(torch.arange(s, device=device), torch.arange(s, device=device), indexing='ij')
    eps = 1e-3
    wy = 1 - (1 - eps) * torch.clamp(torch.abs(y - mid).to(dtype) / mid, 0, 1)
    wx = 1 - (1 - eps) * torch.clamp(torch.abs(x - mid).to(dtype) / mid, 0, 1)
    return (wy * wx)[None, None]

def _constant_weight_window(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.ones(size, size, device=device, dtype=dtype)


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), max(1, stride)))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts