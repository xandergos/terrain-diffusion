import torch
from terrain_diffusion.inference.samplers.sampler import Sampler


class StackedSampler(Sampler):
    """
    A sampler that stacks multiple samplers along the batch dimensions.
    """

    def __init__(self, *samplers: list):
        self.samplers = samplers

    def get_region(self, top: int, left: int, bottom: int, right: int, generate=True):
        return torch.concat([sampler.get_region(top, left, bottom, right, generate) for sampler in self.samplers], dim=0)
