import numpy as np


class SqrtLRScheduler:
    def __init__(self, lr, ref_nimg, warmup_nimg):
        """
        Parameters:
            lr (float): The learning rate.
            ref_nimg (int): The number of images where the lr decay = 1. Before this, there is no decay.
            warmup_nimg (int): The number of images to linearly increase the lr from 0 to lr.
        """
        self.lr = lr
        self.ref_nimg = ref_nimg
        self.warmup_nimg = warmup_nimg

    def get(self, nimg):
        lr = self.lr
        if self.ref_nimg > 0:
            lr /= np.sqrt(max(nimg / self.ref_nimg, 1))
        if self.warmup_nimg > 0:
            lr *= min(nimg / self.warmup_nimg, 1)
        return lr
    
class CosineLRScheduler:
    def __init__(self, lr, ref_nimg, warmup_nimg):
        self.lr = lr
        self.ref_nimg = ref_nimg
        self.warmup_nimg = warmup_nimg

    def get(self, nimg):
        lr = self.lr
        if self.ref_nimg > 0:
            lr *= 0.5 * (1 + np.cos(np.pi * nimg / self.ref_nimg))
        if self.warmup_nimg > 0:
            lr *= min(nimg / self.warmup_nimg, 1)
        return lr
