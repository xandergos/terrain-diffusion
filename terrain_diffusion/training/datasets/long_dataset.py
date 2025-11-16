import math
import random
from torch.utils.data import Dataset
import numpy as np


class LongDataset(Dataset):
    def __init__(self, base_dataset, length=10 ** 12, shuffle=True, seed=None):
        self.base_dataset = base_dataset
        self.length = length
        self.shuffle = shuffle
        self.seed = seed or random.randint(0, 2 ** 32 - 1)
        self._internal_epoch = 0
        self._last_position = None
        self._reset_order()

    def __len__(self):
        return self.length

    def base_length(self, batch_size):
        return math.ceil(len(self.base_dataset) / batch_size)
    
    def set_seed(self, seed):
        self.seed = seed
        self._internal_epoch = 0
        self._last_position = None
        self._reset_order()
    
    def _reset_order(self):
        rng = np.random.default_rng((self.seed + self._internal_epoch * 15485863)%(2**32))
        if self.shuffle:
            self.order = rng.permutation(len(self.base_dataset))
        else:
            self.order = np.arange(len(self.base_dataset))
        self.base_seeds = rng.integers(0, 2 ** 32 - 1, size=len(self.base_dataset))
        self._internal_epoch += 1
        self._last_position = None

    def __getitem__(self, index):
        position = index % len(self.base_dataset)
        if self.shuffle and self._last_position is not None and position <= self._last_position:
            self._reset_order()
        self._last_position = position
        
        self.base_dataset.set_seed(int(self.base_seeds[position]))
        return self.base_dataset[int(self.order[position])]

