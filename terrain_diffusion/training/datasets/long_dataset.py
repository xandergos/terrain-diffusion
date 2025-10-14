import math
import torch
from torch.utils.data import Dataset


class LongDataset(Dataset):
    def __init__(self, base_dataset, length=10 ** 12, shuffle=True):
        self.base_dataset = base_dataset
        self.length = length
        self.shuffle = shuffle
        self.order = torch.randperm(len(self.base_dataset)) if shuffle else torch.arange(len(self.base_dataset))

    def __len__(self):
        return self.length

    def base_length(self, batch_size):
        return math.ceil(len(self.base_dataset) / batch_size)

    def __getitem__(self, index):
        if index % len(self.base_dataset) == 0 and self.shuffle:
            self.order = torch.randperm(len(self.base_dataset))
        return self.base_dataset[self.order[index % len(self.base_dataset)]]

