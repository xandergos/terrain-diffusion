import torch


def constant_label_network_inputs(label):
    def func(tiles_y, tiles_x, batch_idx):
        return {'label_index': torch.full([len(batch_idx)], label, dtype=torch.int64)}
    return func