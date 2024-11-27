import torch


def constant_label_network_inputs(label):
    def func(tiles_y, tiles_x, batch_idx):
        return {'label_index': torch.full([len(batch_idx)], label, dtype=torch.int64)}
    return func

def base_terrain_nocond(label, image_size):
    def func(tiles_y, tiles_x, batch_idx):
        return {'x': torch.randn(len(batch_idx), 1, image_size, image_size), 
                'conditional_inputs': [torch.tensor(label, dtype=torch.int64).repeat(len(batch_idx)), 
                                       torch.tensor(2, dtype=torch.float32).repeat(len(batch_idx))]}
    return func

