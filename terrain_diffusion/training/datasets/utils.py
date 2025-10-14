import torch


def stacking_collate_fn(batch):
    batch_size = len(batch[0])
    stacked_tensors = []
    for i in range(batch_size):
        if isinstance(batch[0][i], torch.Tensor):
            stacked_tensor = torch.stack([item[i] for item in batch])
            stacked_tensors.append(stacked_tensor)
        else:
            stacked_tensors.append(stacking_collate_fn([item[i] for item in batch]))
    return tuple(stacked_tensors)

