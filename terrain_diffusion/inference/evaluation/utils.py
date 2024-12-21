import os
import torch
from torch.utils.data import DataLoader
from terrain_diffusion.training.datasets.datasets import LongDataset
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.unet import EDMUnet2D

def get_dataloader(main_resolved_cfg, batch_size):
    val_dataset = main_resolved_cfg['val_dataset']
    dataloader = DataLoader(LongDataset(val_dataset, shuffle=True), 
                            batch_size=batch_size,
                            **main_resolved_cfg['dataloader_kwargs'])
    return dataloader
