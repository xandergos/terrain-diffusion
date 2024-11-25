import torch
from tqdm import tqdm
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from diffusion.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_model(channels, layers, tag, sigma_rel=None, ema_step=None, fs=1.0, checkpoint='latest_checkpoint', sigma_rels=[0.04, 0.09]):
    model = EDMUnet2D(
        image_size=512,
        in_channels=5,
        out_channels=1,
        model_channels=channels,
        model_channel_mults=[1, 2, 3, 4],
        layers_per_block=layers,
        attn_resolutions=[],
        midblock_attention=False,
        concat_balance=0.5,
        conditional_inputs=[],
        fourier_scale=fs
    )
    load_model(model, f'checkpoints/diffusion_x8-{tag}/{checkpoint}/model.safetensors')

    if sigma_rel is not None:
        ema = PostHocEMA(model, sigma_rels=sigma_rels, update_every=1, checkpoint_every_num_steps=12800,
                        checkpoint_folder=f'checkpoints/diffusion_x8-{tag}/phema')
        ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/{checkpoint}/phema.pt', map_location='cpu'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()

    return model

model = get_model(64, 3, '64x3', None, fs='pos')
model.save_pretrained('checkpoints/models/x8_64x3')
