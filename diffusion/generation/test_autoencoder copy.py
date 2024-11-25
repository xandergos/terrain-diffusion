import torch
from tqdm import tqdm
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import EDMAutoencoder, EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from diffusion.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

scheduler = EDMDPMSolverMultistepScheduler(0.002, 10.0, 0.5)


device = 'cpu'

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
        ema = PostHocEMA(model, sigma_rels=sigma_rels, update_every=1, checkpoint_every_num_steps=12800, allow_different_devices=True,
                        checkpoint_folder=f'checkpoints/diffusion_x8-{tag}/phema').to(device)
        ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/{checkpoint}/phema.pt', map_location='cpu'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()

    return model

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 512, [0.6, 1], '480m', eval_dataset=False,
                                   latents_mean=[0, 0, 0, 0],
                                   latents_std=[1.0, 1.0, 1.0, 1.0])

model_cfg = EDMAutoencoder.load_config('checkpoints/autoencoder_x8-plain_ft/configs/model_config_latest')
model = EDMAutoencoder.from_config(model_cfg).to(device)
load_model(model, 'checkpoints/autoencoder_x8-plain_ft/latest_checkpoint/model.safetensors')

dataloader = DataLoader(dataset, batch_size=1)

torch.set_grad_enabled(False)


for batch in dataloader:
    images = batch['image'].to(device)
    cond_img = batch.get('cond_img').to(device)
    
    with torch.no_grad():
        #enc_mean, enc_logvar = model.preencode(images / 0.5)
        #z = model.postencode(enc_mean, enc_logvar, use_mode=False)
        decoded_x = model.decode(cond_img[:, :, ::8, ::8]) * 0.5
        
    # Plot the decoded image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    vmin = images[0].min().item()
    vmax = images[0].max().item()
    
    ax1.imshow(images[0].permute(1, 2, 0).cpu().numpy(), vmin=vmin, vmax=vmax)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(decoded_x[0].permute(1, 2, 0).cpu().numpy(), vmin=vmin, vmax=vmax)
    ax2.set_title('Decoded')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    