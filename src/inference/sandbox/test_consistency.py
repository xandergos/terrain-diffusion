import numpy as np
import torch
from tqdm import tqdm
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from diffusion.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cpu'

mode = 'consistency'

def get_model(channels, layers, tag, sigma_rel=None, ema_step=None, checkpoint='latest_checkpoint'):
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
        fourier_scale=1.0
    )
    load_model(model, f'checkpoints/{mode}_x8-{tag}/{checkpoint}/model.safetensors')

    if sigma_rel is not None:
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], update_every=1, checkpoint_every_num_steps=12800, allow_different_devices=True,
                        checkpoint_folder=f'checkpoints/{mode}_x8-{tag}/phema')
        ema.load_state_dict(torch.load(f'checkpoints/{mode}_x8-{tag}/{checkpoint}/phema.pt'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()

    return model

model = get_model(64, 3, '64x3', None)

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 128, [0.9999, 1], '240m', eval_dataset=False,
                                   latents_mean=[0, 0.07, 0.12, 0.07],
                                   latents_std=[1.4127, 0.8170, 0.8386, 0.8414])

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

torch.set_grad_enabled(False)

sigma_data = 0.5
for batch in dataloader:
    images = batch['image'].to(device)
    cond_img = batch.get('cond_img').to(device)
    conditional_inputs = batch.get('cond_inputs')
    images_np = images.squeeze().cpu().numpy()
    
    timesteps = torch.as_tensor([np.arctan(80/0.5), 1.1], device=device)
    #timesteps = torch.as_tensor([np.arctan(80/0.5), 1.3, 1.0, 0.7, 0.3, 0.05], device=device)
    
    z = torch.randn_like(images) * sigma_data
    pred_x0 = images
    for t in timesteps:
        x_t = torch.cos(t) * pred_x0 + torch.sin(t) * z
        t = t.view(1).to(device)
        model_input = torch.cat([x_t / 0.5, cond_img], dim=1)
        pred = -model(model_input, noise_labels=t.flatten(), conditional_inputs=[])
        print(torch.cos(t), torch.sin(t))
        pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
    
        # Plot the predictions and original image side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(pred_x0.squeeze().cpu().numpy())
        ax1.set_title(f'Predicted x0 at t = {t.item():.2f}')
        ax2.imshow(images[0, 0])
        ax2.set_title('Original Image')
        plt.show()
    
        pred_x0 = images
