import os
import numpy as np
import torch
from terrain_diffusion.training.diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cpu'

mode = 'consistency'

def get_model(checkpoint_path, ema_step=None, sigma_rel=None, model_config_path=None):
    """
    Load a model from a checkpoint and optionally apply EMA synthesis.
    
    Args:
        checkpoint_path (str): Path to the checkpoint directory.
        ema_step (int, optional): EMA step to use. Defaults to None.
        sigma_rel (float, optional): Sigma relative value. Defaults to None.
        
    Returns:
        EDMUnet2D: The loaded model with optional EMA synthesis applied.
    """
    # Load model configuration
    if model_config_path is None:
        config_path = os.path.join(checkpoint_path, 'model_config')
    else:
        config_path = model_config_path
    model = EDMUnet2D.from_config(EDMUnet2D.load_config(config_path))

    if sigma_rel is not None:
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], update_every=1, checkpoint_every_num_steps=12800,
                         checkpoint_folder=os.path.join(checkpoint_path, '..', 'phema'))
        ema.load_state_dict(torch.load(os.path.join(checkpoint_path, 'phema.pt'), map_location='cpu', weights_only=True))
        ema.gammas = tuple(ema_model.gamma for ema_model in ema.ema_models)
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))

    return model

model = get_model("checkpoints/consistency_x8-64x3/latest_checkpoint", sigma_rel=0.05,
                  model_config_path="checkpoints/consistency_x8-64x3/latest_checkpoint/model_config")

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 128, [0.9999, 1], '480m', eval_dataset=False,
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
        pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
    
        # Plot the predictions and original image side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(pred_x0.squeeze().cpu().numpy())
        ax1.set_title(f'Predicted x0 at t = {t.item():.2f}')
        ax2.imshow(images[0, 0])
        ax2.set_title('Original Image')
        plt.show()
    
        pred_x0 = images
