import os

import torch
from tqdm import tqdm
from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.datasets.datasets import H5LatentsDataset, H5LatentsSimpleDataset
from terrain_diffusion.training.gan.generator import MPGenerator
from ema_pytorch import PostHocEMA
from safetensors.torch import load_model
import matplotlib.pyplot as plt
from terrain_diffusion.training.unet import EDMAutoencoder, EDMUnet2D
from terrain_diffusion.inference.scheduler.functional_dpmsolver import multistep_dpm_solver_second_order_update, dpm_solver_first_order_update, precondition_outputs
from matplotlib.widgets import Slider

def make_weights(size):
    s = size
    mid = (s - 1) / 2
    y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
    epsilon = 1e-3
    distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y - mid).float() / mid, 0, 1)
    distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x - mid).float() / mid, 0, 1)
    return (distance_y * distance_x)[None, None, :, :]

def display_tensor(tensor, title=None, channel=0):
    """
    Display a tensor directly without interactive elements.
    
    Args:
        tensor: PyTorch tensor of shape [1, C, H, W]
        title: Optional title for the plot
        channel: Channel to display (default: 0)
    """
    # Normalize with last channel if available
    if tensor.shape[1] > 1:
        data = tensor[0, channel:channel+1] / tensor[0, -1:]
    else:
        data = tensor[0, channel:channel+1]
    
    # Convert to numpy for display
    data_np = data.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    img = plt.imshow(data_np)
    plt.colorbar(img)
    
    if title:
        plt.title(title)
    
    plt.show()

def get_model(cls, checkpoint_path, sigma_rel=None, ema_step=None, device='cpu'):
    config_path = os.path.join(checkpoint_path, 'model_config')
    model = cls.from_config(cls.load_config(config_path))

    if sigma_rel is not None:
        # sigma_rels are placeholders since we dont use them
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], checkpoint_folder=os.path.join(checkpoint_path, '..', 'phema')).to(device)
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))

    return model.to(device)

device = 'cuda'
torch.no_grad().__enter__()
gan = get_model(MPGenerator, 'checkpoints/gan/latest_checkpoint', sigma_rel=0.05, device=device)
model = get_model(EDMUnet2D, 'checkpoints/consistency_base-192x3/latest_checkpoint', sigma_rel=0.05, device=device)
autoencoder = EDMAutoencoder.from_pretrained('checkpoints/models/autoencoder').to(device)

dataset = H5LatentsDataset('dataset.h5', 64, [[0.1, 1.0]], [90], [1], eval_dataset=False,
                                   latents_mean=[0, 0, 0, 0],
                                   latents_std=[1, 1, 1, 1],
                                   sigma_data=0.5,
                                   split="train",
                                   beauty_dist=[[1, 1, 1, 1, 1]])

generation_size = 256
window_size = 64
stride = 32

latent = torch.randn(1, 128, 14, 14).to(device)
cond_inputs = gan.raw_forward(latent)
while cond_inputs[0, 0, 0, 0] < 1:
    latent = torch.randn(1, 128, 14, 14).to(device)
    cond_inputs = gan.raw_forward(latent)
cond_inputs = torch.nn.functional.interpolate(cond_inputs, scale_factor=32, mode='nearest')[0]
cond_inputs = cond_inputs[:, :generation_size, :generation_size]

def get_weights(size):
    s = size
    mid = (s - 1) / 2
    y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
    epsilon = 1e-3
    distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y - mid).float() / mid, 0, 1)
    distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x - mid).float() / mid, 0, 1)
    return (distance_y * distance_x)[None, None, :, :]

def process_in_windows(scaled_inputs, cond_inputs, t, window_size=64, stride=32, pbar=True):
    w = get_weights(window_size).to(device)
    output = torch.zeros_like(scaled_inputs)
    output_weights = torch.zeros_like(scaled_inputs)
    for i in tqdm(range(0, generation_size - window_size + 1, stride), disable=not pbar):
        for j in range(0, generation_size - window_size + 1, stride):
            cond_inputs_window = cond_inputs[:, i:i+window_size, j:j+window_size]
            
            thresholded = (torch.amax(cond_inputs_window[5], dim=[-1, -2]) < -1).long()
            
            weighted_cond_inputs = torch.sum(cond_inputs_window * torch.sigmoid(cond_inputs_window[5:]), dim=[-1, -2]) \
                / (torch.sum(torch.sigmoid(cond_inputs_window[5:]), dim=[-1, -2]) + 1e-8)
            mean_elev = cond_inputs_window[0] * 2435 - 2607
            mean_elev = (torch.sign(mean_elev) * torch.sqrt(torch.abs(mean_elev)) + 31.4) / 38.6
            mean_elev = torch.mean(mean_elev)
            mean_temp = weighted_cond_inputs[1] * (1 - thresholded.float())
            std_temp = weighted_cond_inputs[2] * (1 - thresholded.float())
            mean_prec = weighted_cond_inputs[3] * (1 - thresholded.float())
            std_prec = weighted_cond_inputs[4] * (1 - thresholded.float())
            all_water = thresholded
            
            scaled_input_slice = scaled_inputs[:, :, i:i+window_size, j:j+window_size]
            model_output = model(scaled_input_slice,
                                noise_labels=t.to(device).expand(scaled_inputs.shape[0]),
                                conditional_inputs=[mean_elev.to(device).view(1).expand(scaled_inputs.shape[0])])
            output[:, :, i:i+window_size, j:j+window_size] += model_output * w
            output_weights[:, :, i:i+window_size, j:j+window_size] += w
    return output / output_weights

sigma_data = 0.5

pred_x0 = torch.zeros(1, 5, generation_size, generation_size, device=device) * sigma_data
noise = torch.randn_like(pred_x0) * sigma_data
    
i = 0
for sigma in [80]:
    sigma = torch.tensor(sigma, dtype=torch.float32)
    t = torch.atan(sigma / sigma_data)
    sigma, t = sigma.to(device), t.to(device)
    x_t = torch.sin(t) * noise + torch.cos(t) * pred_x0
    cnoise = t.expand(pred_x0.shape[0])
    
    model_output = process_in_windows(x_t / sigma_data, cond_inputs, cnoise)
    
    #plt.imshow(model_output[0, 4].detach().cpu().numpy())
    #plt.show()
    
    pred_x0 = torch.cos(t) * x_t + torch.sin(t) * model_output * sigma_data
    noise = torch.randn_like(noise) * sigma_data
    i += 1

samples = pred_x0 * 2
latent = samples[:, :4]
lowfreq = samples[:, 4:5]
decoded = autoencoder.decode(latent)
residual, watercover = decoded[:, :1], decoded[:, 1:2]
watercover = torch.sigmoid(watercover)
residual = dataset.denormalize_residual(residual, 90)
lowfreq = dataset.denormalize_lowfreq(lowfreq, 90)
residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
decoded_terrain = laplacian_decode(residual, lowfreq)

plt.imshow(decoded_terrain.detach().cpu()[0, 0])
plt.show()