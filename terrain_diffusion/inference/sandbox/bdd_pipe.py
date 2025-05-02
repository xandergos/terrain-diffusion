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
diffusion_m = get_model(EDMUnet2D, 'checkpoints/diffusion_base-192x3/latest_checkpoint', sigma_rel=0.05, device=device)
diffusion_g = get_model(EDMUnet2D, 'checkpoints/diffusion_base-128x3/latest_checkpoint', sigma_rel=0.05, device=device)
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

def process_in_windows(scaled_inputs, cond_inputs, t, window_size=64, stride=32):
    w = get_weights(window_size).to(device)
    output = torch.zeros_like(scaled_inputs)
    output_weights = torch.zeros_like(scaled_inputs)
    for i in range(0, generation_size - window_size + 1, stride):
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
            model_output_m = diffusion_m(scaled_input_slice,
                                noise_labels=t.to(device).expand(scaled_inputs.shape[0]),
                                conditional_inputs=[mean_elev.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    mean_temp.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    std_temp.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    mean_prec.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    std_prec.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    all_water.to(device).view(1).expand(scaled_inputs.shape[0])])
            model_output_g = diffusion_g(scaled_input_slice,
                                noise_labels=t.to(device).expand(scaled_inputs.shape[0]),
                                conditional_inputs=[mean_elev.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    mean_temp.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    std_temp.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    mean_prec.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    std_prec.to(device).view(1).expand(scaled_inputs.shape[0]),
                                                    all_water.to(device).view(1).expand(scaled_inputs.shape[0])])
            guidance_scale = 2.0
            model_output = model_output_g + (model_output_m - model_output_g) * guidance_scale
            output[:, :, i:i+window_size, j:j+window_size] += model_output * w
            output_weights[:, :, i:i+window_size, j:j+window_size] += w
    return output / output_weights

scheduler = EDMDPMSolverMultistepScheduler(0.002, 80.0, 0.5)
scheduler.set_timesteps(15)

samples = torch.randn(1, 34, generation_size, generation_size, device=device) * scheduler.sigmas[0]
    
i = 0
for t, sigma in tqdm(zip(scheduler.timesteps, scheduler.sigmas)):
    sigma, t = sigma.to(device), t.to(device)
    scaled_input = scheduler.precondition_inputs(samples, sigma)
    cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1)).expand(samples.shape[0])
    
    model_output = process_in_windows(scaled_input, cond_inputs, cnoise)
    
    if i == 0:
        plt.imshow(model_output[0, 4].detach().cpu().numpy())
        plt.show()
    
    samples = scheduler.step(model_output, t, samples).prev_sample
    i += 1

samples = samples * 2
latent = samples[:, :4]
lowfreq = samples[:, 4:5]
decoded = autoencoder.decode(latent)
residual, watercover = decoded[:, :1], decoded[:, 1:2]
watercover = torch.sigmoid(watercover)
residual = dataset.denormalize_residual(residual, 90)
lowfreq = dataset.denormalize_lowfreq(lowfreq, 90)
residual, lowfreq = laplacian_denoise(residual, lowfreq, 5.0)
decoded_terrain = laplacian_decode(residual, lowfreq)
below_0 = (decoded_terrain < 0).float()

samples_up = torch.nn.functional.interpolate(samples, scale_factor=8, mode='nearest')
img = torch.cat([decoded_terrain, samples_up[:, 5:] * 2, below_0], dim=1)

# Add interactive visualization with slider for channel selection

# Get the number of channels in the image
num_channels = img.shape[1]

# Define landcover names mapping
def get_landcover_name(channel_idx):
    # First 7 channels are not landcover data
    if channel_idx < 7:
        return f"Channel {channel_idx}"
    
    # Map landcover indices to names (0-based index for channels 7+)
    landcover_names = {
        0: "Unknown",
        1: "Shrubs",
        2: "Herbaceous vegetation",
        3: "Cultivated and managed vegetation / agriculture",
        4: "Urban / built up",
        5: "Bare / sparse vegetation",
        6: "Snow and ice",
        7: "Permanent water bodies",
        8: "Herbaceous wetland",
        9: "Moss and lichen",
        10: "Closed forest, evergreen needle leaf",
        11: "Closed forest, evergreen broad leaf",
        12: "Closed forest, deciduous needle leaf",
        13: "Closed forest, deciduous broad leaf",
        14: "Closed forest, mixed",
        15: "Closed forest, not matching any other definitions",
        16: "Open forest, evergreen needle leaf",
        17: "Open forest, evergreen broad leaf",
        18: "Open forest, deciduous needle leaf",
        19: "Open forest, deciduous broad leaf",
        20: "Open forest, mixed",
        21: "Open forest, not matching any other definitions",
        22: "Oceans, seas"
    }
    
    # Get the landcover index (subtract 7 to get 0-based index in the landcover list)
    landcover_idx = channel_idx - 7
    
    if landcover_idx in landcover_names:
        return f"Channel {channel_idx}: {landcover_names[landcover_idx]}"
    else:
        return f"Channel {channel_idx}"

# Create a function to update the plot when slider changes
def plot_with_channel_slider(img_tensor):
    # Convert tensor to numpy for display
    img_np = img_tensor[0].detach().cpu().numpy()
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)  # Make room for slider
    
    # Initial channel to display
    channel = 0
    
    # Display initial image
    im = ax.imshow(img_np[channel])
    plt.colorbar(im)
    ax.set_title(get_landcover_name(channel))
    
    # Create slider axis and slider
    ax_channel = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_channel,
        label='Channel',
        valmin=0,
        valmax=num_channels-1,
        valinit=channel,
        valstep=1
    )
    
    # Update function for slider
    def update(val):
        channel = int(slider.val)
        im.set_data(img_np[channel])
        im.set_clim(img_np[channel].min(), img_np[channel].max())
        ax.set_title(get_landcover_name(channel))
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

# Call the function with your image tensor
plot_with_channel_slider(img)