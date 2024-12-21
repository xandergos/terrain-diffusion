import torch
from tqdm import tqdm
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

scheduler = EDMDPMSolverMultistepScheduler(0.002, 10.0, 0.5)

device = 'cuda'

def get_model(channels, layers, tag, sigma_rel=None, ema_step=None, fs=1.0, checkpoint='latest_checkpoint'):
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
        conditional_inputs=[("embedding", 2, 0.2)],
        fourier_scale=fs
    )

    if sigma_rel is not None:
        # sigma_rels are placeholders since we dont use them
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], checkpoint_folder=f'checkpoints/diffusion_x8-{tag}/phema').to(device)
        #ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/{checkpoint}/phema.pt', map_location='cpu'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, f'checkpoints/diffusion_x8-{tag}/{checkpoint}/model.safetensors')

    return model

models = [
   # get_model(32, 2, '32x2', 0.05, fs='pos').to(device),
    get_model(32, 2, '32x2', 0.05, fs='pos').to(device)
]

# Enable parallel processing on CPU
torch.set_num_threads(16)

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 128, [[0.9999, 1], [0.0, 0.9999]], [480, 480], eval_dataset=False,
                                   latents_mean=[0, 0, 0, 0],
                                   latents_std=[1, 1, 1, 1])

dataloader = DataLoader(dataset, batch_size=64)

torch.set_grad_enabled(False)

sigma_data = 0.5

# Generate log-spaced sigma values from 0.002 to 80
sigmas = torch.logspace(np.log10(0.002), np.log10(80), 30)
mse_values = {i: {j: [] for j in range(dataloader.batch_size)} for i in range(len(models))}

batch = next(iter(dataloader))
images = batch['image'].to(device)
cond_img = batch.get('cond_img').to(device)

image_std_ratio = torch.std(images, dim=(1, 2, 3), keepdim=True) / sigma_data
step = 0
for sigma in tqdm(sigmas):
    sigma = sigma.to(device)
    sigma = sigma.expand(images.shape[0]).view(-1, 1, 1, 1)
    #sigma = sigma * image_std_ratio
    
    t = torch.atan(sigma / sigma_data)
    cnoise = t.flatten()
    
    # Add noise to images
    noise = torch.randn_like(images) * sigma_data
    x_t = torch.cos(t) * images + torch.sin(t) * noise
    
    # Get model predictions
    scaled_input = x_t / sigma_data
    x = torch.cat([scaled_input, cond_img], dim=1)
    
    for i, model in enumerate(models):
        model_output = model(x, noise_labels=cnoise, conditional_inputs=[torch.zeros(x.shape[0], device=device, dtype=torch.int64)])
        pred_v_t = -sigma_data * model_output
        
        # Calculate MSE for each sample
        v_t = torch.cos(t) * noise - torch.sin(t) * images
        mse = (1 / sigma_data ** 2) * ((pred_v_t - v_t) ** 2).mean(dim=(1,2,3))
        
        # Store MSE for each sample
        for j in range(images.shape[0]):
            mse_values[i][j].append(mse[j].item())
        
    step += 1

image_std_ratio = image_std_ratio.to('cpu')
# Plot MSE vs sigma for each sample
fig, ax = plt.subplots(figsize=(15, 8))
norm = colors.Normalize(vmin=image_std_ratio.min().item(), 
                       vmax=image_std_ratio.max().item())
cmap = plt.cm.viridis

for i in range(len(models)):
    for j in range(images.shape[0]):
        color = cmap(norm(image_std_ratio[j].item()))
        snr = sigmas / max(0.05, image_std_ratio[j].item())
        if j == 0:  # Only add label for first sample to avoid cluttered legend
            ax.loglog(snr, mse_values[i][j], alpha=0.5, color=color, 
                     label=f'Model {i+1}')
        else:
            ax.loglog(snr, mse_values[i][j], alpha=0.5, color=color)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, ax=ax, label='Image STD Ratio')

ax.grid(True)
ax.set_xlabel('Sigma')
ax.set_ylabel('MSE')
ax.set_title('MSE vs Noise Level (Individual Samples)')
ax.legend()
plt.show()

# Plot MSE ratios if multiple models exist
if len(models) > 1:
    fig, ax = plt.subplots(figsize=(15, 8))
    for j in range(images.shape[0]):
        mse_ratios = [mse_values[0][j][k] / mse_values[1][j][k] for k in range(len(sigmas))]
        if j == 0:
            ax.loglog(sigmas, mse_ratios, alpha=0.1, label='Individual Samples')
        else:
            ax.loglog(sigmas, mse_ratios, alpha=0.1)
    
    # Add mean ratio line
    mean_ratios = np.mean([[mse_values[0][j][k] / mse_values[1][j][k] 
                           for k in range(len(sigmas))] 
                          for j in range(images.shape[0])], axis=0)
    ax.loglog(sigmas, mean_ratios, linewidth=2, label='Mean Ratio', linestyle='--')
    
    ax.grid(True)
    ax.set_xlabel('Sigma')
    ax.set_ylabel('MSE Ratio')
    ax.set_title('Relative MSE (Individual Samples)')
    ax.axhline(y=1, color='r', linestyle='--')
    ax.legend()
    plt.show()
