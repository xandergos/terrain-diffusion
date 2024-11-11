import torch
from tqdm import tqdm
from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from diffusion.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

scheduler = EDMDPMSolverMultistepScheduler(0.002, 80, 0.5)


device = 'cpu'

def get_model(channels, layers, tag, sigma_rel=None, ema_step=None):
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
    load_model(model, f'checkpoints/diffusion_x8-{tag}/latest_checkpoint/model.safetensors')

    if sigma_rel is not None:
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], update_every=1, checkpoint_every_num_steps=12800, allow_different_devices=True,
                        checkpoint_folder=f'checkpoints/diffusion_x8-{tag}/phema')
        ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/latest_checkpoint/phema.pt'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()

    return model

model_m = get_model(64, 3, '64x3', 0.05)
model_g = get_model(64, 2, '64', None)

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 128, [0.9999, 1], '240m', eval_dataset=False,
                                   latents_mean=[0, 0.07, 0.12, 0.07],
                                   latents_std=[1.4127, 0.8170, 0.8386, 0.8414])

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

torch.set_grad_enabled(False)


for batch in dataloader:
    # Experiment with different guidance scales
    guidance_scales = [1.0] # [1.0, 1.25, 1.5, 1.75, 2.0]
    all_samples = []
    
    images = batch['image'].to(device)
    cond_img = batch.get('cond_img').to(device)
    conditional_inputs = batch.get('cond_inputs')
    images_np = images.squeeze().cpu().numpy()
    
    for guidance_scale in guidance_scales:
        scheduler.set_timesteps(20)
        samples = torch.randn(images.shape, device=device) * scheduler.sigmas[0]
        sigma_data = scheduler.config.sigma_data
        
        for t, sigma in tqdm(zip(scheduler.timesteps, scheduler.sigmas)):
            sigma, t = sigma.to(device), t.to(device)
            scaled_input = scheduler.precondition_inputs(samples, sigma)
            cnoise = scheduler.trigflow_precondition_noise(sigma.view(-1))
            
            # Get predictions from both models
            x = torch.cat([scaled_input, cond_img], dim=1)
            if guidance_scale == 1.0:
                model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
                model_output = model_output_m
            else:
                model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
                model_output_g = model_g(x, noise_labels=cnoise, conditional_inputs=[])
            
                # Combine predictions using autoguidance
                model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
            
            samples = scheduler.step(model_output, t, samples).prev_sample
            
        all_samples.append(samples.squeeze().cpu().numpy())
    
    # Plot all guidance scales side by side, plus original
    fig, axes = plt.subplots(1, len(guidance_scales) + 1, figsize=(5*(len(guidance_scales) + 1), 5))
    
    vmin = min(min(s.min() for s in all_samples), images_np.min())
    vmax = max(max(s.max() for s in all_samples), images_np.max())
    
    for i, (samples_np, guidance_scale) in enumerate(zip(all_samples, guidance_scales)):
        axes[i].imshow(samples_np, vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Generated (scale={guidance_scale})')
        axes[i].axis('off')
    
    axes[-1].imshow(images_np, vmin=vmin, vmax=vmax)
    axes[-1].set_title('Original')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()
    