import numpy as np
import torch
from tqdm import tqdm
from inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from training.diffusion.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from training.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

scheduler = EDMDPMSolverMultistepScheduler(0.002, 10.0, 0.5, solver_order=2)


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

model_m = get_model(64, 3, '64x3', 0.05, fs='pos').to(device)
model_g = get_model(32, 2, '32x2', 0.05, ema_step=2048*4, fs='pos').to(device)

dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 128, [0.9999, 1], '480m', eval_dataset=False,
                                   latents_mean=[0, 0.07, 0.12, 0.07],
                                   latents_std=[1.4127, 0.8170, 0.8386, 0.8414])

dataloader = DataLoader(dataset, batch_size=1)

torch.set_grad_enabled(False)


for batch in dataloader:
    # Experiment with different guidance scales
    guidance_scales = [2.0] # [1.0, 1.25, 1.5, 1.75, 2.0]
    all_samples = []
    
    images = batch['image'].to(device)
    cond_img = batch.get('cond_img').to(device)
    conditional_inputs = batch.get('cond_inputs')
    images_np = images.squeeze().cpu().numpy()
    
    pred_x0s = []
    for guidance_scale in guidance_scales:
        # Get initial prediction at high noise level
        samples = torch.randn(images.shape, device=device) * 80
        all_samples.append(scheduler.precondition_inputs(samples, torch.tensor([80], device=device)).squeeze().cpu().numpy())
        scaled_input = scheduler.precondition_inputs(samples, torch.tensor([80], device=device))
        cnoise = scheduler.trigflow_precondition_noise(torch.tensor([80], device=device))
        x = torch.cat([scaled_input, cond_img], dim=1)
        if guidance_scale == 1.0:
            model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
            model_output = model_output_m
        else:
            model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
            model_output_g = model_g(x, noise_labels=cnoise, conditional_inputs=[])
        
            # Combine predictions using autoguidance
            model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
        
        pred_x0 = scheduler.precondition_outputs(samples, model_output, torch.tensor([80], device=device))
        
        # Plot the initial prediction
        pred_x0s.append(pred_x0.squeeze().cpu().numpy())
            
        scheduler.set_timesteps(12)
        samples = pred_x0 + torch.randn(images.shape, device=device) * scheduler.sigmas[0]
        sigma_data = scheduler.config.sigma_data
        
        i = 0
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
            
            pred_x0 = scheduler.precondition_outputs(samples, model_output, sigma)
            pred_x0s.append(pred_x0.squeeze().cpu().numpy())
            samples = scheduler.step(model_output, t, samples).prev_sample
            i += 1
            
            all_samples.append(scheduler.precondition_inputs(samples, sigma).squeeze().cpu().numpy())
        pred_x0s.append(samples.squeeze().cpu().numpy())
    
    # Create a figure with sliders to show prediction sequences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(bottom=0.2)  # Make room for sliders
    
    # Calculate min/max for pred_x0s and images
    all_pred_x0s = np.array(pred_x0s)
    pred_vmin = min(all_pred_x0s.min(), images_np.min())
    pred_vmax = max(all_pred_x0s.max(), images_np.max())
    
    # Initial plots
    im1 = ax1.imshow(pred_x0s[0], vmin=pred_vmin, vmax=pred_vmax)
    im2 = ax2.imshow(images_np, vmin=pred_vmin, vmax=pred_vmax)
    
    for ax in [ax1, ax2]:
        ax.axis('off')
    
    ax2.set_title('Original Image')
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    
    slider = plt.Slider(
        ax=ax_slider,
        label='Denoising Step (pred_x0)',
        valmin=0,
        valmax=len(pred_x0s)-1,
        valinit=0,
        valstep=1
    )
    
    # Update function for slider
    def update(val):
        step = int(slider.val)
        im1.set_array(pred_x0s[step])
        guidance_idx = step // (len(pred_x0s) // len(guidance_scales))
        if guidance_idx >= len(guidance_scales):
            guidance_idx = len(guidance_scales) - 1
        ax1.set_title(f'Denoising Step {step} (guidance={guidance_scales[guidance_idx]})')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    update(0)  # Set initial title
    
    plt.show()
    