import numpy as np
import torch
from tqdm import tqdm
from terrain_diffusion.inference.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.unet import EDMUnet2D
from safetensors.torch import load_model
from ema_pytorch import PostHocEMA
from terrain_diffusion.training.datasets.datasets import H5SuperresTerrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

scheduler = EDMDPMSolverMultistepScheduler(0.002, 80.0, 0.5, solver_order=2)


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
        conditional_inputs=[],
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

model_m = get_model(128, 3, '128x3', 0.05, fs='pos').to(device)
model_g = get_model(64, 3, '64x3', 0.05, fs='pos').to(device)

torch.set_num_threads(16)
dataset = H5SuperresTerrainDataset('dataset_full_encoded.h5', 128, [[0.9999, 1]], [480], eval_dataset=False,
                                   latents_mean=[0, 0.07, 0.12, 0.07],
                                   latents_std=[1.4127, 0.8170, 0.8386, 0.8414])
print(len(dataset))

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

torch.set_grad_enabled(False)


for batch in dataloader:
    # Experiment with different guidance scales
    guidance_scales = [1.0] # [1.0, 1.25, 1.5, 1.75, 2.0]
    all_samples = []
    
    images = batch['image'].to(device)
    cond_img = batch.get('cond_img').to(device)
    conditional_inputs = batch.get('cond_inputs')
    images_np = images.squeeze().cpu().numpy()
    
    pred_x0s = {guidance_scale: [] for guidance_scale in guidance_scales}
    for guidance_scale in guidance_scales:
        # Get initial prediction at high noise level
        #samples = torch.randn(images.shape, device=device) * 80
        #all_samples.append(scheduler.precondition_inputs(samples, torch.tensor([80], device=device)).squeeze().cpu().numpy())
        #scaled_input = scheduler.precondition_inputs(samples, torch.tensor([80], device=device))
        #cnoise = scheduler.trigflow_precondition_noise(torch.tensor([80], device=device))
        #x = torch.cat([scaled_input, cond_img], dim=1)
        #if guidance_scale == 1.0:
        #    model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
        #    model_output = model_output_m
        #else:
        #    model_output_m = model_m(x, noise_labels=cnoise, conditional_inputs=[])
        #    model_output_g = model_g(x, noise_labels=cnoise, conditional_inputs=[])
        #
        #    # Combine predictions using autoguidance
        #    model_output = model_output_g + guidance_scale * (model_output_m - model_output_g)
        #
        #pred_x0 = scheduler.precondition_outputs(samples, model_output, torch.tensor([80], device=device))
        #
        ## Plot the initial prediction
        #pred_x0s[guidance_scale].append(pred_x0.squeeze().cpu().numpy())
            
        scheduler.set_timesteps(15)
        samples = torch.randn(images.shape, device=device) * scheduler.sigmas[0]
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
            if i == 0:
                print((pred_x0 - images).square().mean().item())
            pred_x0s[guidance_scale].append(pred_x0.squeeze().cpu().numpy())
            samples = scheduler.step(model_output, t, samples).prev_sample
            i += 1
            
            all_samples.append(scheduler.precondition_inputs(samples, sigma).squeeze().cpu().numpy())
        pred_x0s[guidance_scale].append(samples.squeeze().cpu().numpy())
    
    # Create a figure with subplots for each guidance scale plus original
    fig, axs = plt.subplots(1, len(pred_x0s) + 1, figsize=(4*(len(pred_x0s) + 1), 4))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.1)  # Adjust margins and make room for slider

    # Find global min and max values across all predictions and original image
    all_values = []
    for predictions in pred_x0s.values():
        all_values.extend([p.min() for p in predictions])
        all_values.extend([p.max() for p in predictions])
    all_values.extend([images[0,0].cpu().numpy().min(), images[0,0].cpu().numpy().max()])
    vmin, vmax = min(all_values), max(all_values)

    # Initialize plots
    images_per_scale = []
    plot_objects = []
    for i, (scale, predictions) in enumerate(pred_x0s.items()):
        images_per_scale.append(predictions)
        im = axs[i].imshow(predictions[-1], cmap='terrain', vmin=vmin, vmax=vmax)
        axs[i].set_title(f'Scale {scale}')
        axs[i].axis('off')
        plot_objects.append(im)
        
    # Add original image
    im = axs[-1].imshow(images[0,0].cpu().numpy(), cmap='terrain', vmin=vmin, vmax=vmax)
    axs[-1].set_title('Original')
    axs[-1].axis('off')

    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    max_steps = max(len(preds) for preds in pred_x0s.values()) - 1
    slider = plt.Slider(ax_slider, 'Step', 0, max_steps, valinit=max_steps, valstep=1)

    def update(val):
        step = int(slider.val)
        for i, predictions in enumerate(images_per_scale):
            if step < len(predictions):
                plot_objects[i].set_data(predictions[step])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()