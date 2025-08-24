import os
import uuid

import torch
from terrain_diffusion.training.gan.generator import MPGenerator
from ema_pytorch import PostHocEMA
from safetensors.torch import load_model
from infinite_tensor import MemoryTileStore, TensorWindow
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from terrain_diffusion.training.unet import EDMUnet2D
from terrain_diffusion.inference.scheduler.functional_dpmsolver import multistep_dpm_solver_second_order_update, dpm_solver_first_order_update, precondition_outputs

tile_store = MemoryTileStore()

def make_weights(size):
    s = size
    mid = (s - 1) / 2
    y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
    epsilon = 1e-3
    distance_y = 1 - (1 - epsilon) * torch.clamp(torch.abs(y - mid).float() / mid, 0, 1)
    distance_x = 1 - (1 - epsilon) * torch.clamp(torch.abs(x - mid).float() / mid, 0, 1)
    return (distance_y * distance_x)[None, None, :, :]

def create_infinite_tensor_viewer(infinite_tensor, initial_pos=(0, 0), window_size=(512, 512)):
    """
    Creates an interactive viewer for an infinite tensor using direct indexing.
    
    Args:
        infinite_tensor: InfiniteTensor instance
        initial_pos: Initial (x, y) position to view
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)  # Make more room for sliders
    
    # Initial plot
    x, y = initial_pos
    data = infinite_tensor[0, :1, y:y+window_size[1], x:x+window_size[0]] / infinite_tensor[0, -1:, y:y+window_size[1], x:x+window_size[0]]
    img = ax.imshow(data.permute(1, 2, 0).cpu().numpy())
    plt.colorbar(img)  # Add colorbar
    
    # Create sliders for X and Y position and channel
    ax_x = plt.axes([0.2, 0.05, 0.6, 0.03])
    ax_y = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_ch = plt.axes([0.2, 0.15, 0.6, 0.03])
    
    s_x = Slider(ax_x, 'X Position', -1000, 1000, valinit=x, valstep=32)
    s_y = Slider(ax_y, 'Y Position', -1000, 1000, valinit=y, valstep=32)
    s_ch = Slider(ax_ch, 'Channel', 0, infinite_tensor.shape[1]-2, valinit=0, valstep=1)
    
    def update(val):
        x_pos = int(s_x.val)
        y_pos = int(s_y.val)
        ch = int(s_ch.val)
        data = infinite_tensor[0, ch:ch+1, y_pos:y_pos+window_size[1], x_pos:x_pos+window_size[0]].clone() / infinite_tensor[0, -1:, y_pos:y_pos+window_size[1], x_pos:x_pos+window_size[0]]
        data_np = data.permute(1, 2, 0).cpu().numpy()
        
        # Update vmin/vmax based on current channel data
        vmin = data_np.min()
        vmax = data_np.max()
        img.set_clim(vmin, vmax)
        
        img.set_array(data_np)
        fig.canvas.draw_idle()
    
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_ch.on_changed(update)
    
    plt.show()
    
def get_model(cls, checkpoint_path, sigma_rel=None, ema_step=None, device='cpu'):
    config_path = os.path.join(checkpoint_path, 'model_config')
    model = cls.from_config(cls.load_config(config_path))

    if sigma_rel is not None:
        # sigma_rels are placeholders since we dont use them
        ema = PostHocEMA(model, sigma_rels=[0.05, 0.1], checkpoint_folder=os.path.join(checkpoint_path, '..', 'phema')).to(device)
        #ema.load_state_dict(torch.load(f'checkpoints/diffusion_x8-{tag}/{checkpoint}/phema.pt', map_location='cpu'))
        ema.synthesize_ema_model(sigma_rel=sigma_rel, step=ema_step).copy_params_from_ema_to_model()
    else:
        load_model(model, os.path.join(checkpoint_path, 'model.safetensors'))

    return model.to(device)

device = 'cuda'
torch.no_grad().__enter__()
gan = get_model(MPGenerator, 'checkpoints/gan/latest_checkpoint', sigma_rel=0.05, device=device)
diffusion = get_model(EDMUnet2D, 'checkpoints/diffusion_base-192x3/latest_checkpoint', sigma_rel=0.05, device=device)

def base_generator(ctx):
    return torch.randn((1, 128, 512, 512))


latent_tensor = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(1, 128, None, None),
    f=base_generator,
    output_window=TensorWindow((1, 128, 512, 512))
)

def f(ctx, x):
    out = gan.raw_forward(x.to('cuda')).to('cpu')
    return torch.cat([out, torch.ones([1, 1, 252, 252])], dim=1)

gan_tensor = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(1, 7, None, None),
    f=f,
    output_window=TensorWindow((1, 7, 252, 252), window_stride=(1, 7, 240, 240)),
    args=(latent_tensor,),
    args_windows=(TensorWindow((1, 128, 40, 40), window_stride=(1, 128, 30, 30)),),
)
while gan_tensor[0, 0, 1, 1] < 1.5:
    latent_tensor = tile_store.get_or_create(
        uuid.uuid4(),
        shape=(1, 128, None, None),
        f=base_generator,
        output_window=TensorWindow((1, 128, 512, 512))
    )
    gan_tensor = tile_store.get_or_create(
        uuid.uuid4(),
        shape=(1, 7, None, None),
        f=f,
        output_window=TensorWindow((1, 7, 252, 252), window_stride=(1, 7, 240, 240)),
        args=(latent_tensor,),
        args_windows=(TensorWindow((1, 128, 40, 40), window_stride=(1, 128, 30, 30)),),
    )
gan_tensor = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(1, 6, None, None),
    f=lambda ctx, x: x[:, :-1] / x[:, [-1]],
    output_window=TensorWindow((1, 6, 64, 64)),
    args=(gan_tensor,),
    args_windows=(TensorWindow((1, 7, 64, 64)),)
)

cond_inputs = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(1, 6, None, None),
    f=lambda ctx, x: torch.nn.functional.interpolate(x, scale_factor=32, mode='nearest'),
    output_window=TensorWindow((1, 6, 256, 256)),
    args=(gan_tensor,),
    args_windows=(TensorWindow((1, 6, 8, 8)),)
)

# Create Karras noise schedule from sigma_min to sigma_max with rho=7
def get_karras_schedule(sigma_min=0.002, sigma_max=80, rho=7, num_steps=40):
    """
    Creates a noise schedule following Karras et al. 2022.
    Returned sigmas are ordered from highest (sigma_max) to lowest (sigma_min).
    """
    min_inv_rho = sigma_min ** (1/rho)
    max_inv_rho = sigma_max ** (1/rho)
    t = torch.linspace(0, 1, num_steps)
    sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

sigmas = get_karras_schedule(sigma_min=0.002, num_steps=20)

nfe_count = 0
sigma_data = 0.5
def pred_x0(ctx, noisy_latent, sigma_s, cond_inputs):
    noisy_latent = noisy_latent[:, :-1] / noisy_latent[:, [-1]]
    thresholded = (torch.amax(cond_inputs[:, 5], dim=[-1, -2]) < -1).long()
    
    weighted_cond_inputs = torch.sum(cond_inputs * torch.sigmoid(cond_inputs[:, 5:]), dim=[-1, -2]) / (torch.sum(torch.sigmoid(cond_inputs[:, 5:]), dim=[-1, -2]) + 1e-8)
    mean_elev = ((torch.mean(cond_inputs[:, 0], dim=[-1, -2]) * 2435 - 2607) + 2128) / 2353
    mean_temp = weighted_cond_inputs[:, 1] * (1 - thresholded.float())
    std_temp = weighted_cond_inputs[:, 2] * (1 - thresholded.float())
    mean_prec = weighted_cond_inputs[:, 3] * (1 - thresholded.float())
    std_prec = weighted_cond_inputs[:, 4] * (1 - thresholded.float())
    all_water = thresholded
    
    t = torch.atan(sigma_s / sigma_data).flatten()
    
    # Move inputs to CUDA
    noisy_latent_cuda = noisy_latent.cuda()
    t_cuda = t.cuda()
    mean_elev_cuda = mean_elev.cuda()
    mean_temp_cuda = mean_temp.cuda() 
    std_temp_cuda = std_temp.cuda()
    mean_prec_cuda = mean_prec.cuda()
    std_prec_cuda = std_prec.cuda()
    all_water_cuda = all_water.cuda()

    # Run model on GPU
    global nfe_count
    nfe_count += 1
    print("NFE Count:", nfe_count, sigma_s)
    model_output = diffusion(noisy_latent_cuda / torch.sqrt(sigma_data**2 + sigma_s**2),
                           noise_labels=t_cuda.expand(noisy_latent_cuda.shape[0]),
                           conditional_inputs=[mean_elev_cuda, mean_temp_cuda, std_temp_cuda, 
                                            mean_prec_cuda, std_prec_cuda, all_water_cuda])
    
    # Move output back to CPU
    model_output = model_output.cpu()
    
    denoised = precondition_outputs(noisy_latent, model_output, sigma_s, sigma_data, "epsilon")
    return torch.cat([denoised * make_weights(64), make_weights(64)], dim=1)

def denoise_latent_singlestep(ctx, noisy_latent, sigma_t, sigma_s, model_pred):
    noisy_latent = noisy_latent[:, :-1] / noisy_latent[:, [-1]]
    model_pred = model_pred[:, :-1] / model_pred[:, [-1]]
    next_latents = dpm_solver_first_order_update(
        model_pred,
        noisy_latent,
        sigma_t=sigma_t,
        sigma_s=sigma_s
    )
    return torch.cat([next_latents, torch.ones([1, 1, 64, 64])], dim=1)

def denoise_latent_multistep(ctx, noisy_latent, sigma_t, sigma_s0, sigma_s1, model_pred_cur, model_pred_prev):
    noisy_latent = noisy_latent[:, :-1] / noisy_latent[:, [-1]]
    model_pred_cur = model_pred_cur[:, :-1] / model_pred_cur[:, [-1]]
    model_pred_prev = model_pred_prev[:, :-1] / model_pred_prev[:, [-1]]
    next_latents = multistep_dpm_solver_second_order_update(
        [model_pred_cur, model_pred_prev],
        noisy_latent,
        sigma_t=sigma_t,
        sigma_s0=sigma_s0,
        sigma_s1=sigma_s1
    )
    return torch.cat([next_latents, torch.ones([1, 1, 64, 64])], dim=1)

noisy_latents = [tile_store.get_or_create(
    uuid.uuid4(),
    shape=(1, 35, None, None), # Extra channel for weight
    f=lambda ctx: torch.cat([torch.randn([1, 34, 512, 512]) * sigmas[0], torch.ones([1, 1, 512, 512])], dim=1),
    output_window=TensorWindow((1, 35, 512, 512)),
)]
model_preds = []
stride = 32
sigma_s1 = None
i = 0
stride_every = 5
for sigma_t, sigma_s0 in zip(sigmas[1:], sigmas[:-1]):
    model_preds.append(tile_store.get_or_create(
        uuid.uuid4(),
        shape=(1, 35, None, None), # Extra channel for weight
        f=pred_x0,
        output_window=TensorWindow((1, 35, 64, 64), window_stride=(1, 35, stride, stride)),
        args=(noisy_latents[-1], sigma_s0, cond_inputs),
        args_windows=(TensorWindow((1, 35, 64, 64), window_stride=(1, 35, stride, stride)), None, 
                    TensorWindow((1, 6, 64, 64), window_stride=(1, 6, stride, stride)))
    ))
    if len(model_preds) == 1:
        noisy_latents.append(tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, 35, None, None),
            f=denoise_latent_singlestep,
            output_window=TensorWindow((1, 35, 64, 64)),
            args=(noisy_latents[-1], sigma_t, sigma_s0, model_preds[-1]),
            args_windows=(TensorWindow((1, 35, 64, 64)), None, None, TensorWindow((1, 35, 64, 64)))
        ))
    else:
        noisy_latents.append(tile_store.get_or_create(
            uuid.uuid4(),
            shape=(1, 35, None, None),
            f=denoise_latent_multistep,
            output_window=TensorWindow((1, 35, 64, 64)),
            args=(noisy_latents[-1], sigma_t, sigma_s0, sigma_s1, model_preds[-1], model_preds[-2]),
            args_windows=(TensorWindow((1, 35, 64, 64)), None, None, None,
                          TensorWindow((1, 35, 64, 64)),
                          TensorWindow((1, 35, 64, 64)))
        ))
    sigma_s1 = sigma_s0

model_preds.append(tile_store.get_or_create(
    uuid.uuid4(),
    shape=(1, 35, None, None), # Extra channel for weight
    f=pred_x0,
    output_window=TensorWindow((1, 35, 64, 64), window_stride=(1, 35, stride, stride)),
    args=(noisy_latents[-1], sigma_s1, cond_inputs),
    args_windows=(TensorWindow((1, 35, 64, 64), window_stride=(1, 35, stride, stride)), None, 
                  TensorWindow((1, 6, 64, 64), window_stride=(1, 6, stride, stride)))
))

# Example usage:
scaled_model_pred = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(1, 35, None, None),
    f=lambda ctx, x: torch.cat([x[:, :-1] / sigma_data, x[:, [-1]]], dim=1),
    output_window=TensorWindow((1, 35, 64, 64)),
    args=(model_preds[-1],),
    args_windows=(TensorWindow((1, 35, 64, 64)),)
)
create_infinite_tensor_viewer(model_preds[0], window_size=(256, 256), initial_pos=(-8, -8))





