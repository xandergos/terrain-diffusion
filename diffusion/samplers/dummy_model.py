import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from tqdm import tqdm
import matplotlib.pyplot as plt


class DummyModel(nn.Module):
    def __init__(self, sigma_data=1.0, image_size=64, in_channels=2, out_channels=2):
        super().__init__()
        self.sigma_data = sigma_data
        self.config = EasyDict(image_size=image_size, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, t):
        t = t.view(-1, 1, 1, 1)
        sigma = torch.exp(4 * t)
        
        c_in = 1 / ((sigma**2 + self.sigma_data**2) ** 0.5)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        
        original_sample = x / c_in
        
        x_mean = torch.mean(original_sample, dim=[2, 3], keepdim=True)
        denoised = self.sigma_data**2 / ((sigma / self.config.image_size)**2 + self.sigma_data**2) * x_mean
        model_output = (denoised - c_skip * original_sample) / c_out
        return model_output
    
if __name__ == "__main__":
    from diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
    scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5, scaling_p=3, scaling_t=0.001, solver_order=1)
        
    model = DummyModel(0.5, 64, 1, 1)
    
    num_batches = 1024*8
    x = torch.randn(num_batches, 1, 64, 64) * scheduler.sigmas[0]
    
    @torch.no_grad()
    def visualize_diffusion(x, model, scheduler, num_steps=50):
        scheduler.set_timesteps(num_steps)
        
        means = []
        timesteps = []
        for t, sigma in tqdm(zip(scheduler.timesteps, scheduler.sigmas), desc="Visualizing diffusion"):
            x_t = scheduler.precondition_inputs(x.view(-1, 1, 64, 64), sigma)
            pred_noise = model(x_t, t.repeat(x.numel() // (64*64)))
            x = scheduler.step(pred_noise, t, x.view(-1, 1, 64, 64)).prev_sample.view(num_batches, 1, 64, 64)
            means.append(x.mean(dim=(1, 2, 3)).tolist())
            timesteps.append(t.item())
        
        return timesteps, means

    timesteps, means = visualize_diffusion(x, model, scheduler)

    print("Std of final means: ", np.std(means[-1]))

    # Plot the results
    # plt.figure(figsize=(10, 6))
    # for i in range(num_batches):
    #     plt.plot(timesteps, [m[i] for m in means])
    # plt.xlabel('t')
    # plt.ylabel('Mean Image Intensity')
    # plt.title('Diffusion Process: Mean Image Intensity over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('diffusion_process_mean_multiple_batches.png')
    # plt.show()

