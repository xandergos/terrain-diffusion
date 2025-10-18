import numpy as np
import torch
import torch.nn as nn
from terrain_diffusion.models.mp_layers import MPConv, mp_leaky_relu, mp_silu, normalize, resample, mp_sum
from terrain_diffusion.models.unet_block import UNetBlock

class MPDiscriminator(nn.Module):
    """
    A magnitude-preserving discriminator that can handle arbitrary sized images 
    and additional floating point conditioning variables.

    Args:
        in_channels (int): Number of input image channels
        additional_vars (int): Number of additional floating point variables (default: 0)
        model_channels (int): Base number of channels (default: 64)
        channel_mults (list): List of channel multipliers for each level (default: [1, 2, 4, 8])
        layers_per_block (int): Number of layers per resolution block (default: 1)
    """
    def __init__(
        self,
        in_channels,
        additional_vars=0,
        model_channels=64,
        channel_mults=[1, 2, 4, 8],
        layers_per_block=1,
        noise_level=0.0,
        channel_means=None,
        channel_stds=None
    ):
        super().__init__()
        
        # Initial conv to get to model_channels
        self.in_conv = MPConv(in_channels + 1, model_channels, kernel=[1, 1])  # +1 for bias channel
        
        self.noise_level = noise_level
        
        if channel_means is None:
            channel_means = torch.zeros(in_channels)
        if channel_stds is None:
            channel_stds = torch.ones(in_channels)
        self.channel_means = torch.as_tensor(channel_means)
        self.channel_stds = torch.as_tensor(channel_stds)
        
        # Main network body
        self.blocks = nn.ModuleList()
        
        # Current number of channels
        cur_channels = model_channels
        
        # Process each resolution level
        for k, mult in enumerate(channel_mults):
            out_channels = model_channels * mult
            
            for i in range(layers_per_block - 1):
                self.blocks.append(UNetBlock(cur_channels if i == 0 else out_channels, 
                                             out_channels, 
                                             emb_channels=0,
                                             mode='enc',
                                             activation='leaky_relu',
                                             resample_mode='keep'))
            
            self.blocks.append(UNetBlock(cur_channels if layers_per_block == 1 else out_channels, 
                                         out_channels, 
                                         emb_channels=0,
                                         mode='enc',
                                         activation='leaky_relu',
                                         resample_mode='down' if k != len(channel_mults) - 1 else 'keep'))
            
            cur_channels = out_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer
        final_input_size = cur_channels # + (16 if additional_vars > 0 else 0)
        self.final_conv = MPConv(final_input_size, 1, kernel=[1, 1])  # 1x1 conv for final output
        self.gain = nn.Parameter(torch.ones([]))

    def forward(self, x, additional_vars=None):
        x = (x - self.channel_means[None, :, None, None].to(dtype=x.dtype, device=x.device)) / self.channel_stds[None, :, None, None].to(dtype=x.dtype, device=x.device)
        
        # Add bias channel
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        
        # Initial convolution
        x = self.in_conv(x)
        
        # Process blocks
        for block in self.blocks:
            x = block(x, emb=None)
        
        # Final classification
        x = self.final_conv(x, gain=self.gain)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
            
        return x

    def norm_weights(self):
        """Normalize all the weights in the model"""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

if __name__ == '__main__':
    # Example usage with image and additional variables
    x = torch.randn(1, 1, 32, 32)
    additional_vars = torch.randn(1, 5)  # 5 additional variables
    d = MPDiscriminator(1, additional_vars=5, model_channels=32, channel_mults=[1, 2, 4, 8])
    print(d(x, additional_vars).shape)