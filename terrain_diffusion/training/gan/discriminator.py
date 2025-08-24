import numpy as np
import torch
import torch.nn as nn
from terrain_diffusion.training.unet import MPConv, UNetBlock, mp_leaky_relu, mp_silu, normalize, resample, mp_sum

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
        additional_vars_hidden=32,
        model_channels=64,
        channel_mults=[1, 2, 4, 8],
        layers_per_block=1,
        noise_level=0.1
    ):
        super().__init__()
        
        # Initial conv to get to model_channels
        self.in_conv = MPConv(in_channels + 1, model_channels, kernel=[1, 1])  # +1 for bias channel
        
        self.noise_level = noise_level
        
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
        
        # Additional variables linear classification head
        self.additional_vars_head = nn.Sequential(
            MPConv(additional_vars, additional_vars_hidden, kernel=[]),
            nn.LeakyReLU(0.2),
            MPConv(additional_vars_hidden, additional_vars_hidden, kernel=[]),
            nn.LeakyReLU(0.2),
            MPConv(additional_vars_hidden, additional_vars_hidden, kernel=[])
        ) if additional_vars > 0 else None
        
        # Final classification layer
        final_input_size = cur_channels + (16 if additional_vars > 0 else 0)
        self.final_conv = MPConv(final_input_size, 1, kernel=[])  # 1x1 conv for final output
        self.gain = nn.Parameter(torch.ones([]))

    def forward(self, x, additional_vars=None):
        x = (x + torch.randn_like(x) * self.noise_level) / np.sqrt(1 + self.noise_level**2)
        
        # Add bias channel
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        
        # Initial convolution
        x = self.in_conv(x)
        
        # Process blocks
        for block in self.blocks:
            x = block(x, emb=None)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # Process additional variables if provided
        if self.additional_vars_head is not None and additional_vars is not None:
            additional_features = self.additional_vars_head(additional_vars)
            x = torch.cat([x, additional_features], dim=1)
        
        # Final classification
        return self.final_conv(x, gain=self.gain)

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