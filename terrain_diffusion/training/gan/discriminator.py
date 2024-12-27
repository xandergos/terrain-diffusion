import torch
import torch.nn as nn
from terrain_diffusion.training.unet import MPConv, UNetBlock, mp_leaky_relu, mp_silu, normalize, resample, mp_sum

class MPDiscriminator(nn.Module):
    """
    A simplified magnitude-preserving discriminator that can handle arbitrary sized images.
    Uses a series of strided convolutions to progressively downsample the input,
    followed by a global average pooling and final classification layer.

    Args:
        in_channels (int): Number of input channels
        model_channels (int): Base number of channels (default: 64)
        channel_mults (list): List of channel multipliers for each level (default: [1, 2, 4, 8])
    """
    def __init__(
        self,
        in_channels,
        model_channels=64,
        channel_mults=[1, 2, 4, 8],
        layers_per_block=1
    ):
        super().__init__()
        
        # Initial conv to get to model_channels
        self.in_conv = MPConv(in_channels + 1, model_channels, kernel=[3, 3])  # +1 for bias channel
        
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
        
        # Final layers
        self.final_conv = MPConv(cur_channels, 1, kernel=[])  # 1x1 conv for final output
        self.gain = nn.Parameter(torch.zeros([]))

    def forward(self, x):
        # Add bias channel
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        
        # Initial convolution
        x = self.in_conv(x)
        
        # Process blocks
        for i, block in enumerate(self.blocks):
            x = block(x, emb=None)
        
        # Global average pooling
        x = x.mean([2, 3])
        
        # Final classification
        return self.final_conv(x, gain=self.gain)

    def norm_weights(self):
        """Normalize all the weights in the model"""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

if __name__ == '__main__':
    x = torch.randn(1, 1, 32, 32)
    d = MPDiscriminator(1, 32, [1, 2, 4, 8])
    print(d(x).shape)