from functools import partial
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from terrain_diffusion.models.mp_layers import MPConv, mp_silu, mp_concat, normalize, resample, mp_sum, MPConvResample
from terrain_diffusion.models.unet_block import UNetBlock

class MPGenerator(ModelMixin, ConfigMixin):
    """
    A GAN generator that upsamples a latent vector to an image while maintaining translation invariance.
    The architecture is based on EDMUnet2D's decoder path but removes padding for translation invariance.
    """
    @register_to_config
    def __init__(
        self,
        latent_channels,
        out_channels,
        model_channels=32,
        model_channel_mults=None,
        layers_per_block=2,
        block_kwargs=None,
        stem_width=7,
        stem_channels=None,
        no_padding=True
    ):
        """
        Args:
            latent_channels (int): Number of input channels in latent
            out_channels (int): Number of output channels
            model_channels (int): Base channel count (Default: 32)
            model_channel_mults (list, optional): Channel multipliers for each resolution (Default: [8, 4, 2, 1])
            layers_per_block (int, optional): Number of conv layers per resolution (Default: 2)
            block_kwargs (dict, optional): Additional args for UNetBlock (Default: {})
        """
        super().__init__()
        
        block_kwargs = block_kwargs or {}
        model_channel_mults = model_channel_mults or [8, 4, 2, 1]
        
        init_channels = stem_channels or model_channels * model_channel_mults[0]
        
        # Initial stem convolution for spatial context
        self.stem_conv = MPConv(latent_channels, init_channels, 
                              kernel=[stem_width, stem_width], no_padding=no_padding)
        self.initial_skip_conv = MPConv(latent_channels, init_channels, kernel=[1, 1])
        
        # Upsampling after stem
        self.initial_up = MPConvResample('up', kernel=[4, 4], 
                                       in_channels=init_channels,
                                       out_channels=init_channels,
                                       skip_weight=0.0)
        
        # Build decoder blocks
        self.blocks = nn.ModuleList()
        cout = stem_channels or model_channels * model_channel_mults[0]
        
        # For each resolution level
        for level, mult in enumerate(model_channel_mults):
            channels = model_channels * mult
            
            # Replace GeneratorBlock with UNetBlock
            for i in range(layers_per_block - 1):
                self.blocks.append(UNetBlock(cout if i == 0 else channels, 
                                             channels, 
                                             emb_channels=0,
                                             mode='dec',
                                             resample_mode='keep',
                                             no_padding=no_padding,
                                             #activation='leaky_relu',
                                             #resample_type='conv',
                                             #resample_filter=2,
                                             #resample_skip_weight=0.7,
                                             **block_kwargs))
                
            self.blocks.append(UNetBlock(cout if layers_per_block == 1 else channels, 
                                         channels, 
                                         emb_channels=0,
                                         mode='dec',
                                         resample_mode='up_bilinear' if level != len(model_channel_mults) - 1 else 'keep',
                                         no_padding=no_padding,
                                         #activation='leaky_relu',
                                         #resample_type='conv',
                                         #resample_filter=2,
                                         #resample_skip_weight=0.7,
                                         **block_kwargs))
            
            cout = channels
        
        # Final output convolution
        self.out_conv = MPConv(cout, out_channels + 1, kernel=[1, 1], no_padding=no_padding)
        self.out_gain = nn.Parameter(torch.ones([]))

    def raw_forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, latent_channels, latent_size, latent_size]
        
        Returns:
            torch.Tensor: Generated image of shape [batch_size, out_channels, out_size, out_size]
        """
        # Apply stem conv
        stem = self.stem_conv(z)
        
        # Handle skip connection
        init_skip = self.initial_skip_conv(z)
        if init_skip.shape != stem.shape:
            diff_h = init_skip.shape[2] - stem.shape[2]
            diff_w = init_skip.shape[3] - stem.shape[3]
            start_h = diff_h // 2
            start_w = diff_w // 2
            init_skip = init_skip[:, :, start_h:start_h + stem.shape[2], start_w:start_w + stem.shape[3]]
        x = mp_sum([stem, init_skip], w=0.5)
        
        for block in self.blocks:
            x = block(x, emb=None)
                
        x = self.out_conv(x, gain=self.out_gain)
        return x
    
    def forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, latent_channels, latent_size, latent_size]
        
        Returns:
            torch.Tensor: Generated image of shape [batch_size, out_channels, out_size, out_size]
        """
        x = self.raw_forward(z)
        return torch.cat([x[:, :1], x[:, 1:-1] * torch.sigmoid(x[:, -1:])], dim=1)

    def norm_weights(self):
        """Normalize all the weights in the model."""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

# Note: GBlock and ResNetGenerator classes were removed as they depended on 
# ResBlock from the deleted discriminator_basic module and were not used anywhere.


if __name__ == "__main__":
    latent = torch.randn(1, 256, 20, 20)
    model = MPGenerator(latent_channels=256, out_channels=5,
                         model_channels=16,
                         model_channel_mults=[64, 32, 16, 8, 4, 2, 1],
                         layers_per_block=2, no_padding=True,
                         stem_width=7)
    print(model(latent).shape)