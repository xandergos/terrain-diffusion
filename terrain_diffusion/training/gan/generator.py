from functools import partial
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from ..unet import MPConv, UNetBlock, mp_silu, mp_concat, normalize

class GANGenerator(ModelMixin, ConfigMixin):
    """
    A GAN generator that upsamples a latent vector to an image while maintaining translation invariance.
    The architecture is based on EDMUnet2D's decoder path but removes padding for translation invariance.
    """
    @register_to_config
    def __init__(
        self,
        latent_size,
        latent_channels,
        out_channels,
        model_channels=32,
        model_channel_mults=None,
        layers_per_block=2,
        block_kwargs=None,
    ):
        """
        Args:
            latent_size (int): Size of input latent (will be upsampled 2x per block)
            latent_channels (int): Number of input channels in latent
            out_channels (int): Number of output channels
            model_channels (int): Base channel count (Default: 32)
            model_channel_mults (list, optional): Channel multipliers for each resolution (Default: [8, 4, 2, 1])
            layers_per_block (int, optional): Number of conv layers per resolution (Default: 2)
            block_kwargs (dict, optional): Additional args for UNetBlock (Default: {})
        """
        super().__init__()
        
        self.latent_size = latent_size
        self.out_size = latent_size * 64  # 64x upsampling
        
        block_kwargs = block_kwargs or {}
        model_channel_mults = model_channel_mults or [8, 4, 2, 1]  # Reversed from encoder since we're going up
        attn_resolutions = attn_resolutions or []
        
        # Initial projection of latent
        self.initial_conv = MPConv(latent_channels, model_channels * model_channel_mults[0], 
                                 kernel=[1, 1], no_padding=True)
        
        # Build decoder blocks
        self.blocks = nn.ModuleList()
        cout = model_channels * model_channel_mults[0]
        
        # For each resolution level
        for level, mult in enumerate(model_channel_mults):
            channels = model_channels * mult
                
            # Add specified number of blocks at this resolution
            for _ in range(layers_per_block):
                cin = cout
                cout = channels
                self.blocks.append(UNetBlock(cin, cout, 0, mode='dec',
                                             attention=False,
                                             no_padding=True,
                                             **block_kwargs))
            
            # Add upsampling block if not at final resolution
            if level != len(model_channel_mults) - 1:
                self.blocks.append(UNetBlock(cout, cout, 0, mode='dec',
                                             resample_mode='up',
                                             no_padding=True,
                                             **block_kwargs))
        
        # Final output convolution
        self.out_conv = MPConv(cout, out_channels, kernel=[1, 1], no_padding=True)
        self.out_gain = nn.Parameter(torch.zeros([]))

    def forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, latent_channels, latent_size, latent_size]
        
        Returns:
            torch.Tensor: Generated image of shape [batch_size, out_channels, out_size, out_size]
        """
        x = self.initial_conv(z)
        
        for block in self.blocks:
            x = block(x, None)  # None for emb since we don't use conditioning
            
        return self.out_conv(x, gain=self.out_gain)

    def norm_weights(self):
        """Normalize all the weights in the model."""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()
