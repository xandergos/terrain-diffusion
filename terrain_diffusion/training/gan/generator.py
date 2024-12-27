from functools import partial
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from terrain_diffusion.training.unet import MPConv, UNetBlock, mp_silu, mp_concat, normalize, resample, mp_sum

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
        
        # Initial projection of latent
        self.initial_conv = MPConv(latent_channels, model_channels * model_channel_mults[0], 
                                 kernel=[stem_width, stem_width], no_padding=True)
        
        # Build decoder blocks
        self.blocks = nn.ModuleList()
        cout = model_channels * model_channel_mults[0]
        
        # For each resolution level
        for level, mult in enumerate(model_channel_mults):
            channels = model_channels * mult
            
            # Replace GeneratorBlock with UNetBlock
            for i in range(layers_per_block - 1):
                self.blocks.append(UNetBlock(cout if i == 0 else channels, channels, 
                                            emb_channels=0,
                                            mode='dec',
                                            resample_mode='keep',
                                            no_padding=True,
                                            **block_kwargs))
                
            self.blocks.append(UNetBlock(cout if layers_per_block == 1 else channels, channels, 
                                         emb_channels=0,
                                         mode='dec',
                                         resample_mode='up' if level != len(model_channel_mults) - 1 else 'keep',
                                         no_padding=True,
                                         **block_kwargs))
            
            cout = channels
        
        # Final output convolution
        self.out_conv = MPConv(cout, out_channels, kernel=[1, 1], no_padding=True)
        self.out_gain = nn.Parameter(torch.ones([]))

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
            x = block(x, emb=None)
            
        return self.out_conv(x, gain=self.out_gain)

    def norm_weights(self):
        """Normalize all the weights in the model."""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

if __name__ == "__main__":
    latent = torch.randn(1, 64, 25, 25)
    model = MPGenerator(latent_channels=64, out_channels=1,
                         model_channels=8,
                         model_channel_mults=[4, 2, 1],
                         layers_per_block=2)
    print(model(latent).shape)