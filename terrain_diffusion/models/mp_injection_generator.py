from functools import partial
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from terrain_diffusion.models.mp_layers import MPConv, MPFourier, mp_sum
from terrain_diffusion.models.unet_block import UNetBlock

class MPInjectionGenerator(ModelMixin, ConfigMixin):
    """
    A GAN generator that upsamples a latent vector to an image while maintaining translation invariance.
    The architecture is based on EDMUnet2D's decoder path but removes padding for translation invariance.
    """
    @register_to_config
    def __init__(
        self,
        latent_channels,
        image_channels,
        model_channels=32,
        model_channel_mults=None,
        layers_per_block=2,
        block_kwargs=None,
        no_padding=True,
        fourier_channels=64,
        emb_channels=None
    ):
        """
        Args:
            latent_channels (int): Number of input channels in latent
            image_channels (int): Number of output channels
            model_channels (int): Base channel count (Default: 32)
            model_channel_mults (list, optional): Channel multipliers for each resolution (Default: [8, 4, 2, 1])
            layers_per_block (int | list[int], optional): Number of conv layers per resolution at each level. If an int
            is provided, it is broadcast to all levels. (Default: 2)
            block_kwargs (dict, optional): Additional args for UNetBlock (Default: {})
        """
        super().__init__()
        
        block_kwargs = block_kwargs or {}
        model_channel_mults = model_channel_mults or [8, 4, 2, 1]
        emb_channels = emb_channels or model_channels * max(model_channel_mults)
        
        # Normalize layers_per_block to a per-level list
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(model_channel_mults)
        elif not isinstance(layers_per_block, (list, tuple)):
            raise TypeError("layers_per_block must be an int or a list/tuple of ints")
        if len(layers_per_block) != len(model_channel_mults):
            raise ValueError("layers_per_block must have the same length as model_channel_mults")
        if any((not isinstance(n, int) or n < 1) for n in layers_per_block):
            raise ValueError("Each entry in layers_per_block must be an int >= 1")

        init_channels = model_channels * model_channel_mults[0]
        
        self.fouriers = nn.ModuleList()
        for _ in range(image_channels):
            self.fouriers.append(MPFourier(fourier_channels))
        self.emb_linear = MPConv(image_channels * fourier_channels, emb_channels, kernel=[])
        
        self.initial_skip_conv = MPConv(latent_channels + 1, init_channels, kernel=[1, 1])
        self.image_skip_conv = MPConv(image_channels + 1, model_channels * model_channel_mults[-1], kernel=[1, 1])
        
        # Build decoder blocks
        self.latent_blocks = nn.ModuleList()
        self.image_blocks = nn.ModuleList()
        cout = model_channels * model_channel_mults[0]
        
        # For each resolution level
        for level, mult in enumerate(model_channel_mults):
            channels = model_channels * mult
            
            part = self.latent_blocks if level != len(model_channel_mults) - 1 else self.image_blocks
            
            num_layers = layers_per_block[level]
            for i in range(num_layers - 1):
                part.append(UNetBlock(cout if i == 0 else channels, 
                                             channels, 
                                             emb_channels=emb_channels,
                                             mode='dec',
                                             resample_mode='keep',
                                             no_padding=no_padding,
                                             **block_kwargs))
                
            part.append(UNetBlock(cout if num_layers == 1 else channels, 
                                         channels, 
                                         emb_channels=emb_channels,
                                         mode='dec',
                                         resample_mode='up_bilinear' if level != len(model_channel_mults) - 1 else 'keep',
                                         no_padding=no_padding,
                                         **block_kwargs))
            
            cout = channels
        
        # Final output convolution
        self.out_conv = MPConv(cout, image_channels, kernel=[1, 1], no_padding=no_padding)
        self.out_gain = nn.Parameter(torch.ones([]))

    def forward(self, latents, image, t):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, latent_channels, latent_size, latent_size]
        
        Returns:
            torch.Tensor: Generated image of shape [batch_size, out_channels, out_size, out_size]
        """
        c_emb = torch.log(torch.clip(torch.tan(t) * 0.5, 0.002, 80) / 4)
        emb = torch.cat([fourier(c_emb[:, i]) for i, fourier in enumerate(self.fouriers)], dim=1)
        emb = self.emb_linear(emb)
        
        x = torch.cat([latents, torch.ones_like(latents[:, :1])], dim=1)
        x = self.initial_skip_conv(x)
        for block in self.latent_blocks:
            x = block(x, emb=emb)
        
        image_x = torch.cat([image, torch.ones_like(image[:, :1])], dim=1)
        image_x = self.image_skip_conv(image_x)
        x = mp_sum([x, image_x], w=0.5)
        for block in self.image_blocks:
            x = block(x, emb=emb)
        
        x = self.out_conv(x, gain=self.out_gain)
        anti_padding = (image.shape[2] - x.shape[2]) // 2
        if anti_padding == 0:
            return image * torch.cos(t[..., None, None]) - x * torch.sin(t[..., None, None])
        return image[:, :, anti_padding:-anti_padding, anti_padding:-anti_padding] * torch.cos(t[..., None, None]) - x * torch.sin(t[..., None, None])


    def norm_weights(self):
        """Normalize all the weights in the model."""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

if __name__ == "__main__":
    latent = torch.randn(1, 32, 24, 24)
    image = torch.randn(1, 6, 20, 20)
    t = torch.zeros(1, 6)
    model = MPInjectionGenerator(latent_channels=32, image_channels=6,
                                 model_channels=128,
                                 model_channel_mults=[1, 1],
                                 layers_per_block=[4, 4], 
                                 no_padding=True,
                                 fourier_channels=64,
                                 emb_channels=128)
    print(model(latent, image, t).shape)