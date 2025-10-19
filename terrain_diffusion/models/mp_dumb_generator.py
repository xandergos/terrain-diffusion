from functools import partial
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from terrain_diffusion.models.mp_layers import MPConv, MPFourier, mp_leaky_relu, mp_silu, mp_sum
from terrain_diffusion.models.unet_block import UNetBlock

class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        dropout=0,
        res_balance=0.3,
        clip_act=256,
        activation='silu'
    ):
        """
        Linear block using 1x1 convolutions (no spatial context).
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_channels (int): Number of embedding channels.
            dropout (float, optional): Dropout rate. Default is 0.
            res_balance (float, optional): Balance factor for residual connection. Default is 0.3.
            clip_act (int, optional): Activation clipping value. Default is 256.
            activation (str, optional): Activation function. Default is 'silu'.
        """
        super().__init__()
        self.out_channels = out_channels
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act
        self.emb_gain = nn.Parameter(torch.zeros([]))
        
        if activation == 'silu':
            self.activation = mp_silu
        elif activation == 'leaky_relu':
            self.activation = partial(mp_leaky_relu, alpha=0.2)
        else:
            raise ValueError(f"Activation {activation} not supported")
        
        # Two 1x1 convolutions for the residual branch
        self.conv_res0 = MPConv(in_channels, out_channels, kernel=[1, 1])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[]) if emb_channels > 0 else None
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[1, 1])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None

    def forward(self, x, emb):
        # Residual branch
        y = self.conv_res0(self.activation(x))
        
        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            c = c / torch.sqrt(torch.mean(c ** 2, dim=1, keepdim=True) + 1e-8)
            y = self.activation(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        else:
            y = self.activation(y)
        
        if self.training and self.dropout != 0:
            y = nn.functional.dropout(y, p=self.dropout)
        
        y = self.conv_res1(y)

        # Skip connection
        if self.conv_skip is not None:
            x = self.conv_skip(x)
        
        # Combine branches
        x = mp_sum([x, y], w=self.res_balance)
        
        # Clip activations
        if self.clip_act is not None:
            x = torch.clip(x, -self.clip_act, self.clip_act)

        return x

class MPDumbGenerator(ModelMixin, ConfigMixin):
    """
    A GAN generator that upsamples a latent vector to an image while maintaining translation invariance.
    The architecture is based on EDMUnet2D's decoder path but removes padding for translation invariance.
    """
    @register_to_config
    def __init__(
        self,
        channels,
        model_channels=32,
        conv_depth=2,
        linear_depth=2,
        no_padding=True,
        fourier_channels=64
    ):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            model_channels (int): Base channel count (Default: 32)
            model_channel_mults (list, optional): Channel multipliers for each resolution (Default: [8, 4, 2, 1])
            layers_per_block (int, optional): Number of conv layers per resolution (Default: 2)
        """
        super().__init__()
        
        self.stem = MPConv(channels + 1, model_channels, kernel=[1, 1])
        
        self.fouriers = nn.ModuleList()
        for _ in range(channels):
            self.fouriers.append(MPFourier(fourier_channels))
        self.emb_linear = MPConv(channels * fourier_channels, model_channels, kernel=[])
        
        self.blocks = nn.ModuleList()
        for _ in range(conv_depth):
            self.blocks.append(UNetBlock(
                model_channels,
                model_channels,
                emb_channels=model_channels,
                mode='dec',
                no_padding=no_padding
            ))
        for _ in range(linear_depth):
            self.blocks.append(LinearBlock(
                model_channels,
                model_channels,
                emb_channels=model_channels
            ))
        
        self.out_conv = MPConv(model_channels, channels, kernel=[1, 1])
        self.out_gain = nn.Parameter(torch.ones([]))

    def forward(self, z, t):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, latent_channels, latent_size, latent_size]
        
        Returns:
            torch.Tensor: Generated image of shape [batch_size, out_channels, out_size, out_size]
        """
        z_init = z
        z = torch.cat([z, torch.ones_like(z[:, :1])], dim=1)
        
        emb = torch.cat([fourier(t[:, i]) for i, fourier in enumerate(self.fouriers)], dim=1)
        emb = self.emb_linear(emb)
        
        x = self.stem(z)
        for block in self.blocks:
            x = block(x, emb=emb)
        x = self.out_conv(x, gain=self.out_gain)
        
        anti_padding = (z_init.shape[2] - x.shape[2]) // 2
        if anti_padding == 0:
            return z_init * torch.cos(t[..., None, None]) - x * torch.sin(t[..., None, None])
        return z_init[:, :, anti_padding:-anti_padding, anti_padding:-anti_padding] * torch.cos(t[..., None, None]) - x * torch.sin(t[..., None, None])
    
    def norm_weights(self):
        """Normalize all the weights in the model."""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

if __name__ == "__main__":
    latent = torch.randn(1, 6, 16, 16)
    model = MPDumbGenerator(channels=6,
                            model_channels=128,
                            conv_depth=3,
                            linear_depth=3,
                            no_padding=True)
    emb = torch.rand(1, 6) * torch.pi/2
    print(model(latent, emb).shape)