"""EDM Autoencoder model implementation."""
import torch
import torch.nn as nn
from diffusers import ConfigMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .mp_layers import MPConv, MPFourier
from .unet_block import UNetBlock
from .edm_unet import EDMUnet2D


class EDMAutoencoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels=None,
        model_channels=128,
        model_channel_mults=None,
        layers_per_block=3,
        layers_per_block_decoder=None,
        attn_resolutions=None,
        midblock_attention=True,
        logvar_channels=128,
        block_kwargs=None,
        conditional_inputs=[],
        latent_channels=None,
        n_logvar=1,
        direct_skips=[]
    ):
        """
        EDMAutoencoder class that uses EDMUnet2D for encoding and UNetBlocks for decoding.

        Parameters:
            image_size (int): The size of the input image.
            in_channels (int): The number of channels in the input image.
            out_channels (int, optional): The number of channels in the output image. Default is in_channels.
            model_channels (int, optional): The dimension of the model. Default is 128.
            model_channel_mults (list, optional): The channel multipliers for each block. Default is [1, 2, 3, 4].
            layers_per_block (int, optional): The number of layers per block. Default is 2.
            layers_per_block_decoder (int, optional): The number of layers per block for the decoder. Default is layers_per_block.
            attn_resolutions (list, optional): The resolutions at which attention is applied. Default is None.
            midblock_attention (bool, optional): Whether to apply attention in the midblock. Default is True.
            logvar_channels (int, optional): The number of channels for uncertainty estimation. Default is 128.
            block_kwargs (dict, optional): Additional keyword arguments for UNetBlock. Default is None.
            conditional_inputs (list, optional): A list of tuples describing additional inputs to the model.
            latent_channels (int, optional): The number of channels in the latent space. Required.
            direct_skips (list, optional): A list of channels where direct skips are used. Default is []. These channels will be encoded directly into the latent space.
                This is useful when you want exact precision in the latent space.
        """
        super().__init__()
        
        block_kwargs = block_kwargs or {}
        model_channel_mults = model_channel_mults or [1, 2, 3, 4]
        attn_resolutions = attn_resolutions or []
        out_channels = out_channels or in_channels
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(model_channel_mults)
        layers_per_block_decoder = layers_per_block_decoder or layers_per_block
        if isinstance(layers_per_block_decoder, int):
            layers_per_block_decoder = [layers_per_block_decoder] * len(model_channel_mults)
        assert latent_channels is not None, "latent_channels must be specified"
        
        # Encoder (EDMUnet2D)
        self.encoder = EDMUnet2D(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=latent_channels * 2,
            model_channels=model_channels,
            model_channel_mults=model_channel_mults,
            layers_per_block=layers_per_block,
            emb_channels=0,
            noise_emb_dims=0,
            attn_resolutions=attn_resolutions,
            midblock_attention=midblock_attention,
            logvar_channels=logvar_channels,
            block_kwargs=block_kwargs,
            conditional_inputs=conditional_inputs,
            encode_only=True,
            disable_out_gain=False
        )
        self.encoder.out_gain = nn.Parameter(torch.ones([]))
        
        # Decoder (UNetBlocks)
        block_channels = [model_channels * m for m in model_channel_mults]
        self.decoder = nn.ModuleList()
        self.decoder_conv = MPConv(latent_channels + len(direct_skips) + 1, model_channels * model_channel_mults[-1], kernel=[1, 1])
        cout = model_channels * model_channel_mults[-1]  # +1 because we add a ones channel to simulate a bias
        for level, (channels, nb) in reversed(list(enumerate(zip(block_channels, layers_per_block_decoder)))):
            res = image_size // 2**level
            if level == len(block_channels) - 1:
                self.decoder.append(UNetBlock(cout, cout, 0, mode='dec', attention=midblock_attention, **block_kwargs))
                self.decoder.append(UNetBlock(cout, cout, 0, mode='dec', **block_kwargs))
            else:
                self.decoder.append(UNetBlock(cout, cout, 0, mode='dec', resample_mode='up', **block_kwargs))
            for idx in range(nb + 1):
                cin = cout
                cout = channels
                self.decoder.append(UNetBlock(cin, cout, 0, mode='dec', attention=(res in attn_resolutions), **block_kwargs))
        self.out_conv = MPConv(cout, out_channels, kernel=[3, 3])
        self.out_gain = nn.Parameter(torch.ones([]) * 0.1)
        
        self.logvar = nn.Parameter(torch.zeros([n_logvar]))

    def preencode(self, x, conditional_inputs=None):
        encodings = self.encoder(x, noise_labels=None, conditional_inputs=conditional_inputs)
        means = encodings[:, :encodings.shape[1] // 2]
        logvars = encodings[:, encodings.shape[1] // 2:]
        
        final_means = [means]
        for channel in self.config.direct_skips:
            x_c = x[:, channel:channel+1]
            pooled = torch.nn.functional.adaptive_avg_pool2d(x_c, means.shape[-2:])
            final_means.append(pooled)
        final_means = torch.cat(final_means, dim=1)
        
        final_logvars = torch.cat([logvars, 
                                   torch.full([logvars.shape[0], len(self.config.direct_skips), logvars.shape[2], logvars.shape[3]], 
                                              device=logvars.device, fill_value=-20)], dim=1)
        
        return final_means, final_logvars
    
    def postencode(self, means, logvars, use_mode=False):
        if use_mode:
            return means
        std = torch.exp(logvars * 0.5)
        eps = torch.randn_like(std)
        return means + eps * std
    
    def decode(self, z, include_logvar=False):
        # Extract direct skip channels from end of latent
        direct_channels = z[:, self.config.latent_channels:]
        
        # Process through decoder
        z = torch.cat([z, torch.ones_like(z[:, :1])], dim=1)  # Add ones channel to simulate bias
        z = self.decoder_conv(z)
        for block in self.decoder:
            z = block(z, None)
            
        # Get decoder output
        decoder_out = self.out_conv(z, gain=self.out_gain)
        
        # Create output tensor and insert direct channels at specified indices
        if len(self.config.direct_skips) > 0:
            direct_output = torch.zeros_like(decoder_out)
            direct_output_mask = torch.zeros_like(decoder_out)
            for i, channel_idx in enumerate(self.config.direct_skips):
                direct_output[:, channel_idx:channel_idx+1] = torch.nn.functional.interpolate(direct_channels[:, i:i+1], size=direct_output.shape[-2:], mode='nearest')
                direct_output_mask[:, channel_idx:channel_idx+1] = 1
                
            decoder_out = direct_output_mask * direct_output + (1 - direct_output_mask) * decoder_out
            
        if include_logvar:
            logvar = self.logvar.reshape(-1, 1, 1, 1)
            return decoder_out, logvar
        return decoder_out

    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Logvar(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.logvar_fourier = MPFourier(channels)
        self.logvar_linear = MPConv(channels, 1, kernel=[])

    def forward(self, x):
        return self.logvar_linear(self.logvar_fourier(x))

