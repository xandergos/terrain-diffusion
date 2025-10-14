"""EDM UNet2D model implementation."""
import torch
import torch.nn as nn
from diffusers import ConfigMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .mp_layers import (
    MPConv, MPFourier, MPPositionalEmbedding, MPEmbedding,
    mp_silu, mp_sum, mp_concat
)
from .unet_block import UNetBlock


class EDMUnet2D(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels=None,
        model_channels=128,
        model_channel_mults=None,
        layers_per_block=2,
        emb_channels=None,
        noise_emb_dims=None,
        attn_resolutions=None,
        midblock_attention=True,
        concat_balance=0.3,
        logvar_channels=128,
        block_kwargs=None,
        conditional_inputs=[],
        encode_only=False,
        disable_out_gain=False,
        fourier_scale=1,
        n_logvar=1
    ):
        """
        Parameters:
            image_size (int): The size of the input image.
            in_channels (int): The number of channels in the input image. 
                Usually the same as out_channels, unless some channels are used for conditioning.
            out_channels (int): The number of channels in the output image. Default is in_channels.
            label_dim (int, optional): The number of labels. Defaults to 0.
            model_channels (int, optional): The dimension of the model. Default is 128.
            model_channel_mults (list, optional): The channel multipliers for each block. Default is [1, 2, 3, 4].
            layers_per_block (int, optional): The number of layers per block. Default is 2.
            emb_channels (int, optional): The number of channels in the final conditional embedding. Default is model_channels * max(model_channel_mults).
            noise_emb_dims (int, optional): The number of channels in the noise (fourier) embedding. Default is model_channels. 0 will disable the noise input.
            attn_resolutions (list, optional): The resolutions at which attention is applied. Default is None.
            midblock_attention (bool, optional): Whether to apply attention in the midblock. Default is True.
            concat_balance (float, optional): Balance factor for concatenation. Default is 0.3.
            logvar_channels (int, optional): The number of channels for uncertainty estimation. Default is 128.
            conditional_inputs (list, optional): A list of tuples describing additional inputs to the model.
                Each tuple should be in the form (type, x, weight), where type is either 'float' or 'embedding'. 
                x depends on the type:
                'float' indicates the conditional input a single float, and in this case 'x' is the number of fourier channels to use to describe the float.
                'tensor' indicates the conditional input is a tensor, and in this case 'x' is the dimensionality of the tensor.
                'embedding' indicates the conditional input is an embedding of an integer id, and in this case 'x' is the number of possible ids.
                In all cases, 'weight' is a float that describes the weight of the conditional input relative to the other inputs.
                The 'weight' of the noise input is fixed at 1.
            encode_only (bool, optional): Whether to only encode the input and not decode it. Default is False.
            fourier_scale (float, optional): The scale factor for the Fourier embedding. Default is 1. Can also use 'pos' to use a positional embedding.
            n_logvar (int, optional): The number of logvar channels. Default is 1.
        """
        super().__init__()        
        self.concat_balance = concat_balance
        
        block_kwargs = block_kwargs or {}
        model_channel_mults = model_channel_mults or [1, 2, 3, 4]
        emb_channels = emb_channels or model_channels * max(model_channel_mults)
        noise_emb_dims = model_channels if noise_emb_dims is None else noise_emb_dims
        attn_resolutions = attn_resolutions or []
        out_channels = out_channels or in_channels
        
        self.emb_channels = emb_channels
        if noise_emb_dims == 0 and len(conditional_inputs) == 0:
            emb_channels = 0
            self.emb_channels = 0
        
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(model_channel_mults)

        block_channels = [model_channels * m for m in model_channel_mults]

        if fourier_scale == 'pos':
            self.noise_fourier = MPPositionalEmbedding(noise_emb_dims) if noise_emb_dims > 0 else None
        else:
            self.noise_fourier = MPFourier(noise_emb_dims, s=fourier_scale) if noise_emb_dims > 0 else None
        self.noise_linear = MPConv(noise_emb_dims, emb_channels, kernel=[]) if noise_emb_dims > 0 else None
        self.conditional_layers = nn.ModuleList([])
        self.conditional_weights = [1] if self.noise_linear is not None else []
        for type, x, weight  in conditional_inputs:
            if type == 'float':
                self.conditional_layers.append(nn.Sequential(MPFourier(x), MPConv(x, emb_channels, kernel=[])))
            elif type == 'tensor':
                self.conditional_layers.append(MPConv(x, emb_channels, kernel=[]))
            elif type == 'embedding':
                self.conditional_layers.append(MPEmbedding(x, emb_channels))
            self.conditional_weights.append(weight)

        if not disable_out_gain:
            self.out_gain = nn.Parameter(torch.zeros([]))
        else:
            self.out_gain = 1.0

        # Encoder.
        self.enc = nn.ModuleDict()
        cout = in_channels + 1  # +1 because we add a ones channel to simulate a bias
        for level, (channels, nb) in enumerate(zip(block_channels, layers_per_block)):
            res = image_size // 2**level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(cout, cout, emb_channels, mode='enc', resample_mode='down', **block_kwargs)
            for idx in range(nb):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(cin, cout, emb_channels, mode='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, (channels, nb) in reversed(list(enumerate(zip(block_channels, layers_per_block)))):
            res = image_size // 2**level
            if encode_only:
                continue
            if level == len(block_channels) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(cout, cout, emb_channels, mode='dec', attention=midblock_attention, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(cout, cout, emb_channels, mode='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(cout, cout, emb_channels, mode='dec', resample_mode='up', **block_kwargs)
            for idx in range(nb + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(cin, cout, emb_channels, mode='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, out_channels, kernel=[3, 3])
        
        # logvar
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, n_logvar, kernel=[])

    def compute_embeddings(self, noise_labels, conditional_inputs):
        conditional_inputs = conditional_inputs or []
        embeds = []
        if self.noise_linear is not None:
            embeds.append(self.noise_linear(self.noise_fourier(noise_labels)))
        for cond_layer, cond_input in zip(self.conditional_layers, conditional_inputs):
            embeds.append(cond_layer(cond_input))
        if len(embeds) == 0:
            return None
        emb = mp_sum(embeds, self.conditional_weights)
        emb = mp_silu(emb)
        return emb

    def forward(self, x, noise_labels, conditional_inputs, return_logvar=False, precomputed_embeds=None):
        conditional_inputs = conditional_inputs or []
        assert len(conditional_inputs) == len(self.conditional_layers), "Invalid number of conditional inputs"
        
        emb = precomputed_embeds if precomputed_embeds is not None else self.compute_embeddings(noise_labels, conditional_inputs)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)  # Add ones channel to simulate bias
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_concat([x, skips.pop()], w=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
    
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(noise_labels)).reshape(-1, 1, 1, 1)
            return x, logvar
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def norm_weights(self):
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

