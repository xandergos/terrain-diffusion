from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
from diffusers import ConfigMixin

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def normalize(x, dim=None, eps=1e-4):
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm


def resample(x, mode='keep', factor=2):
    """Resample the input tensor x.
    If mode is 'keep', the input tensor is returned as is.
    If the mode is 'down', the input tensor is downsampled by a factor of 2 by a 1x1 convolution with stride 2.
    If the mode is 'up', the input tensor is upsampled by a factor of 2 by a 2x2 convolution with stride 2 and uniform weight 1.
    """
    if mode == 'keep':
        return x
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, torch.ones([c, 1, 1, 1], device=x.device, dtype=x.dtype), groups=c, stride=factor)
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, torch.ones([c, 1, factor, factor], device=x.device, dtype=x.dtype), groups=c, stride=factor)


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

def mp_hardsilu(x):
    return torch.nn.functional.hardswish(x) / 0.576

def mp_sigmoid(x):
    return torch.sigmoid(x) / 0.208


def mp_sum(args, w=None):
    """
    Magnitude preserving sum of tensors.
    parameters:
        args: list of tensors to sum.
        w: list of weights for each tensor. If None, all tensors are weighted equally. Should sum to 1 to preserve magnitude.
            If a float, the weights are [1-w, w] (a linear interpolation).
    """
    if w is None:
        w = torch.full((len(args),), 1 / len(args), dtype=args[0].dtype, device=args[0].device)
    elif isinstance(w, float):
        w = torch.tensor([1-w, w], dtype=args[0].dtype, device=args[0].device)
    else:
        w = torch.tensor(w, dtype=args[0].dtype, device=args[0].device)
    
    return torch.sum(torch.stack([args * w for args, w in zip(args, w)]), dim=0) / torch.linalg.vector_norm(w)


def mp_concat(args, dim=1, w=None):
    """
    Magnitude preserving concatenation of tensors.
    It should be noted that the concatenated tensors are already magnitude preserving, however the
    contribution of each tensor in subsequent layers is proportional to the number of channels it has.
    This function corrects for this by scaling the tensors to have the same overall magnitude, but
    the contributions of each tensor is the same.
    parameters:
        args: list of tensors to concatenate.
        w: list of weights for each tensor. If None, all tensors are weighted equally. Should sum to 1 to preserve magnitude.
            If a float, the weights are [1-w, w] (a linear interpolation).
    """
    if w is None:
        w = torch.full((len(args),), 1 / len(args), dtype=args[0].dtype, device=args[0].device)
    elif isinstance(w, float):
        w = torch.tensor([1-w, w], dtype=args[0].dtype, device=args[0].device)
    else:
        w = torch.tensor(w, dtype=args[0].dtype, device=args[0].device)
    N = [x.shape[dim] for x in args]
    sum_N = torch.tensor(sum(N), dtype=args[0].dtype, device=args[0].device)
    C = torch.sqrt(sum_N / torch.sum(torch.square(w)))
    return torch.concat([args[i] * (C / np.sqrt(args[i].shape[dim]) * w[i]) for i in range(len(args))], dim=dim)

class MPPositionalEmbedding(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        half_dim = num_channels // 2
        emb = math.log(10) / (half_dim - 1)
        self.register_buffer('freqs', torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, x):
        # Convert input to float32 for higher precision calculations
        y = x.to(torch.float32)
        
        # Compute outer product of input with frequencies
        y = y.outer(self.freqs.to(torch.float32))
        
        # Apply sin and cos, concatenate, and normalize by sqrt(2) to maintain unit variance
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=1) * np.sqrt(2)
        
        # Convert back to original dtype and return
        return y.to(x.dtype)

class MPFourier(nn.Module):
    def __init__(self, num_channels, s=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * s)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        # Convert input to float32 for higher precision calculations
        y = x.to(torch.float32)

        # Compute outer product of input with frequencies
        # This creates a 2D tensor where each row is the input multiplied by a frequency
        y = y.outer(self.freqs.to(torch.float32))

        # Add phase shifts to each element
        y = y + self.phases.to(torch.float32)

        # Apply cosine function to get periodic features
        # Multiply by sqrt(2) to maintain unit variance
        y = y.cos() * np.sqrt(2)

        # Convert back to original dtype and return
        return y.to(x.dtype)

class MPConvResample(nn.Module):
    def __init__(self, resample_mode, kernel, in_channels, out_channels, skip_weight=0.0):
        """Resamples a tensor with MP convolution or transposed convolution.

        Args:
            resample_mode (str): Either 'up' or 'down'.
            kernel (list): Kernel size for the convolution.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.resample_mode = resample_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_weight = skip_weight
        if self.resample_mode == 'down':
            self.weight = nn.Parameter(torch.ones(out_channels, in_channels, *kernel))
        elif self.resample_mode == 'up':
            self.weight = nn.Parameter(torch.ones(in_channels, out_channels, *kernel))
        else:
            raise ValueError("resample_mode must be either 'up' or 'down'")

    def forward(self, x, gain=1):
        # Keep weight in float32 during normalization
        w = self.weight.to(torch.float32)

        # For numerical stability, we normalize the weights to internally have a norm of 1.
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))

        # Weights are already normalized, but this is critical so that gradients are propogated through the normalization.
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)

        upsampled = resample(x, mode=self.resample_mode, factor=2)
        if self.resample_mode == 'down':
            y = torch.nn.functional.conv2d(x, w, stride=2, padding=w.shape[-1]//2-1)
        else:
            y = torch.nn.functional.conv_transpose2d(x, w, stride=2, padding=w.shape[-1]//2-1)
        return mp_sum([y, upsampled], w=self.skip_weight)

    def norm_weights(self):
        with torch.no_grad():
            self.weight.copy_(normalize(self.weight.to(torch.float32)))

class MPConv(nn.Module):
    """
    Magnitude preserving convolution. Conveniently, a kernel of [] is the same as a linear layer.

    This class is a wrapper around the standard Conv2d layer, but with the following modifications:
    - During training, the weight is forced to be normalized to have a magnitude of 1.
    - The weights are then normalized to have a norm of 1 and then scaled to preserve the magnitude of the outputs.

    `gain` is used to scale the output of the layer to potentially provide more control. The default value of 1 keeps output magnitudes similar to input magnitudes.
    """
    def __init__(self, in_channels, out_channels, kernel, groups=1, no_padding=False):
        super().__init__()
        self.out_channels = out_channels
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert groups == 1 or len(kernel) == 2, "Groups other than 1 require a 2D kernel"
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel))
        self.groups = groups
        self.no_padding = no_padding
    def forward(self, x, gain=1):
        # Keep weight in float32 during normalization
        w = self.weight.to(torch.float32)

        # For numerical stability, we normalize the weights to internally have a norm of 1.
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))

        # Weights are already normalized, but this is critical so that gradients are propogated through the normalization.
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)

        # If the kernel is 0D, just do a linear layer
        if w.ndim == 2:
            return nn.functional.linear(x, w)
        
        # Otherwise do a 2D convolution
        assert w.ndim == 4
        return nn.functional.conv2d(x, w, padding=(0 if self.no_padding else w.shape[-1]//2,), groups=self.groups)
    
    def norm_weights(self):
        with torch.no_grad():
            self.weight.copy_(normalize(self.weight.to(torch.float32)))

class MPEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x):
        w = self.weight.to(torch.float32)

        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
                
        w = normalize(w)
        w = w.to(x.dtype)

        return nn.functional.embedding(x, self.weight)
    
    def norm_weights(self):
        with torch.no_grad():
            self.weight.copy_(normalize(self.weight.to(torch.float32)))

class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        mode='enc',
        conv_type='default',
        resample_mode='keep',
        attention=False,
        channels_per_head=64,
        dropout=0,
        res_balance=0.3,
        attn_balance=0.3,
        clip_act=256,
        expansion_factor=1,
        resample_type='pooling',
        resample_filter=4,
        resample_skip_weight=0.5,
        no_padding=False
    ):
        """
        Block module for the EDM2Unet2D architecture.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_channels (int): Number of embedding channels.
            mode (str, optional): Block mode, either 'enc' for encoder or 'dec' for decoder. Default is 'enc'.
            conv_type (str, optional): Convolution type, either 'fused', 'mobile', or 'default'. Default is 'default'.
            resample_mode (str, optional): Resampling mode, either 'keep', 'up', 'down'. Default is 'keep'.
            attention (bool, optional): Whether to use attention mechanism. Default is False.
            channels_per_head (int, optional): Number of channels per attention head. Default is 64.
            dropout (float, optional): Dropout rate. Default is 0.
            res_balance (float, optional): Balance factor for residual connection. Default is 0.3.
            attn_balance (float, optional): Balance factor for attention. Default is 0.3.
            clip_act (int, optional): Activation clipping value. Default is 256.
            expansion_factor (int, optional): Expansion factor for mobile/fused convolutions. Default is 1.
            resample_type (str, optional): Resampling type, either 'pooling' or 'conv'. Default is 'pooling'.
            resample_filter (int, optional): Kernel size for the resampling convolution. Default is 4.
            resample_skip_weight (float, optional): Weight for the skip connection in resampling. Default is 0.75.
        The Block module consists of the following components:
        1. Resampling (if needed)
        2. Skip connection (if needed)
        3. Residual branch with two MPConv layers and embedding injection
        4. Attention mechanism (if enabled)
        5. Activation clipping

        The module uses magnitude-preserving operations throughout to maintain
        consistent signal strength across the network.
        """
        super().__init__()
        self.out_channels = out_channels
        self.mode = mode
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = nn.Parameter(torch.zeros([]))
        
        self.conv_type = conv_type
        if conv_type == 'fused' or conv_type == 'default':
            self.conv_res0 = MPConv(out_channels if mode == 'enc' else in_channels, out_channels * expansion_factor, kernel=[3, 3], no_padding=no_padding)
        elif conv_type == 'mobile':
            self.conv_res0 = nn.ModuleList([
                MPConv(out_channels if mode == 'enc' else in_channels, out_channels * expansion_factor, kernel=[1, 1]),
                MPConv(out_channels * expansion_factor, out_channels * expansion_factor, kernel=[3, 3], groups=out_channels * expansion_factor, no_padding=no_padding),
            ])
            
        self.emb_linear = MPConv(emb_channels, out_channels * expansion_factor, kernel=[]) if emb_channels > 0 else None
        self.conv_res1 = MPConv(out_channels * expansion_factor, out_channels, kernel=[3, 3] if conv_type == 'default' else [1, 1], no_padding=no_padding)
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1]) if self.num_heads != 0 else None
        
        if resample_type == 'conv' and resample_mode != 'keep':
            self.resample = MPConvResample(resample_mode, kernel=[resample_filter, resample_filter], in_channels=in_channels, out_channels=out_channels, 
                                           skip_weight=resample_skip_weight)
        else:
            self.resample = partial(resample, mode=resample_mode)

    def attn(self, x):
        y = self.attn_qkv(x)
        y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
        q, k, v = normalize(y, dim=2).unbind(3)
        y = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.attn_proj(y.reshape(*x.shape))

    def forward(self, x, emb):
        x = self.resample(x)
        if self.mode == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        # Residual branch.
        if self.conv_type == 'mobile':
            y = self.conv_res0[0](mp_silu(x))
            y = self.conv_res0[1](mp_silu(y))
        else:
            y = self.conv_res0(mp_silu(x))
        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            c = c / torch.sqrt(torch.mean(c ** 2, dim=1, keepdim=True) + 1e-8)
            y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        else:
            y = mp_silu(y)
        if self.training and self.dropout != 0:
            y = nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.mode == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum([x, y], w=self.res_balance)

        if self.num_heads != 0:
            x = mp_sum([x, self.attn(x)], w=self.attn_balance)
        
        # Clip activations.
        if self.clip_act is not None:
            x = torch.clip(x, -self.clip_act, self.clip_act)

        return x


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
        fourier_scale=1
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
        """
        super().__init__()        
        self.concat_balance = concat_balance
        
        block_kwargs = block_kwargs or {}
        model_channel_mults = model_channel_mults or [1, 2, 3, 4]
        emb_channels = emb_channels or model_channels * max(model_channel_mults)
        noise_emb_dims = model_channels if noise_emb_dims is None else noise_emb_dims
        attn_resolutions = attn_resolutions or []
        
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
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

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
        attn_resolutions=None,
        midblock_attention=True,
        logvar_channels=128,
        block_kwargs=None,
        conditional_inputs=[],
        latent_channels=None
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
            attn_resolutions (list, optional): The resolutions at which attention is applied. Default is None.
            midblock_attention (bool, optional): Whether to apply attention in the midblock. Default is True.
            logvar_channels (int, optional): The number of channels for uncertainty estimation. Default is 128.
            block_kwargs (dict, optional): Additional keyword arguments for UNetBlock. Default is None.
            conditional_inputs (list, optional): A list of tuples describing additional inputs to the model.
            latent_channels (int, optional): The number of channels in the latent space. Required.
        """
        super().__init__()
        
        block_kwargs = block_kwargs or {}
        model_channel_mults = model_channel_mults or [1, 2, 3, 4]
        attn_resolutions = attn_resolutions or []
        out_channels = out_channels or in_channels
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(model_channel_mults)
        
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
        self.decoder_conv = MPConv(latent_channels + 1, model_channels * model_channel_mults[-1], kernel=[1, 1])
        cout = model_channels * model_channel_mults[-1]  # +1 because we add a ones channel to simulate a bias
        for level, (channels, nb) in reversed(list(enumerate(zip(block_channels, layers_per_block)))):
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

    def preencode(self, x, conditional_inputs=None):
        encodings = self.encoder(x, noise_labels=None, conditional_inputs=conditional_inputs)
        means = encodings[:, :encodings.shape[1] // 2]
        logvars = encodings[:, encodings.shape[1] // 2:]
        return means, logvars
    
    def postencode(self, means, logvars, use_mode=False):
        if use_mode:
            return means
        std = torch.exp(logvars * 0.5)
        eps = torch.randn_like(std)
        return means + eps * std
    
    def decode(self, z, return_logvar=False):
        z = torch.cat([z, torch.ones_like(z[:, :1])], dim=1)  # Add ones channel to simulate bias
        z = self.decoder_conv(z)
        for block in self.decoder:
            z = block(z, None)
        return self.out_conv(z, gain=self.out_gain)

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