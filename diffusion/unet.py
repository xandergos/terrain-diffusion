import numpy as np
import torch
import torch.nn as nn
from diffusers import ConfigMixin


def normalize(x, dim=None, eps=1e-4):
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm


def resample(x, mode='keep'):
    """Resample the input tensor x.
    If mode is 'keep', the input tensor is returned as is.
    If the mode is 'down', the input tensor is downsampled by a factor of 2 by a 1x1 convolution with stride 2.
    If the mode is 'up', the input tensor is upsampled by a factor of 2 by a 2x2 convolution with stride 2 and uniform weight 1.
    """
    if mode == 'keep':
        return x
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, torch.ones([c, 1, 1, 1], device=x.device, dtype=x.dtype), groups=c, stride=2)
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, torch.ones([c, 1, 2, 2], device=x.device, dtype=x.dtype), groups=c, stride=2)


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


def mp_sum(args, w=None):
    """
    Magnitude preserving sum of tensors.
    parameters:
        args: list of tensors to sum.
        w: list of weights for each tensor. If None, all tensors are weighted equally. Should sum to 1 to preserve magnitude.
            If a float, the weights are [1-w, w] (a linear interpolation).
    """
    if w is None:
        w = torch.full((len(args),), 1 / len(args))
    elif isinstance(w, float):
        w = torch.tensor([1-w, w])
    else:
        w = torch.tensor(w)
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
        w = torch.full(len(args), 1 / len(args))
    elif isinstance(w, float):
        w = torch.tensor([1-w, w])
    else:
        w = torch.tensor(w)
    N = [x.shape[dim] for x in args]
    C = np.sqrt(sum(N) / torch.sum(torch.square(w)))
    return torch.concat([args[i] * (C / np.sqrt(args[i].shape[dim]) * w[i]) for i in range(len(args))], dim=dim)


class MPFourier(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels))
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


class MPConv(nn.Module):
    """
    Magnitude preserving convolution. Conveniently, a kernel of [] is the same as a linear layer.

    This class is a wrapper around the standard Conv2d layer, but with the following modifications:
    - During training, the weight is forced to be normalized to have a magnitude of 1.
    - The weights are then normalized to have a norm of 1 and then scaled to preserve the magnitude of the outputs.

    `gain` is used to scale the output of the layer to potentially provide more control. The default value of 1 keeps output magnitudes similar to input magnitudes.
    """
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

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
            return torch.nn.functional.linear(x, w)
        
        # Otherwise do a 2D convolution
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))


class MPEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x):
        w = self.weight.to(torch.float32)

        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
                
        w = normalize(w)
        w = w.to(x.dtype)

        return torch.nn.functional.embedding(x, self.weight)


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        mode='enc',
        resample_mode='keep',
        attention=False,
        channels_per_head=64,
        dropout=0,
        res_balance=0.3,
        attn_balance=0.3,
        clip_act=256,
    ):
        """
        Block module for the EDM2Unet2D architecture.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_channels (int): Number of embedding channels.
            mode (str, optional): Block mode, either 'enc' for encoder or 'dec' for decoder. Default is 'enc'.
            resample_mode (str, optional): Resampling mode, either 'keep', 'up', or 'down'. Default is 'keep'.
            attention (bool, optional): Whether to use attention mechanism. Default is False.
            channels_per_head (int, optional): Number of channels per attention head. Default is 64.
            dropout (float, optional): Dropout rate. Default is 0.
            res_balance (float, optional): Balance factor for residual connection. Default is 0.3.
            attn_balance (float, optional): Balance factor for attention. Default is 0.3.
            clip_act (int, optional): Activation clipping value. Default is 256.

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
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if mode == 'enc' else in_channels, out_channels, kernel=[3, 3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1]) if self.num_heads != 0 else None

    def attn(self, x):
        y = self.attn_qkv(x)
        y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
        q, k, v = normalize(y, dim=2).unbind(3)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.attn_proj(y.reshape(*x.shape))

    def forward(self, x, emb):
        x = resample(x, mode=self.resample_mode)
        if self.mode == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
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


class EDMUnet2D(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels=None,
        label_dim=0,
        model_channels=128,
        model_channel_mults=None,
        layers_per_block=2,
        emb_channels=None,
        noise_emb_dims=None,
        custom_cond_emb_dims=0,
        attn_resolutions=None,
        midblock_attention=True,
        concat_balance=0.3,
        logvar_channels=128,
        **block_kwargs
    ):
        """
        Parameters:
            image_size (int): The size of the input image.
            in_channels (int): The number of channels in the input image. 
                Usually the same as out_channels, unless some channels are used for conditioning.
            out_channels (int): The number of channels in the output image. Default is in_channels.
            label_dim (int, optional): The number of channels in the label image. Defaults to 0.
            model_channels (int, optional): The dimension of the model. Default is 128.
            model_channel_mults (list, optional): The channel multipliers for each block. Default is [1, 2, 3, 4].
            layers_per_block (int, optional): The number of layers per block. Default is 2.
            emb_channels (int, optional): The number of channels in the embedding. Default is model_channels * max(model_channel_mults).
            noise_emb_dims (int, optional): The number of channels in the noise embedding. Default is model_channels.
            custom_cond_emb_dims (int, optional): The number of channels in the custom conditional embedding. Default is 0 (disabled).
            attn_resolutions (list, optional): The resolutions at which attention is applied. Default is None.
            midblock_attention (bool, optional): Whether to apply attention in the midblock. Default is True.
            concat_balance (float, optional): Balance factor for concatenation. Default is 0.3.
            logvar_channels (int, optional): The number of channels for uncertainty estimation. Default is 128.
        """
        super().__init__()
        self.config = dict(locals())
        
        self.concat_balance = concat_balance

        model_channel_mults = model_channel_mults or [1, 2, 3, 4]
        emb_channels = emb_channels or model_channels * max(model_channel_mults)
        noise_emb_dims = noise_emb_dims or model_channels
        attn_resolutions = attn_resolutions or []

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(model_channel_mults)

        block_channels = [model_channels * m for m in model_channel_mults]

        self.noise_fourier = MPFourier(model_channels)
        self.noise_linear = MPConv(noise_emb_dims, emb_channels, kernel=[])
        self.custom_cond_linear = MPConv(custom_cond_emb_dims, emb_channels, kernel=[]) if custom_cond_emb_dims != 0 else None
        self.label_embeds = MPEmbedding(label_dim, emb_channels) if label_dim != 0 else None

        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Encoder.
        self.enc = torch.nn.ModuleDict()
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
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, (channels, nb) in reversed(list(enumerate(zip(block_channels, layers_per_block)))):
            res = image_size // 2**level
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

    def forward(self, x, noise_labels, label_index=None, conditional_embeddings=None,
                return_logvar=False):
        embeds = []
        embeds.append(self.noise_linear(self.noise_fourier(noise_labels)))
        if self.custom_cond_linear is not None:
            embeds.append(self.custom_cond_linear(conditional_embeddings))
        if self.label_embeds is not None:
            embeds.append(self.label_embeds(label_index))
        emb = mp_sum(embeds)
        emb = mp_silu(emb)

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