"""UNet block implementation using magnitude-preserving layers."""
from functools import partial
import torch
import torch.nn as nn
from .mp_layers import (
    MPConv, MPConvResample, mp_silu, mp_leaky_relu, 
    mp_sum, normalize, resample
)


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
        no_padding=False,
        activation='silu'
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
        
        if activation == 'silu':
            self.activation = mp_silu
        elif activation == 'leaky_relu':
            self.activation = partial(mp_leaky_relu, alpha=0.2)
        else:
            raise ValueError(f"Activation {activation} not supported")
        
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
        q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
        w = torch.einsum('nhcq,nhck->nhqk', q, k / torch.sqrt(torch.tensor(q.shape[2], dtype=q.dtype, device=q.device))).softmax(dim=3)
        y = torch.einsum('nhqk,nhck->nhcq', w, v)
        return self.attn_proj(y.reshape(*x.shape))
            
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
            y = self.conv_res0[0](self.activation(x))
            y = self.conv_res0[1](self.activation(y))
        else:
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

        # Connect the branches.
        if self.mode == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        
        if x.shape[2:] != y.shape[2:]:
            diff_h = x.shape[2] - y.shape[2]
            diff_w = x.shape[3] - y.shape[3]
            x = x[:, :, diff_h//2:-(diff_h-diff_h//2), diff_w//2:-(diff_w-diff_w//2)]
        x = mp_sum([x, y], w=self.res_balance)

        if self.num_heads != 0:
            x = mp_sum([x, self.attn(x)], w=self.attn_balance)
        
        # Clip activations.
        if self.clip_act is not None:
            x = torch.clip(x, -self.clip_act, self.clip_act)

        return x

