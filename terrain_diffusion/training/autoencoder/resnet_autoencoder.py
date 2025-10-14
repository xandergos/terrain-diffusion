import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from terrain_diffusion.training.gan.discriminator_basic import (
    ResidualBlock, Convolution, InterpolativeDownsampler, DownsampleLayer
)


class InterpolativeUpsampler(nn.Module):
    """Filter-based upsampler for anti-aliasing"""
    def __init__(self, resampling_filter=[1, 2, 1]):
        super(InterpolativeUpsampler, self).__init__()
        
        # Create 2D filter from 1D filter
        f = torch.tensor(resampling_filter, dtype=torch.float32)
        f = f[:, None] * f[None, :]
        f = f / f.sum()
        
        self.register_buffer('filter', f[None, None, :, :])
        self.pad = (f.shape[0] - 1) // 2
        
    def forward(self, x):
        # Upsample by 2 using nearest neighbor
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # Apply filter per channel
        c = x.shape[1]
        filter_expanded = self.filter.repeat(c, 1, 1, 1)
        x = F.conv2d(x, filter_expanded.to(x.dtype), padding=self.pad, groups=c)
        return x


class UpsampleLayer(nn.Module):
    """Upsample layer with optional channel change"""
    def __init__(self, input_channels, output_channels, resampling_filter):
        super(UpsampleLayer, self).__init__()
        
        self.resampler = InterpolativeUpsampler(resampling_filter)
        
        if input_channels != output_channels:
            self.linear_layer = Convolution(input_channels, output_channels, kernel_size=1)
        
    def forward(self, x):
        # Linear transform BEFORE upsampling (matches R3GAN Generator UpsampleLayer)
        x = self.linear_layer(x) if hasattr(self, 'linear_layer') else x
        x = self.resampler(x)
        
        return x


class EncoderStage(nn.Module):
    """Encoder stage with residual blocks and downsampling"""
    def __init__(self, input_channels, output_channels, cardinality, number_of_blocks, expansion_factor, kernel_size, variance_scaling_parameter, resampling_filter=[1, 2, 1], data_type=torch.float32):
        super(EncoderStage, self).__init__()
        
        transition_layer = DownsampleLayer(input_channels, output_channels, resampling_filter)
        self.layers = nn.ModuleList([ResidualBlock(input_channels, cardinality, expansion_factor, kernel_size, variance_scaling_parameter) for _ in range(number_of_blocks)] + [transition_layer])
        self.data_type = data_type
        
    def forward(self, x):
        x = x.to(self.data_type)
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class DecoderStage(nn.Module):
    """Decoder stage with residual blocks and upsampling"""
    def __init__(self, input_channels, output_channels, cardinality, number_of_blocks, expansion_factor, kernel_size, variance_scaling_parameter, resampling_filter=[1, 2, 1], data_type=torch.float32):
        super(DecoderStage, self).__init__()
        
        transition_layer = UpsampleLayer(input_channels, output_channels, resampling_filter)
        self.layers = nn.ModuleList([transition_layer] + [ResidualBlock(output_channels, cardinality, expansion_factor, kernel_size, variance_scaling_parameter) for _ in range(number_of_blocks)])
        self.data_type = data_type
        
    def forward(self, x):
        x = x.to(self.data_type)
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class ResNetAutoencoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels=None,
        width_per_stage=None,
        cardinality_per_stage=None,
        blocks_per_stage=None,
        expansion_factor=3,
        kernel_size=3,
        resampling_filter=[1, 2, 1],
        latent_channels=None,
        conditional_inputs=[],
        n_logvar=1
    ):
        """
        Variance-scaled ResNet autoencoder with encoder/decoder structure.
        
        Args:
            image_size (int): The size of the input image (not used for computation but kept for interface compatibility).
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Default is in_channels.
            width_per_stage (tuple, optional): Channel counts per stage. Default is (64, 128, 256, 512).
            cardinality_per_stage (tuple, optional): Cardinality (groups) per stage. If None, auto-computed as channels//32.
            blocks_per_stage (tuple, optional): Residual blocks per stage. If None, uses (2,) * len(width_per_stage).
            expansion_factor (int): Channel expansion factor in residual blocks. Default is 3.
            kernel_size (int): Kernel size for grouped convolutions. Default is 3.
            resampling_filter (list): Filter for anti-aliased up/downsampling. Default is [1, 2, 1].
            latent_channels (int, optional): Number of latent channels. Required.
            conditional_inputs (list, optional): List of conditional inputs (not yet implemented). Default is [].
            n_logvar (int): Number of logvar parameters. Default is 1.
        """
        super().__init__()
        
        width_per_stage = width_per_stage or (64, 128, 256, 512)
        out_channels = out_channels or in_channels
        assert latent_channels is not None, "latent_channels must be specified"
        
        # Default values
        if cardinality_per_stage is None:
            cardinality_per_stage = tuple(max(1, w // 32) for w in width_per_stage)
        if blocks_per_stage is None:
            blocks_per_stage = (2,) * len(width_per_stage)
        
        # Variance scaling parameter
        variance_scaling_parameter = sum(blocks_per_stage)
        
        # Encoder
        self.encoder_stem = Convolution(in_channels, width_per_stage[0], kernel_size=1)
        
        encoder_stages = []
        for i in range(len(width_per_stage) - 1):
            encoder_stages.append(
                EncoderStage(
                    width_per_stage[i],
                    width_per_stage[i + 1],
                    cardinality_per_stage[i],
                    blocks_per_stage[i],
                    expansion_factor,
                    kernel_size,
                    variance_scaling_parameter,
                    resampling_filter
                )
            )
        
        # Final encoder stage (no downsampling)
        final_encoder_blocks = [
            ResidualBlock(
                width_per_stage[-1],
                cardinality_per_stage[-1],
                expansion_factor,
                kernel_size,
                variance_scaling_parameter
            ) for _ in range(blocks_per_stage[-1])
        ]
        encoder_stages.extend(final_encoder_blocks)
        
        self.encoder_stages = nn.ModuleList(encoder_stages)
        
        # Encoder output to latent (mean and logvar) - use gain=0 for output layer
        self.encoder_final = Convolution(width_per_stage[-1], latent_channels * 2, kernel_size=1, activation_gain=1.0)
        
        # Decoder
        self.decoder_stem = Convolution(latent_channels, width_per_stage[-1], kernel_size=1, activation_gain=1.0)
        
        decoder_stages = []
        
        # Initial decoder stage (no upsampling)
        initial_decoder_blocks = [
            ResidualBlock(
                width_per_stage[-1],
                cardinality_per_stage[-1],
                expansion_factor,
                kernel_size,
                variance_scaling_parameter
            ) for _ in range(blocks_per_stage[-1])
        ]
        decoder_stages.extend(initial_decoder_blocks)
        
        # Decoder upsampling stages
        for i in reversed(range(len(width_per_stage) - 1)):
            decoder_stages.append(
                DecoderStage(
                    width_per_stage[i + 1],
                    width_per_stage[i],
                    cardinality_per_stage[i],
                    blocks_per_stage[i],
                    expansion_factor,
                    kernel_size,
                    variance_scaling_parameter,
                    resampling_filter
                )
            )
        
        self.decoder_stages = nn.ModuleList(decoder_stages)
        
        # Decoder output - use gain=1 to match R3GAN Generator's AggregationLayer
        self.decoder_final = Convolution(width_per_stage[0], out_channels, kernel_size=1, activation_gain=1)
        
        # Logvar for reconstruction uncertainty
        self.logvar = nn.Parameter(torch.zeros([n_logvar]))
        
    def preencode(self, x, conditional_inputs=None):
        """Encode input to mean and logvar."""
        h = self.encoder_stem(x)
        
        for stage in self.encoder_stages:
            h = stage(h)
        
        encodings = self.encoder_final(h)
        
        means = encodings[:, :encodings.shape[1] // 2]
        logvars = encodings[:, encodings.shape[1] // 2:]
        
        return means, logvars
    
    def postencode(self, means, logvars, use_mode=False):
        """Sample from latent distribution."""
        if use_mode:
            return means
        std = torch.exp(logvars * 0.5)
        eps = torch.randn_like(std)
        return means + eps * std
    
    def decode(self, z, include_logvar=False):
        """Decode latent to output."""
        x = self.decoder_stem(z)
        
        for stage in self.decoder_stages:
            x = stage(x)
        
        decoder_out = self.decoder_final(x)
        
        if include_logvar:
            logvar = self.logvar.reshape(-1, 1, 1, 1)
            return decoder_out, logvar
        return decoder_out
    
    def forward(self, x, conditional_inputs=None, use_mode=False):
        """Full forward pass through autoencoder."""
        means, logvars = self.preencode(x, conditional_inputs)
        z = self.postencode(means, logvars, use_mode)
        return self.decode(z)
    
    def count_parameters(self):
        """Count the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test autoencoder with config-compatible parameters
    model = ResNetAutoencoder(
        image_size=512,
        in_channels=2,
        out_channels=2,
        width_per_stage=(64, 128, 256, 512),
        blocks_per_stage=(2, 2, 2, 2),
        latent_channels=4
    )
    
    x = torch.randn(2, 2, 512, 512)
    
    # Test encode
    means, logvars = model.preencode(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent means shape: {means.shape}")
    print(f"Latent logvars shape: {logvars.shape}")
    
    # Test decode
    z = model.postencode(means, logvars)
    out = model.decode(z)
    print(f"Output shape: {out.shape}")
    print(f"Output STD: {out.std()}")
    
    # Test full forward
    out_full = model(x)
    print(f"Full forward output shape: {out_full.shape}")
    
    print(f"Total parameters: {model.count_parameters():,}")
