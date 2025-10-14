import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def leaky_relu(x, leak=0.2):
    return nn.functional.leaky_relu(x, negative_slope=leak)

# MSR Initialization
def msr_initializer(layer, activation_gain=1):
    fan_in = layer.weight.data.size(1) * layer.weight.data[0][0].numel()
    layer.weight.data.normal_(0, activation_gain / math.sqrt(fan_in))
    
    if layer.bias is not None:
        layer.bias.data.zero_()
    
    return layer

# BiasedActivation (Reference implementation since we may not have CUDA ops)
class BiasedActivation(nn.Module):
    gain = math.sqrt(2 / (1 + 0.2 ** 2))
    
    def __init__(self, input_units):
        super(BiasedActivation, self).__init__()
        
        self.bias = nn.Parameter(torch.empty(input_units))
        self.bias.data.zero_()
        
    def forward(self, x):
        y = x + self.bias.to(x.dtype).view(1, -1, 1, 1) if len(x.shape) > 2 else x + self.bias.to(x.dtype).view(1, -1)
        return F.leaky_relu(y, negative_slope=0.2)

class Convolution(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, groups=1, activation_gain=1):
        super(Convolution, self).__init__()
        
        self.layer = msr_initializer(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, 
                     padding=(kernel_size - 1) // 2, groups=groups, bias=False), 
            activation_gain=activation_gain
        )
        
    def forward(self, x):
        return F.conv2d(x, self.layer.weight.to(x.dtype), padding=self.layer.padding, groups=self.layer.groups)

class InterpolativeDownsampler(nn.Module):
    """Filter-based downsampler for anti-aliasing"""
    def __init__(self, resampling_filter=[1, 2, 1]):
        super(InterpolativeDownsampler, self).__init__()
        
        # Create 2D filter from 1D filter
        f = torch.tensor(resampling_filter, dtype=torch.float32)
        f = f[:, None] * f[None, :]
        f = f / f.sum()
        self.register_buffer('filter', f[None, None, :, :])
        self.pad = (f.shape[0] - 1) // 2
        
    def forward(self, x):
        # Apply filter per channel
        c = x.shape[1]
        filter_expanded = self.filter.repeat(c, 1, 1, 1)
        x = F.conv2d(x, filter_expanded.to(x.dtype), padding=self.pad, groups=c)
        # Downsample by 2
        x = x[..., ::2, ::2]
        return x

class ResBlock(nn.Module):
    """Legacy ResBlock - kept for backwards compatibility"""
    def __init__(self, channels, leak=0.2):
        super().__init__()
        self.leak = leak
        self.conv1 = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.conv2 = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, groups=(channels * 3)//32, padding=1)
        self.conv3 = nn.Conv2d(channels * 3, channels, kernel_size=1)
        
        self.gn1 = nn.GroupNorm(16, channels * 3)
        self.gn2 = nn.GroupNorm(16, channels * 3)
        self.gn3 = nn.GroupNorm(16, channels)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.gn1(out)
        out = leaky_relu(out, self.leak)
        out = self.conv2(out)
        #out = self.gn2(out)
        out = leaky_relu(out, self.leak)
        out = self.conv3(out)
        out = x + out
        #out = self.gn3(out)
        return out

class ResidualBlock(nn.Module):
    """Variance-scaled residual block matching reference architecture"""
    def __init__(self, input_channels, cardinality, expansion_factor, kernel_size, variance_scaling_parameter):
        super(ResidualBlock, self).__init__()
        
        number_of_linear_layers = 3
        expanded_channels = input_channels * expansion_factor
        activation_gain = BiasedActivation.gain * variance_scaling_parameter ** (-1 / (2 * number_of_linear_layers - 2))
        
        self.linear_layer1 = Convolution(input_channels, expanded_channels, kernel_size=1, activation_gain=activation_gain)
        self.linear_layer2 = Convolution(expanded_channels, expanded_channels, kernel_size=kernel_size, groups=cardinality, activation_gain=activation_gain)
        self.linear_layer3 = Convolution(expanded_channels, input_channels, kernel_size=1, activation_gain=0)
        
        self.non_linearity1 = BiasedActivation(expanded_channels)
        self.non_linearity2 = BiasedActivation(expanded_channels)
        
    def forward(self, x):
        y = self.linear_layer1(x)
        y = self.linear_layer2(self.non_linearity1(y))
        y = self.linear_layer3(self.non_linearity2(y))
        
        return x + y

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, model_channels=64, n_layers=3):
        super().__init__()
        
        # Initial conv layer
        layers = [
            nn.Conv2d(in_channels, model_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        ]
        
        # Intermediate layers with increasing channels
        cur_channels = model_channels
        for i in range(n_layers - 1):
            out_channels = min(cur_channels * 2, 512)
            layers += [
                nn.Conv2d(cur_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            ]
            cur_channels = out_channels
            
        # Final layer
        layers += [
            nn.Conv2d(cur_channels, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class DownsampleLayer(nn.Module):
    """Downsample layer with optional channel change"""
    def __init__(self, input_channels, output_channels, resampling_filter):
        super(DownsampleLayer, self).__init__()
        
        self.resampler = InterpolativeDownsampler(resampling_filter)
        
        if input_channels != output_channels:
            self.linear_layer = Convolution(input_channels, output_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.resampler(x)
        x = self.linear_layer(x) if hasattr(self, 'linear_layer') else x
        
        return x

class DiscriminativeBasis(nn.Module):
    """Final basis layer for discriminator"""
    def __init__(self, input_channels, output_dimension):
        super(DiscriminativeBasis, self).__init__()
        
        self.basis = msr_initializer(nn.Conv2d(input_channels, input_channels, kernel_size=4, stride=1, padding=0, groups=input_channels, bias=False))
        self.linear_layer = msr_initializer(nn.Linear(input_channels, output_dimension, bias=False))
        
    def forward(self, x):
        return self.linear_layer(self.basis(x).view(x.shape[0], -1))

class DiscriminatorStage(nn.Module):
    """Full discriminator stage with residual blocks and downsampling"""
    def __init__(self, input_channels, output_channels, cardinality, number_of_blocks, expansion_factor, kernel_size, variance_scaling_parameter, resampling_filter=None, data_type=torch.float32):
        super(DiscriminatorStage, self).__init__()
        
        transition_layer = DiscriminativeBasis(input_channels, output_channels) if resampling_filter is None else DownsampleLayer(input_channels, output_channels, resampling_filter)
        self.layers = nn.ModuleList([ResidualBlock(input_channels, cardinality, expansion_factor, kernel_size, variance_scaling_parameter) for _ in range(number_of_blocks)] + [transition_layer])
        self.data_type = data_type
        
    def forward(self, x):
        x = x.to(self.data_type)
        
        for layer in self.layers:
            x = layer(x)
        
        return x

class ResNetDiscriminator(nn.Module):
    def __init__(self, 
                 in_channels=2,
                 width_per_stage=(64, 128, 256, 512),
                 cardinality_per_stage=None,
                 blocks_per_stage=None,
                 expansion_factor=3,
                 kernel_size=3,
                 resampling_filter=[1, 2, 1],
                 patch_then_average=False,
                 channel_means=None,
                 channel_stds=None):
        """
        Variance-scaled discriminator matching reference architecture.
        Always outputs a scalar per sample.
        
        Args:
            in_channels: Number of input channels
            width_per_stage: Tuple of channel counts per stage
            cardinality_per_stage: Tuple of cardinality (groups) per stage. If None, auto-computed as channels//32
            blocks_per_stage: Tuple of residual blocks per stage. If None, uses (2,) * len(width_per_stage)
            expansion_factor: Channel expansion factor in residual blocks
            kernel_size: Kernel size for grouped convolutions
            resampling_filter: Filter for anti-aliased downsampling
            patch_then_average: If True, compute patch logits (1x1 conv) then average spatially.
                              If False, use DiscriminativeBasis (4x4 conv + linear) for scalar output.
        """
        super().__init__()
        
        # Default values
        if cardinality_per_stage is None:
            cardinality_per_stage = tuple(max(1, w // 32) for w in width_per_stage)
        if blocks_per_stage is None:
            blocks_per_stage = (2,) * len(width_per_stage)
        if channel_means is None:
            channel_means = torch.zeros(in_channels)
        if channel_stds is None:
            channel_stds = torch.ones(in_channels)
        
        self.channel_means = torch.as_tensor(channel_means)
        self.channel_stds = torch.as_tensor(channel_stds)
        self.patch_then_average = patch_then_average
        
        # Variance scaling parameter
        variance_scaling_parameter = sum(blocks_per_stage)
        
        # Extraction layer (stem)
        self.extraction_layer = Convolution(in_channels, width_per_stage[0], kernel_size=1)
        
        # Main stages
        main_layers = []
        for x in range(len(width_per_stage) - 1):
            main_layers.append(
                DiscriminatorStage(
                    width_per_stage[x], 
                    width_per_stage[x + 1], 
                    cardinality_per_stage[x], 
                    blocks_per_stage[x], 
                    expansion_factor, 
                    kernel_size, 
                    variance_scaling_parameter, 
                    resampling_filter
                )
            )
        
        # Final stage
        if patch_then_average:
            # For patch-then-average, use residual blocks + 1x1 conv
            final_blocks = [
                ResidualBlock(
                    width_per_stage[-1], 
                    cardinality_per_stage[-1], 
                    expansion_factor, 
                    kernel_size, 
                    variance_scaling_parameter
                ) for _ in range(blocks_per_stage[-1])
            ]
            main_layers.extend(final_blocks)
        else:
            # For scalar output with DiscriminativeBasis
            main_layers.append(
                DiscriminatorStage(
                    width_per_stage[-1], 
                    1, 
                    cardinality_per_stage[-1], 
                    blocks_per_stage[-1], 
                    expansion_factor, 
                    kernel_size, 
                    variance_scaling_parameter
                )
            )
        
        self.main_layers = nn.ModuleList(main_layers)
        
        # Final projection for patch-then-average
        if patch_then_average:
            self.final = nn.Conv2d(width_per_stage[-1], 1, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = (x - self.channel_means[None, :, None, None].to(dtype=x.dtype, device=x.device)) / self.channel_stds[None, :, None, None].to(dtype=x.dtype, device=x.device)
        x = self.extraction_layer(x)
        
        for layer in self.main_layers:
            x = layer(x)
        
        if self.patch_then_average:
            # Compute patch logits then average spatially for scalar output
            x = self.final(x)
            x = x.mean(dim=[2, 3]).view(x.shape[0])
        else:
            # Return scalar from DiscriminativeBasis
            x = x.view(x.shape[0])
        
        return x

if __name__ == '__main__':
    # Test discriminators
    print("Testing PatchDiscriminator (legacy)...")
    disc = PatchDiscriminator(in_channels=2)
    x = torch.randn(1, 2, 64, 64)
    out = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    print("\nTesting ResNetDiscriminator (DiscriminativeBasis)...")
    disc2 = ResNetDiscriminator(in_channels=2, width_per_stage=(64, 128, 256, 512))
    x2 = torch.randn(2, 2, 64, 64)
    out2 = disc2(x2)
    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {out2.shape}")
    
    print("\nTesting ResNetDiscriminator (patch-then-average)...")
    disc3 = ResNetDiscriminator(in_channels=2, width_per_stage=(64, 128, 256, 512), patch_then_average=True)
    x3 = torch.randn(2, 2, 64, 64)
    out3 = disc3(x3)
    print(f"Input shape: {x3.shape}")
    print(f"Output shape: {out3.shape}")
