import torch
import torch.nn as nn

def leaky_relu(x, leak=0.2):
    return nn.functional.leaky_relu(x, negative_slope=leak)

class ResBlock(nn.Module):
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
                nn.GroupNorm(16, out_channels),
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

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.block1 = ResBlock(in_channels)
        self.block2 = ResBlock(in_channels)
        self.pixel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x[..., ::2, ::2] if self.downsample else x
        x = self.pixel_conv(x)
        return x

class ResNetDiscriminator(nn.Module):
    def __init__(self, in_channels=2, model_channels=64, n_layers=3):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(nn.Conv2d(in_channels, model_channels, kernel_size=1, stride=1))

        # Residual stages
        blocks = []
        cur_channels = model_channels
        for _ in range(max(n_layers - 1, 0)):
            out_channels = min(cur_channels * 2, 512)
            blocks.append(DBlock(cur_channels, out_channels, downsample=True))
            cur_channels = out_channels
        blocks.append(DBlock(cur_channels, cur_channels, downsample=False))
        self.blocks = nn.Sequential(*blocks)

        # Final 1-channel conv for patch logits
        self.final = nn.Conv2d(cur_channels, 1, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        return x

if __name__ == '__main__':
    # Test discriminator
    disc = PatchDiscriminator(in_channels=2)
    x = torch.randn(1, 2, 64, 64)
    out = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
