import torch
import torch.nn as nn

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

if __name__ == '__main__':
    # Test discriminator
    disc = PatchDiscriminator(in_channels=2)
    x = torch.randn(1, 2, 64, 64)
    out = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
