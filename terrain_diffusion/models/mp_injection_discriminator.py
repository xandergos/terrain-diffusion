from functools import partial
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from terrain_diffusion.models.mp_layers import MPConv, MPFourier, mp_concat
from terrain_diffusion.models.unet_block import UNetBlock


class MPInjectionDiscriminator(ModelMixin, ConfigMixin):
    """
    Discriminator that consumes two maps:
    - Conditioning map: real but noisy, larger spatial size than image map
    - Image map: real or fake

    The conditioning map is first passed through no-padding UNetBlocks (as in MPInjectionGenerator)
    until its spatial size matches the image map. Then both are concatenated channel-wise and
    processed by a standard encoder-style discriminator similar to MPDiscriminator to output
    a realness prediction.
    """
    @register_to_config
    def __init__(
        self,
        channels,
        cond_channels,
        cond_depth=2,
        image_channels=32,
        image_channel_mults=[1],
        layers_per_block=2,
        fourier_channels=64,
        no_padding=True,
        emb_channels=None
    ):
        super().__init__()
        self.channels = channels
        emb_channels = emb_channels or image_channels * max(image_channel_mults)

        # Conditioning processing
        self.cond_stem = MPConv(self.channels + 1, cond_channels, kernel=[1, 1])
        self.cond_blocks = nn.ModuleList([
            UNetBlock(cond_channels,
                      cond_channels,
                      emb_channels=emb_channels,
                      mode='dec',
                      no_padding=no_padding,
                      activation='leaky_relu')
            for _ in range(cond_depth)
        ])
        # Per-channel Fourier embeddings of t and linear projection to emb space
        self.fouriers = nn.ModuleList([MPFourier(fourier_channels) for _ in range(self.channels)])
        self.emb_linear = MPConv(self.channels * fourier_channels, emb_channels, kernel=[])

        # Discriminator body (encoder-like, similar to MPDiscriminator)
        # No bias channel in discriminator input
        disc_in_channels = self.channels + cond_channels
        self.in_conv = MPConv(disc_in_channels, image_channels, kernel=[1, 1])

        self.blocks = nn.ModuleList()
        cur_channels = image_channels
        for k, mult in enumerate(image_channel_mults):
            out_channels = image_channels * mult
            for i in range(layers_per_block - 1):
                self.blocks.append(UNetBlock(
                    cur_channels if i == 0 else out_channels,
                    out_channels,
                    emb_channels=image_channels,
                    mode='enc',
                    activation='leaky_relu'
                ))
            self.blocks.append(UNetBlock(
                cur_channels if layers_per_block == 1 else out_channels,
                out_channels,
                emb_channels=image_channels,
                mode='enc',
                activation='leaky_relu',
                resample_mode='down' if k != len(image_channel_mults) - 1 else 'keep'
            ))
            cur_channels = out_channels

        self.final_conv = MPConv(cur_channels, 1, kernel=[1, 1])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gain = nn.Parameter(torch.ones([]))

    def _center_crop_to(self, x, target_h, target_w):
        h, w = x.shape[-2:]
        if h == target_h and w == target_w:
            return x
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return x[:, :, top:top + target_h, left:left + target_w]

    def forward(self, cond, img, t):
        t = torch.log(torch.clip(torch.tan(t) * 0.5, 0.002, 80) / 4)
        # Process conditioning map with no-padding blocks to shrink spatial dims
        cond = torch.cat([cond, torch.ones_like(cond[:, :1])], dim=1)
        cond = self.cond_stem(cond)
        # Build embedding from t: shape [B, C], produce [B, model_channels]
        emb = torch.cat([fourier(t[:, i]) for i, fourier in enumerate(self.fouriers)], dim=1)
        emb = self.emb_linear(emb)
        for block in self.cond_blocks:
            cond = block(cond, emb=emb)

        # Center-crop conditioning features to match image size if off by a few pixels
        cond = self._center_crop_to(cond, img.shape[-2], img.shape[-1])

        x = mp_concat([img, cond], dim=1)

        # Discriminator body
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x, emb=emb)
        x = self.final_conv(x, gain=self.gain)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        return x

    def norm_weights(self):
        """Normalize all the weights in the model."""
        for module in self.modules():
            if module != self and hasattr(module, 'norm_weights'):
                module.norm_weights()

if __name__ == "__main__":
    # Simple sanity check
    model = MPInjectionDiscriminator(
        channels=6,
        cond_channels=64,
        cond_depth=4,
        image_channels=128,
        image_channel_mults=[1],
        layers_per_block=2,
        fourier_channels=64,
        no_padding=True,
        emb_channels=128
    )
    b, c = 2, 6
    cond = torch.randn(b, c, 20, 20)
    img = torch.randn(b, c, 4, 4)
    t = torch.rand(b, c) * (torch.pi / 2)
    out = model(cond, img, t)
    print(out.shape)