"""
Mask-Conditioned U-Net for Real-Time Lung Tumour Tracking
TrackRAD2025 Grand Challenge — EEN1095 Implementation Project

Author: Chetan Kumar (A00054853)
Supervisor: Prof. Robert Sadleir
Dublin City University, August 2026

Architecture:
    Standard U-Net encoder-decoder with skip connections.
    The key modification is the input: instead of 1 channel (frame only),
    the network accepts 1 + N channels (frame + N previous masks),
    converting the task from blind segmentation to guided mask propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Mask-conditioned U-Net.

    Args:
        in_channels:  1 + N where N is the number of previous masks stacked.
                      N=1 → 2 channels (frame + 1 prev mask)
                      N=3 → 4 channels (frame + 3 prev masks)
                      N=5 → 6 channels (frame + 5 prev masks)
        out_channels: 1 (binary tumour mask)
        features:     encoder feature map sizes at each resolution level

    Total parameters: 7,765,697 (with default features, in_channels=2)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: list = [32, 64, 128, 256],
    ):
        super().__init__()
        self.downs     = nn.ModuleList()
        self.ups       = nn.ModuleList()
        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        # Output head
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _pad(self, x: torch.Tensor):
        """Pad input to nearest multiple of 16 (2^4 pooling stages)."""
        h, w   = x.shape[2], x.shape[3]
        pad_h  = (16 - h % 16) % 16
        pad_w  = (16 - w % 16) % 16
        return F.pad(x, (0, pad_w, 0, pad_h)), h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, orig_h, orig_w = self._pad(x)

        # Encoder pass — store skip connections
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder pass — upsample and concatenate skip connections
        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skips[-(i // 2 + 1)]
            # Crop if sizes differ (can happen with odd spatial dimensions)
            if x.shape != skip.shape:
                x = x[:, :, :skip.shape[2], :skip.shape[3]]
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        # Crop back to original spatial dimensions and apply output conv
        return self.final(x)[:, :, :orig_h, :orig_w]


if __name__ == "__main__":
    # Sanity check
    for n_prev in [1, 3, 5]:
        model  = UNet(in_channels=1 + n_prev)
        dummy  = torch.randn(1, 1 + n_prev, 256, 256)
        output = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"N={n_prev} | Input: {dummy.shape} | Output: {output.shape} "
              f"| Params: {params:,}")
