"""
Multi-task decoder.

Takes the fused feature map and upsamples it back to full resolution (512x512),
then produces three outputs:
  - semantic segmentation 2018  (num_classes channels)
  - semantic segmentation 2019  (num_classes channels)
  - change                      (1 channel)

The number of upsampling steps adapts to the encoder's output resolution:
  - ResNet backbone: 16x16 feature map -> 5 steps
  - DINOv3 backbone: 32x32 feature map -> 4 steps
"""

import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """One upsampling block: doubles the spatial resolution and refines."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class MultiTaskDecoder(nn.Module):
    """Shared decoding trunk + three task heads.

    num_upsamples : how many x2 upsampling steps to reach full resolution.
    """

    def __init__(self, in_channels, num_classes=10, num_upsamples=5):
        super().__init__()

        # Build the upsampling trunk. Channels are halved at each block,
        # down to a minimum of 32, starting from in_channels.
        blocks = []
        current_channels = in_channels
        for i in range(num_upsamples):
            out_channels = max(current_channels // 2, 32)
            if i == num_upsamples - 1:
                out_channels = 32   # last block feeds the heads (32 channels)
            blocks.append(DecoderBlock(current_channels, out_channels))
            current_channels = out_channels

        self.blocks = nn.ModuleList(blocks)

        # Three task heads (1x1 conv).
        self.head_sem_2018 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.head_sem_2019 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.head_change = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, fused):
        x = fused
        for block in self.blocks:
            x = block(x)

        out_sem_2018 = self.head_sem_2018(x)
        out_sem_2019 = self.head_sem_2019(x)
        out_change = self.head_change(x)
        return out_sem_2018, out_sem_2019, out_change