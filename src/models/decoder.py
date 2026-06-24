"""
Multi-task decoder.

Takes the fused feature map (small and deep, 16x16 x 6144) and upsamples it
back to full resolution (512x512), then produces THREE outputs:
  - semantic segmentation 2018  (10 channels: 9 classes + unlabeled)
  - semantic segmentation 2019  (10 channels)
  - change                      (1 channel)

Architecture: a shared decoding trunk (common to all 3 tasks) that upsamples
the resolution, followed by 3 lightweight task heads (one per output).
The shared trunk learns a representation useful for all three tasks at once.
"""

import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """One upsampling block: doubles the spatial resolution and refines.

    Upsamples the feature map (x2 in height and width), then applies a
    convolution to refine the result. Several of these blocks are stacked
    to progressively go from 16x16 up to 512x512.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Upsampling: doubles the spatial size (16->32, then 32->64, etc.)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)

        # Convolution to refine and adjust the channel count.
        # 3x3 kernel with padding 1 keeps the spatial size unchanged.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Activation: introduces non-linearity.
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)   # double the resolution
        x = self.conv(x)       # refine + adjust channels
        x = self.relu(x)       # non-linearity
        return x
    
class MultiTaskDecoder(nn.Module):
    """Shared decoding trunk + three task heads.

    Takes the fused map (B, in_channels, 16, 16), upsamples it to full
    resolution (512x512) through five DecoderBlocks, then produces three
    outputs through three lightweight convolution heads.
    """

    def __init__(self, in_channels, num_classes=10):
        # in_channels : channels of the fused map (6144 for ResNet-50 + concat-diff)
        # num_classes : number of semantic classes (9 classes + unlabeled = 10)
        super().__init__()

        # --- Shared trunk: five upsampling blocks ---
        # Resolution:  16 -> 32 -> 64 -> 128 -> 256 -> 512
        # Channels: in_channels -> 512 -> 256 -> 128 -> 64 -> 32
        self.block1 = DecoderBlock(in_channels, 512)
        self.block2 = DecoderBlock(512, 256)
        self.block3 = DecoderBlock(256, 128)
        self.block4 = DecoderBlock(128, 64)
        self.block5 = DecoderBlock(64, 32)

        # --- Three task heads ---
        # Each head is a single 1x1 convolution: it just maps the 32 shared
        # channels to the right number of output channels for its task.
        self.head_sem_2018 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.head_sem_2019 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.head_change = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, fused):
        # fused: (B, in_channels, 16, 16)

        # Go up through the shared trunk.
        x = self.block1(fused)   # -> (B, 512, 32, 32)
        x = self.block2(x)       # -> (B, 256, 64, 64)
        x = self.block3(x)       # -> (B, 128, 128, 128)
        x = self.block4(x)       # -> (B, 64, 256, 256)
        x = self.block5(x)       # -> (B, 32, 512, 512)

        # Branch into the three heads.
        out_sem_2018 = self.head_sem_2018(x)   # -> (B, 10, 512, 512)
        out_sem_2019 = self.head_sem_2019(x)   # -> (B, 10, 512, 512)
        out_change = self.head_change(x)       # -> (B, 1, 512, 512)

        return out_sem_2018, out_sem_2019, out_change