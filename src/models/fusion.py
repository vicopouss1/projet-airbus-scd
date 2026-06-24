"""
Fusion module for the 2018 and 2019 features.

This is where we encode the comparison between the two dates.
We combine three pieces of information along the channel axis:
  - the 2018 features        (the "before" state)
  - the 2019 features        (the "after" state)
  - their absolute difference (where and by how much things changed)

This fusion has no weights to learn: it is just simple operations
(concatenation, subtraction, absolute value).
Matches the [E_T1 ; E_T2 ; |E_T1 - E_T2|] design.
"""

import torch
import torch.nn as nn


class ConcatDiffFusion(nn.Module):

    def __init__(self, in_channels):
        # in_channels: number of channels of ONE feature map
        #              (2048 for ResNet-50).
        super().__init__()

        # After fusion we stacked 3 maps of in_channels each.
        # The decoder uses this to know how many channels it receives.
        self.out_channels = in_channels * 3

    def forward(self, features_2018, features_2019):
        # features_2018, features_2019: tensors (B, C, H, W)
        # return: fused map (B, 3*C, H, W)

        # Absolute difference between the two dates, channel by channel.
        difference = torch.abs(features_2018 - features_2019)

        # Stack the three along the channel axis (dim=1).
        # dim=0 is the batch, dim=1 the channels, dim=2 and 3 the space.
        fused = torch.cat([features_2018, features_2019, difference], dim=1)

        return fused