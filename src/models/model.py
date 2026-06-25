"""
Full change detection model.

Wraps the three building blocks into a single model:
  siamese encoder -> concat-diff fusion -> multi-task decoder

Once built, you just call:
    out_sem_2018, out_sem_2019, out_change = model(image_2018, image_2019)
and the three stages run internally.
"""

import torch
import torch.nn as nn

from src.models.encoders.resnet_encoder import ResNetEncoder
from src.models.encoders.dino_encoder import DinoEncoder
from src.models.fusion import ConcatDiffFusion
from src.models.decoder import MultiTaskDecoder


class ChangeDetectionModel(nn.Module):

    def __init__(self, num_classes=10, backbone="resnet"):
        super().__init__()

        # Choose the encoder backbone.
        if backbone == "resnet":
            self.encoder = ResNetEncoder()
            num_upsamples = 5          # 16x16 -> 512 (five x2 steps)
        elif backbone == "dino":
            self.encoder = DinoEncoder()
            num_upsamples = 4          # 32x32 -> 512 (four x2 steps)
        else:
            raise ValueError("backbone must be 'resnet' or 'dino', got " + backbone)

        self.fusion = ConcatDiffFusion(in_channels=self.encoder.out_channels)
        self.decoder = MultiTaskDecoder(
            in_channels=self.fusion.out_channels,
            num_classes=num_classes,
            num_upsamples=num_upsamples,
        )

    def forward(self, image_2018, image_2019):
        features_2018, features_2019 = self.encoder(image_2018, image_2019)
        fused = self.fusion(features_2018, features_2019)
        out_sem_2018, out_sem_2019, out_change = self.decoder(fused)
        return out_sem_2018, out_sem_2019, out_change