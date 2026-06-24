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
from src.models.fusion import ConcatDiffFusion
from src.models.decoder import MultiTaskDecoder


class ChangeDetectionModel(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Build the three stages, wiring their channel counts together.
        self.encoder = ResNetEncoder()
        self.fusion = ConcatDiffFusion(in_channels=self.encoder.out_channels)
        self.decoder = MultiTaskDecoder(
            in_channels=self.fusion.out_channels,
            num_classes=num_classes,
        )

    def forward(self, image_2018, image_2019):
        # 1. Encode both images with the shared (frozen) backbone.
        features_2018, features_2019 = self.encoder(image_2018, image_2019)

        # 2. Fuse the two feature maps (concat + absolute difference).
        fused = self.fusion(features_2018, features_2019)

        # 3. Decode into the three task outputs.
        out_sem_2018, out_sem_2019, out_change = self.decoder(fused)

        return out_sem_2018, out_sem_2019, out_change