"""
ResNet-50 siamese encoder.

"Siamese": we apply the SAME ResNet (same weights) to both the 2018 and
2019 images. The weight sharing is what makes the comparison valid:
two identical areas produce identical features.

The backbone is frozen: its ImageNet weights do not change during training.
Only the heads (added later) will learn.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 1. Load ResNet-50 with its ImageNet pre-trained weights.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 2. Remove the head: ResNet ends with a pooling layer and a
        #    classification layer (1000 ImageNet classes) we don't want.
        #    We keep every layer EXCEPT the last two (avgpool + fc).
        layers = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*layers)

        # 3. Freeze the backbone: none of its weights will be updated.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Number of output channels of ResNet-50 (used later by the
        # fusion module and the decoder).
        self.out_channels = 2048

    def encode_one_image(self, image):
        """Run one image through the backbone and return its feature map.

        image  : tensor (B, 3, H, W)
        return : feature map (B, 2048, H/32, W/32)
        """
        return self.backbone(image)

    def forward(self, image_2018, image_2019):
        """Encode both images with the same weights.

        Returns the two feature maps (2018 and 2019).
        """
        features_2018 = self.encode_one_image(image_2018)
        features_2019 = self.encode_one_image(image_2019)
        return features_2018, features_2019