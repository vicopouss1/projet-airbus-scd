"""
Siamese encoder based on DINOv3 ViT-Large (frozen), pretrained on satellite imagery.

Same interface as ResNetEncoder: takes two images, returns two feature maps,
exposes out_channels. So the rest of the pipeline (fusion, decoder) is unchanged.

DINOv3 is a Vision Transformer: it outputs a sequence of tokens, not a spatial
grid. We keep only the patch tokens (dropping the CLS and register tokens) and
reshape them into a 2D feature map.

For a 512x512 image with patch size 16: 32x32 = 1024 patch tokens, each of
dimension 1024 (ViT-Large) -> feature map (B, 1024, 32, 32).

Normalization: the dataloader returns raw images (0-255). DINOv3 satellite
expects its own normalization (different from ImageNet), so we apply it here,
inside the encoder. This keeps the dataloader backbone-agnostic.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class DinoEncoder(nn.Module):

    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-sat493m"):
        super().__init__()

        # Load the pretrained DINOv3 backbone.
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze it: no weights updated during training.
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.out_channels = 1024          # ViT-Large embedding dimension
        self.patch_size = 16
        self.num_special_tokens = 5       # 1 CLS + 4 register tokens

        # DINOv3 satellite (SAT-493M) normalization stats.
        # Stored as buffers so they move to the right device with the model.
        mean = torch.tensor([0.430, 0.411, 0.296]).view(1, 3, 1, 1)
        std = torch.tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)

    def normalize(self, image):
        """Raw image (0-255) -> normalized for DINOv3 satellite.

        First scale to 0-1, then apply mean/std.
        """
        image = image / 255.0
        image = (image - self.norm_mean) / self.norm_std
        return image

    def encode_one_image(self, image):
        """Run one image through DINOv3 and return a spatial feature map.

        image  : tensor (B, 3, H, W), raw values 0-255
        return : feature map (B, 1024, H/16, W/16)
        """
        B, C, H, W = image.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        # Normalize before feeding DINOv3.
        image = self.normalize(image)

        # DINOv3 forward. last_hidden_state: (B, num_tokens, embed_dim).
        outputs = self.backbone(pixel_values=image)
        tokens = outputs.last_hidden_state

        # Drop special tokens (CLS + registers), keep only patch tokens.
        patch_tokens = tokens[:, self.num_special_tokens:, :]

        # Reshape sequence of patches into a 2D spatial grid.
        feat = patch_tokens.reshape(B, grid_h, grid_w, self.out_channels)
        feat = feat.permute(0, 3, 1, 2)   # channels first

        return feat

    def forward(self, image_2018, image_2019):
        """Encode both images with the same frozen DINOv3."""
        features_2018 = self.encode_one_image(image_2018)
        features_2019 = self.encode_one_image(image_2019)
        return features_2018, features_2019