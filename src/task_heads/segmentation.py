"""Segmentation head for semantic segmentation tasks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Lightweight segmentation head that upsamples ViT patch tokens.

    Architecture: Linear projection → reshape to spatial → upsample →
    1×1 conv classifier.

    Parameters
    ----------
    embed_dim:
        Backbone feature dimension (e.g. 768).
    num_classes:
        Number of segmentation classes.
    image_size:
        Target spatial resolution for the output mask.
    patch_size:
        Patch size used by the backbone (needed to compute spatial dims).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 21,
        image_size: int = 512,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_side = image_size // patch_size

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x:
            ``(B, N, D)`` patch token sequence from backbone.
            If a CLS token is present (N = num_patches + 1), it is stripped.
        """
        B, N, D = x.shape
        num_patches = self.num_patches_side ** 2

        # Strip CLS token if present
        if N == num_patches + 1:
            x = x[:, 1:]

        x = self.proj(x)  # (B, num_patches, D)

        # Reshape to spatial grid
        h = w = self.num_patches_side
        x = x.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)

        # Upsample to original image resolution
        x = F.interpolate(
            x, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )

        return self.classifier(x)  # (B, num_classes, H, W)
