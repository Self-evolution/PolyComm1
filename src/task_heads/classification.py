"""Linear classification head for image classification tasks."""

from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    """Simple linear head: global pool → dropout → linear.

    Parameters
    ----------
    embed_dim:
        Dimension of the backbone feature vector (e.g. 768 for ViT-B).
    num_classes:
        Number of target classes.
    dropout:
        Dropout probability before the linear layer.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x:
            Backbone output.  Accepts either:
            - ``(B, D)`` — already pooled feature vector
            - ``(B, N, D)`` — sequence of patch tokens; uses the CLS token
              (index 0) or global average pooling when no CLS token.
        """
        if x.dim() == 3:
            x = x[:, 0]  # CLS token
        x = self.norm(x)
        x = self.drop(x)
        return self.fc(x)
