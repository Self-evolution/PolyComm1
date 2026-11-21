"""Video classification head for action recognition tasks."""

from __future__ import annotations

import torch.nn as nn


class VideoClassificationHead(nn.Module):
    """Classification head for video action recognition.

    Applies temporal average pooling over frame-level CLS tokens,
    followed by layer norm, dropout, and a linear classifier.

    Parameters
    ----------
    embed_dim:
        Backbone feature dimension (e.g. 768).
    num_classes:
        Number of action classes.
    dropout:
        Dropout probability before the linear layer.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 400,
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
            Accepts multiple input shapes:
            - ``(B, D)`` — already pooled
            - ``(B, N, D)`` — single-frame patch sequence; uses CLS token
            - ``(B, T, N, D)`` — multi-frame; averages CLS tokens over T
        """
        if x.dim() == 4:
            # (B, T, N, D) → take CLS from each frame → average
            x = x[:, :, 0, :]  # (B, T, D)
            x = x.mean(dim=1)  # (B, D)
        elif x.dim() == 3:
            x = x[:, 0]  # CLS token
        # else: already (B, D)

        x = self.norm(x)
        x = self.drop(x)
        return self.fc(x)
