"""PolyCLS — polynomial attention plus class-token residual calibration.

PolyAttn changes the attention propagation inside each block. PolyCLS keeps
that structure and adds a zero-initialized low-rank residual adapter on the
final CLS feature. The adapter gives classification tasks a compact way to
reshape the decision feature while preserving the frozen backbone output at
initialization.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.novel.polyattn import PolyAttnMethod
from src.methods.registry import register_method


class ClsResidualAdapter(nn.Module):
    """Low-rank residual adapter applied to the CLS feature only."""

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        gate_init: float | None = None,
        token_index: int = 0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("ClsResidualAdapter rank must be positive")
        self.rank = rank
        self.scaling = alpha / rank
        self.token_index = token_index
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.up = nn.Linear(rank, embed_dim, bias=False)
        self.gate = (
            nn.Parameter(torch.tensor(float(gate_init)))
            if gate_init is not None
            else None
        )

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def _adapt_vector(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.act(self.down(self.dropout(self.norm(x)))))
        residual = residual * self.scaling
        if self.gate is not None:
            residual = residual * self.gate
        return x + residual

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            return self._adapt_vector(features)
        if features.dim() == 3:
            cls = self._adapt_vector(features[:, self.token_index])
            adapted = features.clone()
            adapted[:, self.token_index] = cls
            return adapted
        raise ValueError(
            "ClsResidualAdapter expects backbone features with shape "
            "(B, D) or (B, N, D)"
        )


class ClsResidualBackbone(nn.Module):
    """Backbone wrapper that adapts the final CLS representation."""

    def __init__(self, backbone: nn.Module, adapter: ClsResidualAdapter) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.adapter(self.backbone(*args, **kwargs))


@register_method("polycls")
class PolyCLSMethod(MethodModule):
    """PolyAttn with a zero-initialized CLS residual adapter.

    Config keys:
        All PolyAttn keys are forwarded unchanged.
        cls_rank: rank of the final CLS residual adapter. Default 8.
        cls_alpha: residual scaling alpha. Default 2 * cls_rank.
        cls_dropout: dropout before the CLS adapter bottleneck. Default 0.0.
        cls_gate_init: optional trainable scalar gate. Default disabled.
        cls_token_index: token index to adapt for sequence outputs. Default 0.
    """

    def __init__(self, config: dict) -> None:
        self.poly = PolyAttnMethod(config)
        self.cls_rank = int(config.get("cls_rank", 8))
        if self.cls_rank <= 0:
            raise ValueError("PolyCLS cls_rank must be positive")
        self.cls_alpha = float(config.get("cls_alpha", 2 * self.cls_rank))
        self.cls_dropout = float(config.get("cls_dropout", 0.0))
        gate_value = config.get("cls_gate_init")
        self.cls_gate_init = None if gate_value is None else float(gate_value)
        self.cls_token_index = int(config.get("cls_token_index", 0))
        self._adapter: ClsResidualAdapter | None = None

    def inject(self, backbone: nn.Module) -> nn.Module:
        backbone = self.poly.inject(backbone)
        embed_dim = self._infer_embed_dim(backbone)
        self._adapter = ClsResidualAdapter(
            embed_dim=embed_dim,
            rank=self.cls_rank,
            alpha=self.cls_alpha,
            dropout=self.cls_dropout,
            gate_init=self.cls_gate_init,
            token_index=self.cls_token_index,
        )
        return ClsResidualBackbone(backbone, self._adapter)

    @staticmethod
    def _infer_embed_dim(backbone: nn.Module) -> int:
        if hasattr(backbone, "num_features"):
            return int(backbone.num_features)
        if hasattr(backbone, "embed_dim"):
            return int(backbone.embed_dim)
        if hasattr(backbone, "norm") and isinstance(backbone.norm, nn.LayerNorm):
            return int(backbone.norm.normalized_shape[0])
        raise RuntimeError(
            "PolyCLS could not infer backbone embed_dim; set a backbone with "
            ".num_features, .embed_dim, or final LayerNorm."
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.poly.get_trainable_params())
        if self._adapter is not None:
            params.extend(self._adapter.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
