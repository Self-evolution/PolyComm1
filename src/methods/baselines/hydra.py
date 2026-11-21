"""Hydra — Multi-head adapter with shared trunk for ViT backbones.

Hydra inserts a shared down-projection (trunk) followed by multiple
task-specific up-projections (heads) in parallel. For single-task
fine-tuning, this reduces to a standard bottleneck adapter with an
optional multi-branch structure.

The adapter is inserted sequentially after the MLP block:
    x = x + Adapter(x)   where Adapter = up(act(down(x)))

Hydra extends this by allowing multiple up-projection heads whose outputs
are summed (or averaged), enabling multi-task or multi-scale adaptation.

Reference: "Hydra: Multi-head Low-rank Adaptation for Parameter Efficient
Fine-Tuning" — PETL benchmark baseline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class HydraAdapter(nn.Module):
    """Bottleneck adapter with a shared trunk and multiple up-heads."""

    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # Multiple up-projection heads
        self.up_heads = nn.ModuleList(
            [nn.Linear(bottleneck_dim, in_features) for _ in range(num_heads)]
        )
        self.num_heads = num_heads

        # Near-zero init so initial output ≈ 0 (identity residual)
        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.zeros_(self.down.bias)
        for head in self.up_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.down(self.dropout(x)))
        # Sum outputs from all heads
        out = sum(head(h) for head in self.up_heads)
        return out / self.num_heads  # average to keep scale stable


class HydraBlock(nn.Module):
    """Wraps a ViT block with a sequential Hydra adapter after the MLP."""

    def __init__(self, block: nn.Module, adapter: HydraAdapter) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter
        self._has_split = (
            hasattr(block, "attn") and hasattr(block, "mlp")
            and hasattr(block, "norm1") and hasattr(block, "norm2")
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._has_split:
            b = self.block
            drop = getattr(b, "drop_path", lambda t: t)
            x = x + drop(b.attn(b.norm1(x)))
            mlp_out = drop(b.mlp(b.norm2(x)))
            x = x + mlp_out + self.adapter(x)
            return x
        out = self.block(x, **kwargs)
        if isinstance(out, tuple):
            h, *rest = out
            return (h + self.adapter(h), *rest)
        return out + self.adapter(out)


@register_method("hydra")
class HydraMethod(MethodModule):
    """Hydra PEFT method — multi-head bottleneck adapter.

    Config keys:
        bottleneck_dim (int): adapter hidden dimension. Default 64.
        num_heads (int): number of parallel up-projection heads. Default 1.
        dropout (float): dropout inside the adapter. Default 0.0.
        init_scale (float): std for near-zero weight init. Default 1e-3.
    """

    def __init__(self, config: dict) -> None:
        self.bottleneck_dim: int = int(config.get("bottleneck_dim", 64))
        self.num_heads: int = int(config.get("num_heads", 1))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.init_scale: float = float(config.get("init_scale", 1e-3))
        self._adapters: List[HydraAdapter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._adapters = []
        self._inject_recursive(backbone)
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self._is_vit_block(child):
                hidden_dim = self._infer_hidden_dim(child)
                if hidden_dim is None:
                    self._inject_recursive(child)
                    continue
                adapter = HydraAdapter(
                    in_features=hidden_dim,
                    bottleneck_dim=self.bottleneck_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    init_scale=self.init_scale,
                )
                self._adapters.append(adapter)
                setattr(module, name, HydraBlock(child, adapter))
            else:
                self._inject_recursive(child)

    @staticmethod
    def _is_vit_block(module: nn.Module) -> bool:
        children = {n for n, _ in module.named_children()}
        return "attn" in children and ("mlp" in children or "ffn" in children)

    @staticmethod
    def _infer_hidden_dim(module: nn.Module) -> Optional[int]:
        for attr in ("norm1", "norm2", "norm"):
            ln = getattr(module, attr, None)
            if isinstance(ln, nn.LayerNorm):
                return ln.normalized_shape[0]
        for attr in ("mlp", "ffn"):
            sub = getattr(module, attr, None)
            if sub is not None:
                for m in sub.modules():
                    if isinstance(m, nn.Linear):
                        return m.in_features
        return None

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for a in self._adapters:
            params.extend(a.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
