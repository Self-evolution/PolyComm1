"""AdaptFormer — Parallel Adapter alongside MLP for ViT backbones.

Inserts a parallel adapter branch alongside the MLP block. Unlike sequential
adapters (Pfeiffer/Houlsby), the adapter output is added directly to the MLP
output before the residual connection:
    x = x + MLP(norm(x)) + scale * Adapter(norm(x))

The adapter uses: down-proj → ReLU → up-proj.

Reference: Chen et al., "AdaptFormer: Adapting Vision Transformers for
Scalable Visual Recognition", NeurIPS 2022.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class AdaptFormerLayer(nn.Module):
    """Parallel adapter: down → ReLU → up (no residual, added externally)."""

    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, in_features)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.dropout(x))))


class AdaptFormerBlock(nn.Module):
    """Wraps a ViT block with a parallel AdaptFormer adapter on the MLP branch."""

    def __init__(
        self,
        block: nn.Module,
        adapter: AdaptFormerLayer,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter
        self.scale = scale
        self._has_split = (
            hasattr(block, "attn") and hasattr(block, "mlp")
            and hasattr(block, "norm1") and hasattr(block, "norm2")
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._has_split:
            b = self.block
            drop = getattr(b, "drop_path", lambda t: t)
            # Attention sub-block (unchanged)
            x = x + drop(b.attn(b.norm1(x)))
            # Parallel MLP + adapter
            normed = b.norm2(x)
            x = x + drop(b.mlp(normed)) + self.adapter(normed) * self.scale
            return x
        # Fallback: post-block parallel adapter
        out = self.block(x, **kwargs)
        if isinstance(out, tuple):
            h, *rest = out
            return (h + self.adapter(x) * self.scale, *rest)
        return out + self.adapter(x) * self.scale


@register_method("adaptformer")
class AdaptFormerMethod(MethodModule):
    """AdaptFormer PEFT method — parallel adapter alongside MLP.

    Config keys:
        bottleneck_dim (int): adapter hidden dimension. Default 64.
        scale (float): output scale for the parallel branch. Default 1.0.
        dropout (float): dropout inside the adapter. Default 0.0.
        init_scale (float): std for near-zero weight init. Default 1e-3.
    """

    def __init__(self, config: dict) -> None:
        self.bottleneck_dim: int = int(config.get("bottleneck_dim", 64))
        self.scale: float = float(config.get("scale", 1.0))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.init_scale: float = float(config.get("init_scale", 1e-3))
        self._adapters: List[AdaptFormerLayer] = []

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
                adapter = AdaptFormerLayer(
                    in_features=hidden_dim,
                    bottleneck_dim=self.bottleneck_dim,
                    dropout=self.dropout,
                    init_scale=self.init_scale,
                )
                self._adapters.append(adapter)
                setattr(module, name, AdaptFormerBlock(child, adapter, scale=self.scale))
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
