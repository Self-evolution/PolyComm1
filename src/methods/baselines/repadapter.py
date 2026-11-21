"""RepAdapter — Structural Re-parameterization Adapter for ViT backbones.

Uses parallel branches (identity + depthwise conv) that can be merged at
inference time. Inserted after attention and/or MLP blocks.

Reference: Luo et al., "RepAdapter: Towards Unified Parameter-Efficient
Transfer Learning for Vision Transformers", 2023.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


class RepAdapterLayer(nn.Module):
    """Re-parameterizable adapter: down → (identity + conv) → up, with residual.

    At training time uses two parallel branches; at inference they can be
    merged into a single linear transform.
    """

    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int = 8,
        groups: int = 2,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        self.groups = groups

        self.down = nn.Linear(in_features, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, in_features)

        # Parallel conv branch (1D grouped conv over the bottleneck channels)
        self.conv = nn.Conv1d(
            bottleneck_dim, bottleneck_dim,
            kernel_size=1, groups=groups, bias=True,
        )
        # Identity branch scale
        self.act = nn.ReLU()

        # Near-zero init
        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=init_scale)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        h = self.down(x)                          # (B, N, dim)
        h = self.act(h)
        # Conv branch: treat N as sequence length
        h_conv = self.conv(h.transpose(1, 2)).transpose(1, 2)  # (B, N, dim)
        h = h + h_conv                            # identity + conv
        out = self.up(h)                          # (B, N, C)
        return x + out                            # residual


class RepAdapterBlock(nn.Module):
    """Wraps a ViT block with RepAdapter after attention and/or MLP."""

    def __init__(
        self,
        block: nn.Module,
        adapter_attn: Optional[RepAdapterLayer],
        adapter_mlp: Optional[RepAdapterLayer],
    ) -> None:
        super().__init__()
        self.block = block
        self.adapter_attn = adapter_attn
        self.adapter_mlp = adapter_mlp
        self._has_split = (
            hasattr(block, "attn") and hasattr(block, "mlp")
            and hasattr(block, "norm1") and hasattr(block, "norm2")
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._has_split:
            b = self.block
            drop = getattr(b, "drop_path", lambda t: t)
            attn_out = drop(b.attn(b.norm1(x)))
            if self.adapter_attn is not None:
                attn_out = self.adapter_attn(attn_out)
            x = x + attn_out
            mlp_out = drop(b.mlp(b.norm2(x)))
            if self.adapter_mlp is not None:
                mlp_out = self.adapter_mlp(mlp_out)
            x = x + mlp_out
            return x
        # Fallback
        out = self.block(x, **kwargs)
        if isinstance(out, tuple):
            h, *rest = out
            if self.adapter_mlp is not None:
                h = self.adapter_mlp(h)
            return (h, *rest)
        if self.adapter_mlp is not None:
            out = self.adapter_mlp(out)
        return out


@register_method("repadapter")
class RepAdapterMethod(MethodModule):
    """RepAdapter PEFT method.

    Config keys:
        bottleneck_dim (int): adapter bottleneck size. Default 8.
        groups (int): groups for the parallel conv branch. Default 2.
        init_scale (float): std for near-zero init. Default 1e-3.
        insert_after_attn (bool): insert adapter after attention. Default True.
        insert_after_mlp (bool): insert adapter after MLP. Default True.
    """

    def __init__(self, config: dict) -> None:
        self.bottleneck_dim: int = int(config.get("bottleneck_dim", 8))
        self.groups: int = int(config.get("groups", 2))
        self.init_scale: float = float(config.get("init_scale", 1e-3))
        self.insert_after_attn: bool = bool(config.get("insert_after_attn", True))
        self.insert_after_mlp: bool = bool(config.get("insert_after_mlp", True))
        self._adapters: List[RepAdapterLayer] = []

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
                adapter_attn = self._make_adapter(hidden_dim) if self.insert_after_attn else None
                adapter_mlp = self._make_adapter(hidden_dim) if self.insert_after_mlp else None
                wrapped = RepAdapterBlock(child, adapter_attn, adapter_mlp)
                setattr(module, name, wrapped)
            else:
                self._inject_recursive(child)

    def _make_adapter(self, hidden_dim: int) -> RepAdapterLayer:
        a = RepAdapterLayer(
            in_features=hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            groups=self.groups,
            init_scale=self.init_scale,
        )
        self._adapters.append(a)
        return a

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
