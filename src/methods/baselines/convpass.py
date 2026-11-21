"""ConvPass — Convolutional Bypass Adapter for ViT-style backbones.

Inserts a parallel convolutional bypass (down-proj → 3x3 conv → up-proj)
alongside both the attention and MLP blocks in each transformer layer.
The conv branch processes patch tokens reshaped into 2D spatial feature maps.

Reference: Jie & Deng, "Convolutional Bypasses Are Better Vision Transformer
Adapters", ECCV 2022.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ConvPassModule(nn.Module):
    """Down-proj → act → 3x3 conv → act → dropout → up-proj bypass."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 8,
        dropout: float = 0.1,
        xavier_init: bool = False,
    ) -> None:
        super().__init__()
        self.dim = bottleneck_dim

        self.adapter_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.adapter_conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, 1, 1)
        self.adapter_up = nn.Linear(bottleneck_dim, hidden_dim)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(dropout)

        # Init: down xavier, conv near-identity, up zero
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            eye = torch.eye(bottleneck_dim, dtype=torch.float)
            self.adapter_conv.weight.data[:, :, 1, 1] += eye
        nn.init.zeros_(self.adapter_conv.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Determine spatial size: N may include CLS token
        has_cls = False
        patch_n = N
        if N > 1:
            # Heuristic: if sqrt(N-1) is integer, assume CLS token present
            sqrt_val = math.isqrt(N - 1)
            if sqrt_val * sqrt_val == N - 1:
                has_cls = True
                patch_n = N - 1

        H = W = math.isqrt(patch_n)

        x_down = self.adapter_down(x)  # (B, N, dim)
        x_down = self.act(x_down)

        if has_cls:
            x_cls = x_down[:, :1]  # (B, 1, dim)
            x_patch = x_down[:, 1:]  # (B, H*W, dim)
        else:
            x_cls = None
            x_patch = x_down

        # Reshape to 2D and apply conv
        x_patch_2d = x_patch.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)
        x_patch_2d = self.adapter_conv(x_patch_2d)
        x_patch = x_patch_2d.permute(0, 2, 3, 1).reshape(B, H * W, self.dim)

        if has_cls and x_cls is not None:
            # Apply conv to CLS token as 1x1 spatial
            x_cls_2d = x_cls.reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
            x_cls_2d = self.adapter_conv(x_cls_2d)
            x_cls = x_cls_2d.permute(0, 2, 3, 1).reshape(B, 1, self.dim)
            x_down = torch.cat([x_cls, x_patch], dim=1)
        else:
            x_down = x_patch

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        return self.adapter_up(x_down)  # (B, N, hidden_dim)


class ConvPassBlock(nn.Module):
    """Wraps a timm ViT block with parallel ConvPass adapters on attn and mlp."""

    def __init__(
        self,
        block: nn.Module,
        adapter_attn: ConvPassModule,
        adapter_mlp: ConvPassModule,
        scale: float = 1.0,
        attn_only: bool = False,
    ) -> None:
        super().__init__()
        self.block = block
        self.adapter_attn = adapter_attn
        self.adapter_mlp = adapter_mlp
        self.scale = scale
        self.attn_only = attn_only
        self._has_split = (
            hasattr(block, "attn") and hasattr(block, "mlp")
            and hasattr(block, "norm1") and hasattr(block, "norm2")
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._has_split:
            b = self.block
            drop = getattr(b, "drop_path", lambda t: t)
            x = x + drop(b.attn(b.norm1(x))) + drop(self.adapter_attn(b.norm1(x))) * self.scale
            if self.attn_only:
                x = x + drop(b.mlp(b.norm2(x)))
            else:
                x = x + drop(b.mlp(b.norm2(x))) + drop(self.adapter_mlp(b.norm2(x))) * self.scale
            return x
        # Fallback
        out = self.block(x, **kwargs)
        if isinstance(out, tuple):
            h, *rest = out
            return (h + self.adapter_attn(x) * self.scale, *rest)
        return out + self.adapter_attn(x) * self.scale


@register_method("convpass")
class ConvPassMethod(MethodModule):
    """ConvPass PEFT method.

    Config keys:
        bottleneck_dim (int): conv bottleneck channels. Default 8.
        scale (float): output scale for the bypass branch. Default 1.0.
        dropout (float): dropout in the bypass. Default 0.1.
        xavier_init (bool): use Xavier init for conv weights. Default False.
        attn_only (bool): only inject on attention branch. Default False.
    """

    def __init__(self, config: dict) -> None:
        self.bottleneck_dim: int = int(config.get("bottleneck_dim", 8))
        self.scale: float = float(config.get("scale", 1.0))
        self.dropout: float = float(config.get("dropout", 0.1))
        self.xavier_init: bool = bool(config.get("xavier_init", False))
        self.attn_only: bool = bool(config.get("attn_only", False))
        self._adapters: List[ConvPassModule] = []

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
                adapter_attn = self._make_adapter(hidden_dim)
                adapter_mlp = self._make_adapter(hidden_dim)
                wrapped = ConvPassBlock(
                    child, adapter_attn, adapter_mlp,
                    scale=self.scale, attn_only=self.attn_only,
                )
                setattr(module, name, wrapped)
            else:
                self._inject_recursive(child)

    def _make_adapter(self, hidden_dim: int) -> ConvPassModule:
        a = ConvPassModule(
            hidden_dim=hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
            xavier_init=self.xavier_init,
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
