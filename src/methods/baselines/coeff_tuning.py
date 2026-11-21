"""Coeff-Tuning — Graph Filter Subspace View for Tuning Attention-Based Models.

Injects a learnable (num_heads × num_heads) coefficient matrix into each
Attention module.  After softmax, the attention map is mixed across heads:

    attn_out = einsum('bhnl, hk -> bknl', attn, coeff * scale + I)

where `coeff` is initialized to zero so the initial output is identical to
the original model (residual / identity start).

The key insight from the paper: multi-head attention maps span a subspace of
graph convolutional filters.  Tuning only the mixing coefficients (breaking
the convex-combination constraint) is sufficient to adapt the model while
using very few parameters — just num_heads² per attention layer.

Reference:
    Miao et al., "Coeff-Tuning: A Graph Filter Subspace View for Tuning
    Attention-Based Large Models", CVPR 2025.
"""

from __future__ import annotations

import types
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# Patched forward (bound to individual Attention modules)
# ---------------------------------------------------------------------------

def _coeff_attn_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Patched Attention.forward that applies the head-mixing coefficient."""
    B, N, C = x.shape

    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Head-mixing: coeff * scale + I  (residual identity keeps init behaviour)
    mix = self._coeff_bases_drop(self._coeff_param) * self._coeff_scale \
          + torch.eye(self.num_heads, dtype=attn.dtype, device=attn.device)
    # mix: (num_heads, num_heads)  →  attn: (B, H, N, N)
    attn = torch.einsum("bhnl,kh->bknl", attn, mix)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_attention(module: nn.Module) -> bool:
    return (
        hasattr(module, "qkv")
        and hasattr(module, "proj")
        and hasattr(module, "attn_drop")
        and hasattr(module, "num_heads")
    )


# ---------------------------------------------------------------------------
# MethodModule
# ---------------------------------------------------------------------------

@register_method("coeff_tuning")
class CoeffTuningMethod(MethodModule):
    """Coeff-Tuning PEFT method.

    Adds one (num_heads × num_heads) learnable matrix per Attention block.
    All other parameters remain frozen.

    Config keys:
        scale (float): multiplier applied to coeff before adding identity.
            Default 1.0.
        dropout (float): dropout applied to coeff matrix during training.
            Default 0.0.
        start_layer (int): index of the first transformer block to inject
            (0-based).  Blocks before this index are skipped.  Default 0.
    """

    def __init__(self, config: dict) -> None:
        self.scale: float = float(config.get("scale", 1.0))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.start_layer: int = int(config.get("start_layer", 0))

        self._coeff_params: List[nn.Parameter] = []
        self._dropout_modules: List[nn.Module] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._coeff_params = []
        self._dropout_modules = []

        attn_idx = 0
        for module in backbone.modules():
            if not _is_attention(module):
                continue

            if attn_idx < self.start_layer:
                attn_idx += 1
                continue

            num_heads = module.num_heads

            # Learnable coefficient matrix, zero-init → identity start
            coeff = nn.Parameter(torch.zeros(num_heads, num_heads))
            drop = nn.Dropout(p=self.dropout) if self.dropout > 0.0 else nn.Identity()

            module._coeff_param = coeff
            module._coeff_scale = self.scale
            module._coeff_bases_drop = drop
            module.forward = _coeff_attn_forward.__get__(module, module.__class__)

            self._coeff_params.append(coeff)
            self._dropout_modules.append(drop)
            attn_idx += 1

        return backbone

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._coeff_params)

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self._coeff_params)
        return {"total": trainable, "trainable": trainable}
