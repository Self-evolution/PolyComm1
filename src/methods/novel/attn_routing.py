"""AttnRouting — Attention Support/Routing PEFT.

Modifies the attention score computation by adding a learnable routing bias:

    A' = softmax(QK^T + M_theta)

Two modes for M_theta:
  low_rank:  M_theta = U @ V^T  (per-head low-rank bias matrix)
  anchor:    M_theta computed via anchor-mediated routing using learnable
             anchor embeddings P and routing matrix R.

The routing bias is added to the raw attention logits before softmax,
allowing the model to learn structured attention patterns without modifying
the Q/K/V projections.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# Per-block attention wrapper
# ---------------------------------------------------------------------------

class AttnRoutingWrapper(nn.Module):
    """Wraps a single transformer attention module to apply routing bias.

    Intercepts the attention forward and adds M_theta to QK^T before softmax:

        A' = softmax(QK^T + M_theta)

    Two modes:
      low_rank: M_theta = U @ V^T, shape (num_heads, seq_len, seq_len)
      anchor:   M_theta = P @ R @ P^T (projected through routing matrix)
    """

    def __init__(
        self,
        attn: nn.Module,
        mode: str,
        rank: int,
        seq_len: int,
        num_heads: int,
        hidden_dim: int,
        num_anchors: int,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mode = mode
        self.num_heads = num_heads
        self.seq_len = seq_len

        if mode == "low_rank":
            # M_theta = U @ V^T, shape per head: (seq_len, seq_len)
            # U: (num_heads, seq_len, rank), V: (num_heads, seq_len, rank)
            self.U = nn.Parameter(torch.zeros(num_heads, seq_len, rank))
            self.V = nn.Parameter(torch.zeros(num_heads, seq_len, rank))
            # Small random init for symmetry breaking
            nn.init.normal_(self.U, std=0.01)
            nn.init.normal_(self.V, std=0.01)
            self.P: Optional[nn.Parameter] = None
            self.R: Optional[nn.Parameter] = None
        elif mode == "anchor":
            # P: (num_anchors, hidden_dim) — learnable anchor embeddings
            # R: (num_anchors, num_anchors) — routing matrix
            self.P = nn.Parameter(torch.randn(num_anchors, hidden_dim) * 0.01)
            self.R = nn.Parameter(torch.zeros(num_anchors, num_anchors))
            self.U = None  # type: ignore[assignment]
            self.V = None  # type: ignore[assignment]
        else:
            raise ValueError(f"AttnRouting mode must be 'low_rank' or 'anchor', got '{mode}'")

    def _compute_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing bias M_theta.

        Returns tensor of shape (num_heads, seq_len, seq_len) or broadcastable.
        """
        if self.mode == "low_rank":
            # M = U @ V^T: (num_heads, seq_len, rank) x (num_heads, rank, seq_len)
            # -> (num_heads, seq_len, seq_len)
            return self.U @ self.V.transpose(-2, -1)
        else:
            # anchor mode: M = X @ P^T @ R @ P @ X^T
            # x: (B, N, C), P: (num_anchors, C), R: (num_anchors, num_anchors)
            # scores: (B, N, num_anchors) = x @ P^T
            scores = x @ self.P.t()                          # (B, N, num_anchors)
            # routed: (B, N, num_anchors) = scores @ R
            routed = scores @ self.R                          # (B, N, num_anchors)
            # bias: (B, N, N) = routed @ scores^T
            bias = routed @ scores.transpose(-2, -1)          # (B, N, N)
            # Expand to (B, 1, N, N) for broadcasting over heads
            return bias.unsqueeze(1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Intercept attention forward; add routing bias to QK^T before softmax."""
        B, N, C = x.shape
        head_dim = C // self.num_heads
        attn_mod = self.attn

        if not (hasattr(attn_mod, "qkv") and isinstance(attn_mod.qkv, nn.Linear)):
            # Fallback: run original attention unchanged
            return attn_mod(x, **kwargs)

        # Compute Q, K, V
        qkv = attn_mod.qkv(x)                                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                        # each (B, H, N, head_dim)

        scale = attn_mod.scale if hasattr(attn_mod, "scale") else head_dim ** -0.5
        attn_logits = (q @ k.transpose(-2, -1)) * scale                # (B, H, N, N)

        # Add routing bias M_theta
        bias = self._compute_bias(x)                                    # broadcastable to (B, H, N, N)
        attn_logits = attn_logits + bias

        attn_weights = F.softmax(attn_logits, dim=-1)
        if hasattr(attn_mod, "attn_drop"):
            attn_weights = attn_mod.attn_drop(attn_weights)

        # Weighted sum of values
        out = (attn_weights @ v)                                        # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)                     # (B, N, C)
        out = attn_mod.proj(out)
        if hasattr(attn_mod, "proj_drop"):
            out = attn_mod.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# MethodModule implementation
# ---------------------------------------------------------------------------

@register_method("attn_routing")
class AttnRoutingMethod(MethodModule):
    """Attention Support/Routing PEFT.

    Adds a learnable routing bias M_theta to the attention logits:
        A' = softmax(QK^T + M_theta)

    Config keys:
        mode (str): "low_rank" or "anchor". Default "low_rank".
        rank (int): rank for low_rank mode. Default 4.
        seq_len (int): sequence length (number of tokens). Default 197 (ViT-B/16 with CLS).
        num_anchors (int): number of anchors for anchor mode. Default 16.
    """

    def __init__(self, config: dict) -> None:
        self.mode: str = config.get("mode", "low_rank")
        self.rank: int = int(config.get("rank", 4))
        self.seq_len: int = int(config.get("seq_len", 197))
        self.num_anchors: int = int(config.get("num_anchors", 16))

        if self.mode not in ("low_rank", "anchor"):
            raise ValueError(
                f"AttnRouting mode must be 'low_rank' or 'anchor', got '{self.mode}'"
            )

        self._wrappers: List[AttnRoutingWrapper] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Replace each transformer block's attention with an AttnRoutingWrapper."""
        self._wrappers = []
        self._inject_recursive(backbone)
        if not self._wrappers:
            raise RuntimeError(
                "AttnRoutingMethod.inject: no attention modules found. "
                "Ensure the backbone has timm-style Attention blocks with a .qkv attribute."
            )
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self._is_attention(child):
                hidden_dim, num_heads = self._infer_dims(child)
                wrapper = AttnRoutingWrapper(
                    attn=child,
                    mode=self.mode,
                    rank=self.rank,
                    seq_len=self.seq_len,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    num_anchors=self.num_anchors,
                )
                setattr(module, name, wrapper)
                self._wrappers.append(wrapper)
            else:
                self._inject_recursive(child)

    @staticmethod
    def _is_attention(module: nn.Module) -> bool:
        """Heuristic: timm ViT Attention has both .qkv and .proj as nn.Linear."""
        return (
            hasattr(module, "qkv")
            and isinstance(module.qkv, nn.Linear)
            and hasattr(module, "proj")
            and isinstance(module.proj, nn.Linear)
        )

    @staticmethod
    def _infer_dims(attn: nn.Module):
        """Return (hidden_dim, num_heads) from a timm Attention module."""
        hidden_dim = attn.proj.in_features
        num_heads = getattr(attn, "num_heads", 1)
        return hidden_dim, num_heads

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for w in self._wrappers:
            if w.mode == "low_rank":
                params.append(w.U)
                params.append(w.V)
            else:
                params.append(w.P)
                params.append(w.R)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
