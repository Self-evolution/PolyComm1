"""RoPEGeometry — RoPE Phase Perturbation and Relative Position Bias PEFT.

Implements two complementary positional geometry adaptations:

1. **rope_perturbation**: Adds learnable phase perturbation to RoPE-style
   rotary embeddings applied to Q and K:

       q'_i = RoPE(q_i, theta + phase_delta)
       k'_i = RoPE(k_i, theta + phase_delta)

   where theta are the standard RoPE frequency parameters and phase_delta
   is a learnable per-layer, per-head, per-frequency offset.  When
   phase_delta = 0 the output is identical to standard RoPE.

   Because the ViT backbone uses learned positional embeddings (not RoPE),
   this mode *injects* RoPE-style rotation as an additive perturbation to
   Q and K in each attention layer.

2. **pos_bias**: Adds a learnable low-rank relative position bias to the
   attention logits before softmax:

       A' = softmax(QK^T / sqrt(d) + U @ V^T)

   where U ∈ R^{max_len × rank} and V ∈ R^{max_len × rank} are per-layer
   learnable matrices.  The bias matrix M = U @ V^T has shape
   (max_len, max_len) and is added to the (N, N) attention score matrix.

3. **both**: Apply both modes simultaneously.

Zero / identity initialisation ensures the method starts as a no-op:
  - phase_delta initialised to zeros  → RoPE rotation is identity at init
  - pos_bias U, V initialised to zeros → bias matrix is zero at init

References:
  - RoPE: Su et al., 2021 (RoFormer: Enhanced Transformer with Rotary Position Embedding)
  - Relative position bias: Shaw et al., 2018; Raffel et al., 2020 (T5)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# RoPE utility
# ---------------------------------------------------------------------------

def _build_rope_freqs(dim_per_head: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build standard RoPE frequency parameters.

    Returns theta of shape (dim_per_head // 2,) following the original RoPE
    formula: theta_i = 1 / 10000^(2i / d).
    """
    half = dim_per_head // 2
    i = torch.arange(half, device=device, dtype=dtype)
    return 1.0 / (10000.0 ** (2.0 * i / dim_per_head))


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to x.

    Args:
        x:     shape (B, H, N, head_dim)
        freqs: shape (dim_per_head // 2,) — frequency parameters (theta + delta)

    Returns:
        Rotated tensor of same shape as x.
    """
    B, H, N, head_dim = x.shape
    half = head_dim // 2

    # Position indices: (N,)
    pos = torch.arange(N, device=x.device, dtype=x.dtype)

    # Angles: (N, half) = outer product of positions and frequencies
    angles = torch.outer(pos, freqs)                        # (N, half)

    cos = angles.cos()                                      # (N, half)
    sin = angles.sin()                                      # (N, half)

    # Broadcast to (1, 1, N, half)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split x into two halves along head_dim
    x1 = x[..., :half]                                     # (B, H, N, half)
    x2 = x[..., half:]                                     # (B, H, N, half)

    # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


# ---------------------------------------------------------------------------
# Per-block attention wrapper
# ---------------------------------------------------------------------------

class RoPEGeometryWrapper(nn.Module):
    """Wraps a single transformer attention module with RoPE geometry adaptation.

    Depending on mode:
      rope_perturbation: injects RoPE rotation with learnable phase_delta into Q and K.
      pos_bias:          adds low-rank bias U @ V^T to attention logits.
      both:              applies both simultaneously.

    Args:
        attn:          original timm-style attention module.
        mode:          one of "rope_perturbation", "pos_bias", "both".
        num_heads:     number of attention heads.
        dim_per_head:  dimension per head.
        max_len:       maximum sequence length for position bias.
        rank:          rank for low-rank position bias decomposition.
        dropout:       dropout probability on phase_delta (0 = disabled).
    """

    def __init__(
        self,
        attn: nn.Module,
        mode: str,
        num_heads: int,
        dim_per_head: int,
        max_len: int,
        rank: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mode = mode
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.max_len = max_len
        self.rank = rank

        half = dim_per_head // 2

        if mode in ("rope_perturbation", "both"):
            # phase_delta: (num_heads, half) — zero init so initial output = standard RoPE
            self.phase_delta: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(num_heads, half)
            )
        else:
            self.phase_delta = None

        if mode in ("pos_bias", "both"):
            # U, V: (max_len, rank) — zero init so initial bias = 0
            self.pos_bias_U: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(max_len, rank)
            )
            self.pos_bias_V: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(max_len, rank)
            )
        else:
            self.pos_bias_U = None
            self.pos_bias_V = None

        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def _rope_freqs(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return per-head RoPE frequencies with phase_delta applied.

        Returns tensor of shape (num_heads, dim_per_head // 2).
        """
        base_freqs = _build_rope_freqs(self.dim_per_head, device, dtype)  # (half,)
        # Broadcast base_freqs to (num_heads, half) and add delta
        freqs = base_freqs.unsqueeze(0) + self.phase_delta                # (num_heads, half)
        return freqs

    def _pos_bias(self, N: int) -> torch.Tensor:
        """Compute position bias matrix M = U[:N] @ V[:N]^T.

        Returns tensor of shape (N, N) to be broadcast over (B, H, N, N).
        """
        U = self.pos_bias_U[:N]   # (N, rank)
        V = self.pos_bias_V[:N]   # (N, rank)
        return U @ V.t()          # (N, N)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Intercept attention forward; apply RoPE perturbation and/or position bias."""
        B, N, C = x.shape
        head_dim = C // self.num_heads
        attn_mod = self.attn

        if not (hasattr(attn_mod, "qkv") and isinstance(attn_mod.qkv, nn.Linear)):
            # Fallback: run original attention unchanged
            return attn_mod(x, **kwargs)

        # ------------------------------------------------------------------
        # 1. Compute Q, K, V
        # ------------------------------------------------------------------
        qkv = attn_mod.qkv(x)                                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                        # each (B, H, N, head_dim)

        # ------------------------------------------------------------------
        # 2. Apply RoPE perturbation to Q and K (if enabled)
        # ------------------------------------------------------------------
        if self.mode in ("rope_perturbation", "both"):
            freqs = self._rope_freqs(x.device, x.dtype)                # (num_heads, half)
            # Apply per-head RoPE: iterate over heads or vectorise
            # Vectorised: apply each head's freqs independently
            q_rot_parts = []
            k_rot_parts = []
            for h in range(self.num_heads):
                q_h = q[:, h:h+1, :, :]                                # (B, 1, N, head_dim)
                k_h = k[:, h:h+1, :, :]
                f_h = self.drop(freqs[h])                               # (half,)
                q_rot_parts.append(_apply_rope(q_h, f_h))
                k_rot_parts.append(_apply_rope(k_h, f_h))
            q = torch.cat(q_rot_parts, dim=1)                          # (B, H, N, head_dim)
            k = torch.cat(k_rot_parts, dim=1)

        # ------------------------------------------------------------------
        # 3. Compute attention logits
        # ------------------------------------------------------------------
        scale = attn_mod.scale if hasattr(attn_mod, "scale") else head_dim ** -0.5
        attn_logits = (q @ k.transpose(-2, -1)) * scale                # (B, H, N, N)

        # ------------------------------------------------------------------
        # 4. Add position bias (if enabled)
        # ------------------------------------------------------------------
        if self.mode in ("pos_bias", "both"):
            bias = self._pos_bias(N)                                    # (N, N)
            attn_logits = attn_logits + bias.unsqueeze(0).unsqueeze(0)  # broadcast (B, H, N, N)

        # ------------------------------------------------------------------
        # 5. Softmax + dropout + weighted sum
        # ------------------------------------------------------------------
        attn_weights = F.softmax(attn_logits, dim=-1)
        if hasattr(attn_mod, "attn_drop"):
            attn_weights = attn_mod.attn_drop(attn_weights)

        out = attn_weights @ v                                          # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)                     # (B, N, C)
        out = attn_mod.proj(out)
        if hasattr(attn_mod, "proj_drop"):
            out = attn_mod.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# MethodModule implementation
# ---------------------------------------------------------------------------

@register_method("rope_geometry")
class RoPEGeometryMethod(MethodModule):
    """RoPE Phase Perturbation and Relative Position Bias PEFT.

    Supports three modes:
      rope_perturbation: learnable phase_delta added to RoPE frequency parameters.
      pos_bias:          low-rank relative position bias U @ V^T added to attention logits.
      both:              apply both simultaneously.

    All parameters are zero-initialised so the method starts as a no-op
    (identity transformation), satisfying Property 10.

    Config keys:
        mode (str):          "rope_perturbation" | "pos_bias" | "both". Default "rope_perturbation".
        rank (int):          rank for position bias low-rank decomposition. Default 4.
        max_len (int):       maximum sequence length. Default 197 (ViT-B/16 with CLS token).
        num_layers (int):    number of transformer layers (informational). Default 12.
        num_heads (int):     number of attention heads (inferred from backbone if 0). Default 12.
        dim_per_head (int):  dimension per head (inferred from backbone if 0). Default 64.
        dropout (float):     dropout on phase_delta frequencies. Default 0.0.
    """

    def __init__(self, config: dict) -> None:
        self.mode: str = config.get("mode", "rope_perturbation")
        self.rank: int = int(config.get("rank", 4))
        self.max_len: int = int(config.get("max_len", 197))
        self.num_heads_cfg: int = int(config.get("num_heads", 12))
        self.dim_per_head_cfg: int = int(config.get("dim_per_head", 64))
        self.dropout: float = float(config.get("dropout", 0.0))

        if self.mode not in ("rope_perturbation", "pos_bias", "both"):
            raise ValueError(
                f"RoPEGeometry mode must be 'rope_perturbation', 'pos_bias', or 'both', "
                f"got '{self.mode}'"
            )
        if self.rank < 1:
            raise ValueError(f"RoPEGeometry rank must be >= 1, got {self.rank}")

        self._wrappers: List[RoPEGeometryWrapper] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Replace each transformer block's attention with a RoPEGeometryWrapper."""
        self._wrappers = []
        self._inject_recursive(backbone)
        if not self._wrappers:
            raise RuntimeError(
                "RoPEGeometryMethod.inject: no attention modules found. "
                "Ensure the backbone has timm-style Attention blocks with a .qkv attribute."
            )
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self._is_attention(child):
                num_heads, dim_per_head = self._infer_dims(child)
                wrapper = RoPEGeometryWrapper(
                    attn=child,
                    mode=self.mode,
                    num_heads=num_heads,
                    dim_per_head=dim_per_head,
                    max_len=self.max_len,
                    rank=self.rank,
                    dropout=self.dropout,
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

    def _infer_dims(self, attn: nn.Module):
        """Return (num_heads, dim_per_head) from a timm Attention module."""
        num_heads = getattr(attn, "num_heads", self.num_heads_cfg)
        hidden_dim = attn.proj.in_features
        dim_per_head = hidden_dim // num_heads
        return num_heads, dim_per_head

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for w in self._wrappers:
            if w.phase_delta is not None:
                params.append(w.phase_delta)
            if w.pos_bias_U is not None:
                params.append(w.pos_bias_U)
            if w.pos_bias_V is not None:
                params.append(w.pos_bias_V)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}

    def is_compatible_with(self, other: "MethodModule") -> bool:
        """RoPEGeometry is compatible with all other methods."""
        return True
