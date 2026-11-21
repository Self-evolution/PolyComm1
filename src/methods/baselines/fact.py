"""FacT — Factor-Tuning for ViT-style backbones.

Two variants:
  - TK (Tucker decomposition): shared FacTu/FacTv projections + a shared
    3-D core tensor FacTc and a parameter matrix FacTp that slices FacTc
    per layer/module.  Fewer parameters, more sharing.
  - TT (Tensor-Train / per-module matrices): shared FacTu/FacTv projections
    + per-module small Linear layers (FacTs) that operate in the low-dim
    space.  More expressive, slightly more parameters.

Both variants inject delta updates into Q/K/V/proj (attention) and
fc1/fc2 (MLP) of every transformer block.  The update formula is:

    delta = FacTv( dropout( FacTu(x) @ FacTs ) ) * scale

where FacTs is either a slice of the shared core (TK) or a per-module
Linear (TT).

References:
  Jie & Deng, "FacT: Factor-Tuning for Lightweight Adaptation on Vision
  Transformer", AAAI 2023.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# Patched forward functions (bound to individual attention / MLP modules)
# ---------------------------------------------------------------------------

def _fact_tk_attn_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Patched Attention.forward for FacT-TK."""
    B, N, C = x.shape
    # Slice the core tensor for this block's 4 attention projections
    # FacTc: (dim, dim, dim), FacTp: (dim, total_cols) → slice 4 cols
    FacTc: torch.Tensor = self._fact_root.FacTc @ self._fact_root.FacTp[:, self._fact_idx: self._fact_idx + 4]
    # FacTc now: (dim, dim, 4)
    q_c, k_c, v_c, proj_c = FacTc[:, :, 0], FacTc[:, :, 1], FacTc[:, :, 2], FacTc[:, :, 3]

    qkv = self.qkv(x)
    u = self._fact_root.FacTu(x)  # (B, N, dim)
    q = self._fact_root.FacTv(self._fact_dp(u @ q_c))
    k = self._fact_root.FacTv(self._fact_dp(u @ k_c))
    v = self._fact_root.FacTv(self._fact_dp(u @ v_c))
    qkv = qkv + torch.cat([q, k, v], dim=2) * self._fact_s

    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj = proj + self._fact_root.FacTv(self._fact_dp(self._fact_root.FacTu(x) @ proj_c)) * self._fact_s
    x = self.proj_drop(proj)
    return x


def _fact_tk_mlp_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Patched Mlp.forward for FacT-TK."""
    B, N, C = x.shape
    dim = self._fact_dim
    # Slice 8 columns: first 4 for fc1, next 4 for fc2
    FacTc: torch.Tensor = self._fact_root.FacTc @ self._fact_root.FacTp[:, self._fact_idx: self._fact_idx + 8]
    fc1_c = FacTc[:, :, :4].reshape(dim, dim * 4)   # (dim, 4*dim)
    fc2_c = FacTc[:, :, 4:].reshape(dim, dim * 4)   # (dim, 4*dim)

    h = self.fc1(x)
    u = self._fact_root.FacTu(x)  # (B, N, dim)
    h = h + self._fact_root.FacTv(
        self._fact_dp(u @ fc1_c).reshape(B, N, 4, dim)
    ).reshape(B, N, 4 * C) * self._fact_s
    x = self.act(h)
    x = self.drop1(x) if hasattr(self, 'drop1') else self.drop(x)

    h = self.fc2(x)
    x_r = x.reshape(B, N, 4, C)
    h = h + self._fact_root.FacTv(
        self._fact_dp(self._fact_root.FacTu(x_r).reshape(B, N, 4 * dim) @ fc2_c.t())
    ) * self._fact_s
    x = self.drop2(h) if hasattr(self, 'drop2') else self.drop(h)
    return x


def _fact_tt_attn_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Patched Attention.forward for FacT-TT."""
    B, N, C = x.shape
    qkv = self.qkv(x)
    u = self._fact_root.FacTu(x)  # (B, N, dim)
    q = self._fact_root.FacTv(self._fact_dp(self._fact_q(u)))
    k = self._fact_root.FacTv(self._fact_dp(self._fact_k(u)))
    v = self._fact_root.FacTv(self._fact_dp(self._fact_v(u)))
    qkv = qkv + torch.cat([q, k, v], dim=2) * self._fact_s

    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj = proj + self._fact_root.FacTv(self._fact_dp(self._fact_proj(self._fact_root.FacTu(x)))) * self._fact_s
    x = self.proj_drop(proj)
    return x


def _fact_tt_mlp_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Patched Mlp.forward for FacT-TT."""
    B, N, C = x.shape
    dim = self._fact_dim
    h = self.fc1(x)
    u = self._fact_root.FacTu(x)  # (B, N, dim)
    h = h + self._fact_root.FacTv(
        self._fact_dp(self._fact_fc1(u)).reshape(B, N, 4, dim)
    ).reshape(B, N, 4 * C) * self._fact_s
    x = self.act(h)
    x = self.drop1(x) if hasattr(self, 'drop1') else self.drop(x)

    h = self.fc2(x)
    x_r = x.reshape(B, N, 4, C)
    h = h + self._fact_root.FacTv(
        self._fact_dp(self._fact_fc2(self._fact_root.FacTu(x_r).reshape(B, N, 4 * dim)))
    ) * self._fact_s
    x = self.drop2(h) if hasattr(self, 'drop2') else self.drop(h)
    return x


# ---------------------------------------------------------------------------
# Helper: detect timm-style Attention / Mlp modules
# ---------------------------------------------------------------------------

def _is_attention(module: nn.Module) -> bool:
    return (
        hasattr(module, "qkv")
        and hasattr(module, "proj")
        and hasattr(module, "attn_drop")
        and hasattr(module, "num_heads")
    )


def _is_mlp(module: nn.Module) -> bool:
    return (
        hasattr(module, "fc1")
        and hasattr(module, "fc2")
        and hasattr(module, "act")
    )


def _infer_hidden_dim(backbone: nn.Module) -> int:
    """Infer the hidden dimension from the first LayerNorm in the backbone."""
    for m in backbone.modules():
        if isinstance(m, nn.LayerNorm):
            return m.normalized_shape[0]
    raise RuntimeError("Cannot infer hidden_dim from backbone — no LayerNorm found.")


# ---------------------------------------------------------------------------
# MethodModule implementations
# ---------------------------------------------------------------------------

@register_method("fact_tk")
class FacTTKMethod(MethodModule):
    """FacT-TK: Tucker-decomposition variant.

    Shared parameters on the backbone root:
      FacTu: Linear(hidden_dim → dim, bias=False)
      FacTv: Linear(dim → hidden_dim, bias=False)  — zero-init
      FacTc: Parameter(dim, dim, dim)               — Xavier init
      FacTp: Parameter(dim, num_cols)               — Xavier init
        where num_cols = 12 * num_blocks (4 attn + 8 mlp per block)

    Config keys:
        dim (int): low-rank bottleneck dimension. Default 8.
        scale (float): delta output scale. Default 1.0.
        dropout (float): dropout inside FacT path. Default 0.1.
    """

    def __init__(self, config: dict) -> None:
        self.dim: int = int(config.get("dim", 8))
        self.scale: float = float(config.get("scale", 1.0))
        self.dropout: float = float(config.get("dropout", 0.1))

        self._backbone: Optional[nn.Module] = None
        self._patched_modules: List[nn.Module] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._patched_modules = []
        hidden_dim = _infer_hidden_dim(backbone)
        dim = self.dim

        # Count blocks to size FacTp
        num_attn = sum(1 for m in backbone.modules() if _is_attention(m))
        num_mlp = sum(1 for m in backbone.modules() if _is_mlp(m))
        total_cols = num_attn * 4 + num_mlp * 8

        # Attach shared parameters to backbone root
        backbone.FacTu = nn.Linear(hidden_dim, dim, bias=False)
        backbone.FacTv = nn.Linear(dim, hidden_dim, bias=False)
        backbone.FacTp = nn.Parameter(torch.zeros(dim, total_cols))
        backbone.FacTc = nn.Parameter(torch.zeros(dim, dim, dim))

        nn.init.zeros_(backbone.FacTv.weight)
        nn.init.xavier_uniform_(backbone.FacTc)
        nn.init.xavier_uniform_(backbone.FacTp)

        self._backbone = backbone

        # Create a plain container for shared params to avoid circular module refs
        # (storing backbone directly on submodules causes infinite recursion in .train()/.eval())
        import types
        fact_shared = types.SimpleNamespace(
            FacTu=backbone.FacTu,
            FacTv=backbone.FacTv,
            FacTp=backbone.FacTp,
            FacTc=backbone.FacTc,
        )

        # Patch attention and MLP modules
        col_idx = 0
        for module in backbone.modules():
            if _is_attention(module):
                dp = nn.Dropout(self.dropout)
                module._fact_root = fact_shared
                module._fact_dp = dp
                module._fact_s = self.scale
                module._fact_idx = col_idx
                module.forward = _fact_tk_attn_forward.__get__(module, module.__class__)
                self._patched_modules.append(module)
                col_idx += 4
            elif _is_mlp(module):
                dp = nn.Dropout(self.dropout)
                module._fact_root = fact_shared
                module._fact_dp = dp
                module._fact_s = self.scale
                module._fact_idx = col_idx
                module._fact_dim = dim
                module.forward = _fact_tk_mlp_forward.__get__(module, module.__class__)
                self._patched_modules.append(module)
                col_idx += 8

        return backbone

    def get_trainable_params(self) -> List[nn.Parameter]:
        if self._backbone is None:
            return []
        params: List[nn.Parameter] = []
        params.extend(self._backbone.FacTu.parameters())
        params.extend(self._backbone.FacTv.parameters())
        params.append(self._backbone.FacTp)
        params.append(self._backbone.FacTc)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}


@register_method("fact_tt")
class FacTTTMethod(MethodModule):
    """FacT-TT: Tensor-Train / per-module matrices variant.

    Shared parameters on the backbone root:
      FacTu: Linear(hidden_dim → dim, bias=False)
      FacTv: Linear(dim → hidden_dim, bias=False)  — zero-init

    Per attention module:
      q_FacTs, k_FacTs, v_FacTs, proj_FacTs: Linear(dim → dim, bias=False)

    Per MLP module:
      fc1_FacTs: Linear(dim → 4*dim, bias=False)
      fc2_FacTs: Linear(4*dim → dim, bias=False)

    Config keys:
        dim (int): low-rank bottleneck dimension. Default 8.
        scale (float): delta output scale. Default 1.0.
        dropout (float): dropout inside FacT path. Default 0.1.
    """

    def __init__(self, config: dict) -> None:
        self.dim: int = int(config.get("dim", 8))
        self.scale: float = float(config.get("scale", 1.0))
        self.dropout: float = float(config.get("dropout", 0.1))

        self._backbone: Optional[nn.Module] = None
        self._extra_modules: List[nn.Module] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._extra_modules = []
        hidden_dim = _infer_hidden_dim(backbone)
        dim = self.dim

        # Shared up/down projections
        backbone.FacTu = nn.Linear(hidden_dim, dim, bias=False)
        backbone.FacTv = nn.Linear(dim, hidden_dim, bias=False)
        nn.init.zeros_(backbone.FacTv.weight)

        self._backbone = backbone

        # Plain container to avoid circular module refs
        import types
        fact_shared = types.SimpleNamespace(
            FacTu=backbone.FacTu,
            FacTv=backbone.FacTv,
        )

        for module in backbone.modules():
            if _is_attention(module):
                dp = nn.Dropout(self.dropout)
                q_s = nn.Linear(dim, dim, bias=False)
                k_s = nn.Linear(dim, dim, bias=False)
                v_s = nn.Linear(dim, dim, bias=False)
                proj_s = nn.Linear(dim, dim, bias=False)

                module._fact_root = fact_shared
                module._fact_dp = dp
                module._fact_s = self.scale
                module._fact_q = q_s
                module._fact_k = k_s
                module._fact_v = v_s
                module._fact_proj = proj_s
                module.forward = _fact_tt_attn_forward.__get__(module, module.__class__)

                self._extra_modules.extend([q_s, k_s, v_s, proj_s])

            elif _is_mlp(module):
                dp = nn.Dropout(self.dropout)
                fc1_s = nn.Linear(dim, dim * 4, bias=False)
                fc2_s = nn.Linear(4 * dim, dim, bias=False)

                module._fact_root = fact_shared
                module._fact_dp = dp
                module._fact_s = self.scale
                module._fact_dim = dim
                module._fact_fc1 = fc1_s
                module._fact_fc2 = fc2_s
                module.forward = _fact_tt_mlp_forward.__get__(module, module.__class__)

                self._extra_modules.extend([fc1_s, fc2_s])

        return backbone

    def get_trainable_params(self) -> List[nn.Parameter]:
        if self._backbone is None:
            return []
        params: List[nn.Parameter] = []
        params.extend(self._backbone.FacTu.parameters())
        params.extend(self._backbone.FacTv.parameters())
        for m in self._extra_modules:
            params.extend(m.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
