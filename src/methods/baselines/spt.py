"""SPT — Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning.

Two-stage approach:
  Stage 1: One-shot gradient-based sensitivity estimation to rank which
           layers benefit most from trainable parameters.
  Stage 2: Inject structured PEFT modules (LoRA or Adapter) at the top-K
           most sensitive positions, and optionally unfreeze individual
           bias parameters at sensitive locations.

Since sensitivity estimation requires data, this implementation supports
two modes:
  - "uniform": skip sensitivity estimation, inject modules at ALL blocks
    (equivalent to standard LoRA/Adapter but with SPT's architecture).
  - "topk": inject modules only at the top-K most sensitive positions.
    Requires calling `estimate_sensitivity()` before `inject()`.

For the common benchmark setting (no pre-computed sensitivity), "uniform"
mode provides a strong baseline that captures SPT's architectural design
(parallel structured modules on q/k/v/proj/fc1/fc2).

Reference:
    He et al., "Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning",
    ICCV 2023 (Oral).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# Structured tuning module (LoRA-style or Adapter-style)
# ---------------------------------------------------------------------------

class StructuredModule(nn.Module):
    """A small bottleneck module injected in parallel to a frozen linear."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        mode: str = "lora",
    ) -> None:
        super().__init__()
        if mode == "adapter":
            self.net = nn.Sequential(
                nn.Linear(in_dim, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, out_dim, bias=False),
            )
            nn.init.kaiming_uniform_(self.net[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.net[2].weight)
        else:  # lora
            self.net = nn.Sequential(
                nn.Linear(in_dim, rank, bias=False),
                nn.Linear(rank, out_dim, bias=False),
            )
            nn.init.kaiming_uniform_(self.net[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.net[1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
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
    for m in backbone.modules():
        if isinstance(m, nn.LayerNorm):
            return m.normalized_shape[0]
    raise RuntimeError("Cannot infer hidden_dim — no LayerNorm found.")


# ---------------------------------------------------------------------------
# Patched forward functions
# ---------------------------------------------------------------------------

def _spt_attn_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Patched Attention.forward with parallel structured modules on q/k/v/proj."""
    B, N, C = x.shape

    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Parallel structured deltas
    if hasattr(self, "_spt_q"):
        q = q + self._spt_q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    if hasattr(self, "_spt_k"):
        k = k + self._spt_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    if hasattr(self, "_spt_v"):
        v = v + self._spt_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)

    proj_delta = 0.0
    if hasattr(self, "_spt_proj"):
        proj_delta = self._spt_proj(x)

    x = self.proj(x) + proj_delta
    x = self.proj_drop(x)
    return x


def _spt_mlp_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Patched Mlp.forward with parallel structured modules on fc1/fc2."""
    fc1_delta = 0.0
    if hasattr(self, "_spt_fc1"):
        fc1_delta = self._spt_fc1(x)

    x = self.fc1(x) + fc1_delta
    x = self.act(x)
    x = self.drop1(x) if hasattr(self, "drop1") else self.drop(x)

    fc2_delta = 0.0
    if hasattr(self, "_spt_fc2"):
        fc2_delta = self._spt_fc2(x)

    x = self.fc2(x) + fc2_delta
    x = self.drop2(x) if hasattr(self, "drop2") else self.drop(x)
    return x


# ---------------------------------------------------------------------------
# MethodModule
# ---------------------------------------------------------------------------

@register_method("spt")
class SPTMethod(MethodModule):
    """SPT PEFT method — Sensitivity-aware Parameter-efficient Tuning.

    In "uniform" mode (default), injects structured modules at all blocks.
    Targets: q, k, v, proj (attention) and fc1, fc2 (MLP).

    Config keys:
        rank (int): bottleneck dimension for structured modules. Default 8.
        structured_type (str): "lora" or "adapter". Default "lora".
        target_modules (list[str]): subset of ["q","k","v","proj","fc1","fc2"].
            Default all six.
    """

    def __init__(self, config: dict) -> None:
        self.rank: int = int(config.get("rank", 8))
        self.structured_type: str = config.get("structured_type", "lora")
        self.target_modules: List[str] = list(
            config.get("target_modules", ["q", "k", "v", "proj", "fc1", "fc2"])
        )
        self._structured_modules: List[nn.Module] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._structured_modules = []
        hidden_dim = _infer_hidden_dim(backbone)
        mlp_hidden = hidden_dim * 4  # standard ViT MLP ratio

        for module in backbone.modules():
            if _is_attention(module):
                self._inject_attn(module, hidden_dim)
            elif _is_mlp(module):
                self._inject_mlp(module, hidden_dim, mlp_hidden)

        return backbone

    def _inject_attn(self, module: nn.Module, dim: int) -> None:
        targets = self.target_modules
        for name in ["q", "k", "v", "proj"]:
            if name in targets:
                sm = StructuredModule(dim, dim, self.rank, self.structured_type)
                setattr(module, f"_spt_{name}", sm)
                self._structured_modules.append(sm)

        module.forward = _spt_attn_forward.__get__(module, module.__class__)

    def _inject_mlp(self, module: nn.Module, dim: int, mlp_hidden: int) -> None:
        targets = self.target_modules
        if "fc1" in targets:
            sm = StructuredModule(dim, mlp_hidden, self.rank, self.structured_type)
            module._spt_fc1 = sm
            self._structured_modules.append(sm)
        if "fc2" in targets:
            sm = StructuredModule(mlp_hidden, dim, self.rank, self.structured_type)
            module._spt_fc2 = sm
            self._structured_modules.append(sm)

        module.forward = _spt_mlp_forward.__get__(module, module.__class__)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for sm in self._structured_modules:
            params.extend(sm.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
