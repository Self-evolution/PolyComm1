"""NOAH — Neural Architecture Search for Optimal Adapter Hyperparameters.

NOAH uses a neural architecture search approach to find the optimal
combination of PEFT modules (adapter, LoRA, VPT) per layer. In the
simplified implementation used as a benchmark baseline, NOAH inserts a
combination of adapter and LoRA modules at each transformer block, with
learnable mixing weights (architecture parameters) that control the
contribution of each module.

During fine-tuning, both the module parameters and architecture weights
are optimized jointly (single-level optimization, as used in benchmarks).

Reference: Zhang et al., "NOAH: Neural Architecture Search for Parameter
Efficient Adaptation of Large Language Models", TMLR 2024.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


class NOAHAdapterBranch(nn.Module):
    """Bottleneck adapter branch: down → GELU → up."""

    def __init__(self, in_features: int, bottleneck_dim: int, init_scale: float = 1e-3) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, in_features)
        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class NOAHLoRABranch(nn.Module):
    """LoRA-style low-rank branch applied to the hidden state directly."""

    def __init__(self, in_features: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, in_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


class NOAHMixedBlock(nn.Module):
    """Wraps a ViT block with a NOAH mixed adapter (adapter + LoRA branches)."""

    def __init__(
        self,
        block: nn.Module,
        adapter_branch: NOAHAdapterBranch,
        lora_branch: NOAHLoRABranch,
        arch_weights: nn.Parameter,
    ) -> None:
        super().__init__()
        self.block = block
        self.adapter_branch = adapter_branch
        self.lora_branch = lora_branch
        self.arch_weights = arch_weights  # shape (2,), softmax → mix weights
        self._has_split = (
            hasattr(block, "attn") and hasattr(block, "mlp")
            and hasattr(block, "norm1") and hasattr(block, "norm2")
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        weights = torch.softmax(self.arch_weights, dim=0)
        if self._has_split:
            b = self.block
            drop = getattr(b, "drop_path", lambda t: t)
            x = x + drop(b.attn(b.norm1(x)))
            mlp_out = drop(b.mlp(b.norm2(x)))
            mixed = weights[0] * self.adapter_branch(x) + weights[1] * self.lora_branch(x)
            x = x + mlp_out + mixed
            return x
        out = self.block(x, **kwargs)
        if isinstance(out, tuple):
            h, *rest = out
            mixed = weights[0] * self.adapter_branch(h) + weights[1] * self.lora_branch(h)
            return (h + mixed, *rest)
        mixed = weights[0] * self.adapter_branch(out) + weights[1] * self.lora_branch(out)
        return out + mixed


@register_method("noah")
class NOAHMethod(MethodModule):
    """NOAH PEFT method — mixed adapter+LoRA with learnable architecture weights.

    Config keys:
        bottleneck_dim (int): adapter bottleneck dimension. Default 64.
        lora_rank (int): LoRA branch rank. Default 8.
        init_scale (float): near-zero init std for adapter. Default 1e-3.
    """

    def __init__(self, config: dict) -> None:
        self.bottleneck_dim: int = int(config.get("bottleneck_dim", 64))
        self.lora_rank: int = int(config.get("lora_rank", 8))
        self.init_scale: float = float(config.get("init_scale", 1e-3))
        self._modules_list: List[nn.Module] = []
        self._arch_params: List[nn.Parameter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._modules_list = []
        self._arch_params = []
        self._inject_recursive(backbone)
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self._is_vit_block(child):
                hidden_dim = self._infer_hidden_dim(child)
                if hidden_dim is None:
                    self._inject_recursive(child)
                    continue
                adapter_branch = NOAHAdapterBranch(
                    hidden_dim, self.bottleneck_dim, self.init_scale
                )
                lora_branch = NOAHLoRABranch(hidden_dim, self.lora_rank)
                arch_weights = nn.Parameter(torch.zeros(2))  # equal init
                self._arch_params.append(arch_weights)
                self._modules_list.extend([adapter_branch, lora_branch])
                setattr(module, name, NOAHMixedBlock(
                    child, adapter_branch, lora_branch, arch_weights
                ))
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
        params: List[nn.Parameter] = list(self._arch_params)
        for m in self._modules_list:
            params.extend(m.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
