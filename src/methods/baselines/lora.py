"""LoRA — Low-Rank Adaptation for ViT-style backbones.

Injects low-rank A/B matrices into Q/K/V (or other) projection layers.
Formula: W' = W + (alpha/rank) * B @ A
  - A: initialized with Kaiming uniform (random Gaussian-like)
  - B: initialized to zero  →  initial output equals original output
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with LoRA low-rank adaptation."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Keep original weight/bias frozen
        self.weight = linear.weight  # reference, not a copy
        self.bias = linear.bias

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Standard LoRA init: A ~ Kaiming uniform, B = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base + lora_out * self.scaling


@register_method("lora")
class LoRAMethod(MethodModule):
    """LoRA PEFT method.

    Config keys:
        rank (int): low-rank dimension r. Default 8.
        alpha (float): scaling factor alpha. Default 16.
        target_modules (list[str]): which sub-module names to inject.
            Supports "qkv" (combined projection), "q", "k", "v", "proj", etc.
            Default ["qkv"].
        dropout (float): dropout applied to the input before LoRA path. Default 0.0.
    """

    def __init__(self, config: dict) -> None:
        self.rank: int = config.get("rank", 8)
        self.alpha: float = float(config.get("alpha", 16))
        self.target_modules: List[str] = list(config.get("target_modules", ["qkv"]))
        self.dropout: float = float(config.get("dropout", 0.0))

        # Populated during inject()
        self._lora_layers: List[LoRALinear] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Walk the backbone and replace matching Linear layers with LoRALinear."""
        self._lora_layers = []
        self._inject_recursive(backbone)
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_inject(name):
                lora_layer = LoRALinear(
                    child,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                # Freeze original weight/bias
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)

                setattr(module, name, lora_layer)
                self._lora_layers.append(lora_layer)
            else:
                self._inject_recursive(child)

    def _should_inject(self, name: str) -> bool:
        """Return True if this layer name matches any target module pattern."""
        return any(target in name for target in self.target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self._lora_layers:
            params.append(layer.lora_A)
            params.append(layer.lora_B)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        # Total includes frozen weights referenced by LoRA layers
        total_lora = sum(
            layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            + layer.lora_A.numel() + layer.lora_B.numel()
            for layer in self._lora_layers
        )
        return {"total": total_lora, "trainable": trainable}
