"""DoRA — Weight-Decomposed Low-Rank Adaptation for ViT-style backbones.

Decomposes the pre-trained weight W into magnitude (m) and direction (V/||V||),
then applies LoRA only to the directional component:

    W' = m * (W + B @ A) / ||W + B @ A||_col

where m is initialized to ||W||_col (per-output-channel norm) so the initial
output is identical to the original layer.

Key insight: by separating magnitude and direction, DoRA achieves better
learning capacity and training stability than LoRA, especially at lower ranks.

Reference:
    Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024 (Oral).
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


class DoRALinear(nn.Module):
    """Wraps an existing nn.Linear with DoRA weight-decomposed adaptation."""

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

        # Keep original weight/bias frozen (reference)
        self.weight = linear.weight
        self.bias = linear.bias

        # Low-rank matrices (same init as LoRA)
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Learnable magnitude vector: initialized to column norms of W
        # m_i = ||W[i, :]||_2  for each output channel i
        with torch.no_grad():
            col_norms = linear.weight.norm(p=2, dim=1)  # (out_features,)
        self.magnitude = nn.Parameter(col_norms.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Directional component: W + scaling * B @ A
        lora_delta = self.lora_B @ self.lora_A  # (out, in)
        W_prime = self.weight + self.scaling * lora_delta

        # Normalize each row (output channel) to unit norm
        W_norm = W_prime / (W_prime.norm(p=2, dim=1, keepdim=True) + 1e-8)

        # Apply magnitude
        W_final = self.magnitude.unsqueeze(1) * W_norm

        # Dropout on input (same as LoRA)
        x = self.lora_dropout(x)
        return F.linear(x, W_final, self.bias)


@register_method("dora")
class DoRAMethod(MethodModule):
    """DoRA PEFT method — Weight-Decomposed Low-Rank Adaptation.

    Config keys:
        rank (int): low-rank dimension r. Default 8.
        alpha (float): scaling factor alpha. Default 16.
        target_modules (list[str]): which sub-module names to inject.
            Default ["qkv"].
        dropout (float): dropout on input before LoRA path. Default 0.0.
    """

    def __init__(self, config: dict) -> None:
        self.rank: int = config.get("rank", 8)
        self.alpha: float = float(config.get("alpha", 16))
        self.target_modules: List[str] = list(config.get("target_modules", ["qkv"]))
        self.dropout: float = float(config.get("dropout", 0.0))
        self._dora_layers: List[DoRALinear] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._dora_layers = []
        self._inject_recursive(backbone)
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_inject(name):
                dora_layer = DoRALinear(
                    child,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                setattr(module, name, dora_layer)
                self._dora_layers.append(dora_layer)
            else:
                self._inject_recursive(child)

    def _should_inject(self, name: str) -> bool:
        return any(target in name for target in self.target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self._dora_layers:
            params.append(layer.lora_A)
            params.append(layer.lora_B)
            params.append(layer.magnitude)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
