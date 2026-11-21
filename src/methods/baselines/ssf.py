"""SSF — Scale and Shift Features for ViT-style backbones.

Injects learnable per-channel scale (gamma) and shift (beta) parameters
after target linear layers. The forward pass becomes:
    out = scale * (W x + b) + shift
where scale and shift are initialized to 1 and 0 respectively, so the
initial output is identical to the original layer.

Reference: Lian et al., "Scaling & Shifting Your Features: A New Baseline
for Efficient Model Tuning", NeurIPS 2022.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


class SSFLinear(nn.Module):
    """Wraps an existing nn.Linear with SSF scale-and-shift adaptation."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        out_features = linear.out_features

        # Keep original weight/bias frozen (reference, not copy)
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = out_features

        # Learnable scale and shift, one per output channel
        self.ssf_scale = nn.Parameter(torch.ones(out_features))
        self.ssf_shift = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        return self.ssf_scale * out + self.ssf_shift


class SSFLayerNorm(nn.Module):
    """Wraps an existing nn.LayerNorm with SSF scale-and-shift adaptation."""

    def __init__(self, ln: nn.LayerNorm) -> None:
        super().__init__()
        num_features = ln.normalized_shape[0]

        # Keep original parameters frozen
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.weight = ln.weight  # original gamma
        self.bias = ln.bias      # original beta

        # Learnable SSF scale and shift on top of LayerNorm output
        self.ssf_scale = nn.Parameter(torch.ones(num_features))
        self.ssf_shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return self.ssf_scale * out + self.ssf_shift


@register_method("ssf")
class SSFMethod(MethodModule):
    """SSF PEFT method — Scale and Shift Features.

    Config keys:
        target_modules (list[str]): sub-module name patterns to inject.
            Supports "qkv", "proj", "fc1", "fc2", "norm", etc.
            Default ["qkv", "proj", "fc1", "fc2"].
        inject_norm (bool): also inject SSF into LayerNorm layers whose
            names match any pattern in target_modules. Default True.
    """

    def __init__(self, config: dict) -> None:
        self.target_modules: List[str] = list(
            config.get("target_modules", ["qkv", "proj", "fc1", "fc2"])
        )
        self.inject_norm: bool = bool(config.get("inject_norm", True))

        # Populated during inject()
        self._ssf_layers: List[nn.Module] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Walk the backbone and wrap matching layers with SSF adapters."""
        self._ssf_layers = []
        self._inject_recursive(backbone)
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_inject(name):
                ssf_layer = SSFLinear(child)
                # Freeze original weight/bias
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                setattr(module, name, ssf_layer)
                self._ssf_layers.append(ssf_layer)
            elif (
                self.inject_norm
                and isinstance(child, nn.LayerNorm)
                and self._should_inject(name)
            ):
                ssf_ln = SSFLayerNorm(child)
                if child.weight is not None:
                    child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                setattr(module, name, ssf_ln)
                self._ssf_layers.append(ssf_ln)
            else:
                self._inject_recursive(child)

    def _should_inject(self, name: str) -> bool:
        return any(target in name for target in self.target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self._ssf_layers:
            params.append(layer.ssf_scale)
            params.append(layer.ssf_shift)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
