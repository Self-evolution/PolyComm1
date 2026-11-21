"""PolyScale — polynomial attention plus feature scale calibration.

PolyAttn changes the attention support/value propagation. PolyScale keeps that
structure and adds a tiny residual scale/shift calibration on MLP and
normalization features. This gives the method a low-risk way to correct
task-specific channel statistics without turning the attention adapters into a
larger low-rank model.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.novel.polyattn import PolyAttnMethod, PolyAttnWrapper
from src.methods.registry import register_method


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class ScaleShiftLinear(nn.Module):
    """Apply learnable per-channel scale/shift after a frozen Linear layer."""

    def __init__(self, linear: nn.Linear, init_scale: float = 1.0) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.scale = nn.Parameter(torch.full((linear.out_features,), init_scale))
        self.shift = nn.Parameter(torch.zeros(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) * self.scale + self.shift


class ScaleShiftLayerNorm(nn.Module):
    """Apply learnable per-channel scale/shift after a frozen LayerNorm."""

    def __init__(self, norm: nn.LayerNorm, init_scale: float = 1.0) -> None:
        super().__init__()
        self.normalized_shape = norm.normalized_shape
        self.eps = norm.eps
        self.weight = norm.weight
        self.bias = norm.bias
        features = norm.normalized_shape[0]
        self.scale = nn.Parameter(torch.full((features,), init_scale))
        self.shift = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return out * self.scale + self.shift


@register_method("polyscale")
class PolyScaleMethod(MethodModule):
    """PolyAttn with residual feature scale/shift calibration.

    Config keys:
        All PolyAttn keys are forwarded unchanged.
        scale_targets: linear submodule names to calibrate. Default ["fc1", "fc2"].
        scale_inject_norm: whether to calibrate LayerNorm modules. Default True.
        scale_init: initial multiplicative scale. Default 1.0.
    """

    def __init__(self, config: dict) -> None:
        self.poly = PolyAttnMethod(config)
        self.scale_targets: List[str] = list(config.get("scale_targets", ["fc1", "fc2"]))
        self.scale_inject_norm: bool = _as_bool(config.get("scale_inject_norm", True))
        self.scale_init: float = float(config.get("scale_init", 1.0))
        self._scale_layers: List[nn.Module] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._scale_layers = []
        backbone = self.poly.inject(backbone)
        self._inject_scale_recursive(backbone)
        return backbone

    def _inject_scale_recursive(self, module: nn.Module) -> None:
        # Do not calibrate the newly added PolyAttn internals. The original
        # attention module lives under PolyAttnWrapper.attn; qkv/proj are not in
        # the default targets, and MLP/norm layers sit outside the wrapper.
        if isinstance(module, PolyAttnWrapper):
            return

        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_scale_linear(name):
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                scaled = ScaleShiftLinear(child, init_scale=self.scale_init)
                setattr(module, name, scaled)
                self._scale_layers.append(scaled)
            elif self.scale_inject_norm and isinstance(child, nn.LayerNorm):
                if child.weight is not None:
                    child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                scaled = ScaleShiftLayerNorm(child, init_scale=self.scale_init)
                setattr(module, name, scaled)
                self._scale_layers.append(scaled)
            else:
                self._inject_scale_recursive(child)

    def _should_scale_linear(self, name: str) -> bool:
        return any(target in name for target in self.scale_targets)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.poly.get_trainable_params())
        for layer in self._scale_layers:
            params.append(layer.scale)
            params.append(layer.shift)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
