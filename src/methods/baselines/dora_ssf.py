"""DoRA-SSF hybrid adapter for ViT-style backbones.

This method combines DoRA on QKV projections with SSF-style scale/shift on
QKV, projection, and LayerNorm features. It is intended for small-data regimes
where DoRA gives strong directional updates and SSF adds cheap per-channel
calibration.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method
from src.methods.baselines.ssf import SSFLinear, SSFLayerNorm


class DoRASSFLinear(nn.Module):
    """DoRA linear layer with an SSF scale/shift on the adapted output."""

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

        self.weight = linear.weight
        self.bias = linear.bias

        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        with torch.no_grad():
            row_norms = linear.weight.norm(p=2, dim=1)
        self.magnitude = nn.Parameter(row_norms.clone())
        self.ssf_scale = nn.Parameter(torch.ones(linear.out_features))
        self.ssf_shift = nn.Parameter(torch.zeros(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_delta = self.lora_B @ self.lora_A
        weight = self.weight + self.scaling * lora_delta
        direction = weight / (weight.norm(p=2, dim=1, keepdim=True) + 1e-8)
        adapted = self.magnitude.unsqueeze(1) * direction
        out = F.linear(self.lora_dropout(x), adapted, self.bias)
        return self.ssf_scale * out + self.ssf_shift


@register_method("dora_ssf")
class DoRASSFMethod(MethodModule):
    """DoRA on QKV plus SSF calibration on selected linear/norm layers."""

    def __init__(self, config: dict) -> None:
        self.rank: int = int(config.get("rank", 8))
        self.alpha: float = float(config.get("alpha", 16))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.qkv_modules: List[str] = list(config.get("qkv_modules", ["qkv"]))
        self.ssf_target_modules: List[str] = list(
            config.get("ssf_target_modules", ["qkv", "proj", "norm"])
        )
        self.inject_norm: bool = bool(config.get("inject_norm", True))
        self._dora_ssf_layers: List[DoRASSFLinear] = []
        self._ssf_layers: List[nn.Module] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._dora_ssf_layers = []
        self._ssf_layers = []
        self._inject_recursive(backbone)
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_qkv(name):
                layer = DoRASSFLinear(
                    child,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                setattr(module, name, layer)
                self._dora_ssf_layers.append(layer)
            elif isinstance(child, nn.Linear) and self._should_ssf(name):
                layer = SSFLinear(child)
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                setattr(module, name, layer)
                self._ssf_layers.append(layer)
            elif (
                self.inject_norm
                and isinstance(child, nn.LayerNorm)
                and self._should_ssf(name)
            ):
                layer = SSFLayerNorm(child)
                if child.weight is not None:
                    child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                setattr(module, name, layer)
                self._ssf_layers.append(layer)
            else:
                self._inject_recursive(child)

    def _should_qkv(self, name: str) -> bool:
        return any(target in name for target in self.qkv_modules)

    def _should_ssf(self, name: str) -> bool:
        return any(target in name for target in self.ssf_target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self._dora_ssf_layers:
            params.extend(
                [
                    layer.lora_A,
                    layer.lora_B,
                    layer.magnitude,
                    layer.ssf_scale,
                    layer.ssf_shift,
                ]
            )
        for layer in self._ssf_layers:
            params.append(layer.ssf_scale)
            params.append(layer.ssf_shift)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
