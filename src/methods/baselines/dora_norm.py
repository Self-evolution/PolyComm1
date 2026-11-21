"""DoRA with trainable LayerNorm calibration."""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method
from src.methods.baselines.dora import DoRALinear


@register_method("dora_norm")
class DoRANormMethod(MethodModule):
    """DoRA on target Linear modules plus LayerNorm affine tuning."""

    def __init__(self, config: dict) -> None:
        self.rank: int = int(config.get("rank", 8))
        self.alpha: float = float(config.get("alpha", 16))
        self.target_modules: List[str] = list(config.get("target_modules", ["qkv"]))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.norm_modules: List[str] = list(config.get("norm_modules", ["norm"]))
        self.train_bias: bool = bool(config.get("train_bias", False))
        self.bias_modules = config.get("bias_modules", None)
        if self.bias_modules is not None:
            self.bias_modules = list(self.bias_modules)

        self._dora_layers: List[DoRALinear] = []
        self._norm_params: List[nn.Parameter] = []
        self._bias_params: List[nn.Parameter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._dora_layers = []
        self._norm_params = []
        self._bias_params = []
        self._inject_dora_recursive(backbone)
        self._collect_norm_and_bias_params(backbone)
        return backbone

    def _inject_dora_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_dora(name):
                layer = DoRALinear(
                    child,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                setattr(module, name, layer)
                self._dora_layers.append(layer)
            else:
                self._inject_dora_recursive(child)

    def _collect_norm_and_bias_params(self, backbone: nn.Module) -> None:
        for name, module in backbone.named_modules():
            if isinstance(module, nn.LayerNorm) and self._should_norm(name):
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                    self._norm_params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
                    self._norm_params.append(module.bias)

        if not self.train_bias:
            return
        seen = {id(param) for param in self._norm_params}
        for name, param in backbone.named_parameters():
            if not name.endswith(".bias") and name != "bias":
                continue
            if id(param) in seen or not self._should_bias(name):
                continue
            param.requires_grad_(True)
            self._bias_params.append(param)
            seen.add(id(param))

    def _should_dora(self, name: str) -> bool:
        return any(target in name for target in self.target_modules)

    def _should_norm(self, name: str) -> bool:
        return any(target in name for target in self.norm_modules)

    def _should_bias(self, name: str) -> bool:
        if self.bias_modules is None:
            return True
        return any(target in name for target in self.bias_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self._dora_layers:
            params.append(layer.lora_A)
            params.append(layer.lora_B)
            params.append(layer.magnitude)
        params.extend(self._norm_params)
        params.extend(self._bias_params)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
