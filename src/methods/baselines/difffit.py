"""DiffFit — Differentiable Fine-Tuning via bias + LayerNorm parameters.

Trains only bias terms AND LayerNorm scale (gamma) / shift (beta) parameters.
All weight matrices remain frozen. This is an extension of BitFit that also
includes LayerNorm parameters.

Reference: Xie et al., "DiffFit: Unlocking Transferability of Large Diffusion
Models via Simple Parameter-Efficient Fine-Tuning", ICCV 2023.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


@register_method("difffit")
class DiffFitMethod(MethodModule):
    """DiffFit PEFT method — trains bias terms and LayerNorm parameters.

    Config keys:
        target_modules (list[str] | None): if provided, only unfreeze
            parameters in layers whose name contains one of these strings.
            Default None (all bias + LayerNorm params).
    """

    def __init__(self, config: dict) -> None:
        self.target_modules: List[str] | None = config.get("target_modules", None)
        self._trainable_params: List[nn.Parameter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._trainable_params = []

        for name, param in backbone.named_parameters():
            param.requires_grad_(False)

        for mod_name, module in backbone.named_modules():
            if isinstance(module, nn.LayerNorm):
                if self._should_inject(mod_name):
                    if module.weight is not None:
                        module.weight.requires_grad_(True)
                        self._trainable_params.append(module.weight)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)
                        self._trainable_params.append(module.bias)

        seen_ids = {id(p) for p in self._trainable_params}
        for param_name, param in backbone.named_parameters():
            if self._is_bias(param_name) and self._should_inject(param_name):
                param.requires_grad_(True)
                if id(param) not in seen_ids:
                    self._trainable_params.append(param)
                    seen_ids.add(id(param))

        return backbone

    def _is_bias(self, param_name: str) -> bool:
        return param_name.endswith(".bias") or param_name == "bias"

    def _should_inject(self, name: str) -> bool:
        if self.target_modules is None:
            return True
        return any(target in name for target in self.target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._trainable_params)

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self._trainable_params)
        return {"total": trainable, "trainable": trainable}
