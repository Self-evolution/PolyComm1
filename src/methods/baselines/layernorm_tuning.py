"""LayerNorm Tuning — trains only LayerNorm scale and shift parameters.

All weight matrices and bias terms (except LayerNorm) remain frozen.
Only the gamma (weight) and beta (bias) of each LayerNorm are trainable.

Reference: Used as a baseline in multiple PEFT papers; also known as
"LN-Tuning" or "lntune".
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


@register_method("layernorm_tuning")
class LayerNormTuningMethod(MethodModule):
    """LayerNorm Tuning PEFT method — trains only LayerNorm parameters.

    Config keys:
        target_modules (list[str] | None): if provided, only unfreeze
            LayerNorm layers whose name contains one of these strings.
            Default None (all LayerNorm layers).
    """

    def __init__(self, config: dict) -> None:
        self.target_modules: List[str] | None = config.get("target_modules", None)
        self._trainable_params: List[nn.Parameter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._trainable_params = []

        # Freeze everything first
        for param in backbone.parameters():
            param.requires_grad_(False)

        # Unfreeze LayerNorm parameters
        for mod_name, module in backbone.named_modules():
            if isinstance(module, nn.LayerNorm):
                if self._should_inject(mod_name):
                    if module.weight is not None:
                        module.weight.requires_grad_(True)
                        self._trainable_params.append(module.weight)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)
                        self._trainable_params.append(module.bias)

        return backbone

    def _should_inject(self, name: str) -> bool:
        if self.target_modules is None:
            return True
        return any(target in name for target in self.target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._trainable_params)

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self._trainable_params)
        return {"total": trainable, "trainable": trainable}
