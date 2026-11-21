"""BitFit — Bias-terms Fine-Tuning.

Only the bias parameters of the backbone are made trainable.
All weight matrices remain frozen.

Reference: Ben Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning
for Transformer-based Masked Language-models", ACL 2022.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


@register_method("bitfit")
class BitFitMethod(MethodModule):
    """BitFit PEFT method — trains only bias parameters.

    Config keys:
        target_modules (list[str] | None): if provided, only unfreeze biases
            in layers whose name contains one of these strings.
            Default None (all bias parameters).
    """

    def __init__(self, config: dict) -> None:
        self.target_modules: List[str] | None = config.get("target_modules", None)
        self._bias_params: List[nn.Parameter] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Freeze all weights; unfreeze bias parameters."""
        self._bias_params = []

        for name, param in backbone.named_parameters():
            if self._is_bias(name):
                if self._should_inject(name):
                    param.requires_grad_(True)
                    self._bias_params.append(param)
                else:
                    param.requires_grad_(False)
            else:
                param.requires_grad_(False)

        return backbone

    def _is_bias(self, param_name: str) -> bool:
        return param_name.endswith(".bias") or param_name == "bias"

    def _should_inject(self, param_name: str) -> bool:
        if self.target_modules is None:
            return True
        return any(target in param_name for target in self.target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._bias_params)

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self._bias_params)
        return {"total": trainable, "trainable": trainable}
