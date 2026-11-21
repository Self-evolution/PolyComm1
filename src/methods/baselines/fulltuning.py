"""Full Tuning — full fine-tuning baseline (control group).

Unfreezes all backbone parameters so every weight is updated during training.
This serves as the upper-bound reference for PEFT comparisons.

Reference: Requirements 4.2, 4.5
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


@register_method("fulltuning")
class FullTuningMethod(MethodModule):
    """Full fine-tuning — all backbone parameters are trainable.

    Config keys:
        None (no method-specific hyperparameters).
    """

    def __init__(self, config: dict) -> None:
        self._all_params: List[nn.Parameter] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Unfreeze all backbone parameters for full fine-tuning."""
        self._all_params = []
        for param in backbone.parameters():
            param.requires_grad_(True)
            self._all_params.append(param)
        return backbone

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._all_params)

    def get_param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self._all_params)
        return {"total": total, "trainable": total}
