"""FCLayer — Full fine-tuning of the final FC / classification head only.

Freezes the entire backbone and only trains the final fully-connected layer
(the classification head). This is the simplest possible transfer learning
baseline: linear probing.

Also registered as "linear_probe" for convenience.

Reference: Standard linear probing baseline used across PEFT benchmarks.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


@register_method("fclayer")
class FCLayerMethod(MethodModule):
    """FCLayer PEFT method — trains only the final FC / head layer.

    Freezes all backbone parameters. The task head (added externally by
    ExperimentRunner) remains trainable. This method itself injects nothing
    and returns zero trainable params from the backbone perspective.

    Config keys:
        unfreeze_last_norm (bool): also unfreeze the final LayerNorm before
            the head (e.g. ViT's norm layer). Default False.
        last_norm_name (str): name of the final norm module to unfreeze.
            Default "norm".
    """

    def __init__(self, config: dict) -> None:
        self.unfreeze_last_norm: bool = bool(config.get("unfreeze_last_norm", False))
        self.last_norm_name: str = str(config.get("last_norm_name", "norm"))
        self._trainable_params: List[nn.Parameter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._trainable_params = []

        # Freeze everything
        for param in backbone.parameters():
            param.requires_grad_(False)

        # Optionally unfreeze the final LayerNorm
        if self.unfreeze_last_norm:
            norm = getattr(backbone, self.last_norm_name, None)
            if norm is not None and isinstance(norm, nn.LayerNorm):
                for param in norm.parameters():
                    param.requires_grad_(True)
                    self._trainable_params.append(param)

        return backbone

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._trainable_params)

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self._trainable_params)
        return {"total": trainable, "trainable": trainable}
