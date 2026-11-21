"""PolyComm: directional polynomial communication with calibrated DoRA QKV.

The method keeps the stable PolySSF composition pattern, but moves QKV
calibration into the packed DoRA path. This matters because a DoRA-wrapped QKV
projection bypasses the original qkv module, so regular SSF(qkv) would not see
the tensor used by attention.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch.nn as nn

from src.methods.novel.polyssf import PolySSFMethod
from src.methods.registry import register_method


@register_method("polycomm")
class PolyCommMethod(PolySSFMethod):
    """Compose DoRA-QKV, packed QKV calibration, value residuals, and PolySSF."""

    DEFAULTS = {
        "max_hops": 3,
        "mode": "dynamic",
        "headwise": True,
        "mlp_ratio": 0.03125,
        "parameterization": "delta",
        "delta_scale": 0.1,
        "coeff_activation": "tanh",
        "coeff_dropout": 0.0,
        "dropout_target": "high_order",
        "norm_constraint": 0.25,
        "coeff_trainable": False,
        "dora_qkv_rank": 8,
        "dora_qkv_alpha": 16,
        "dora_qkv_dropout": 0.0,
        "dora_qkv_ssf": True,
        "dora_qkv_train_bias": False,
        "value_adapter_rank": 4,
        "value_adapter_alpha": 8,
        "value_adapter_dropout": 0.0,
        "ssf_target_modules": ["proj", "norm"],
        "inject_norm": True,
        "train_bias": False,
        "bias_modules": None,
    }

    def __init__(self, config: dict) -> None:
        merged = deepcopy(self.DEFAULTS)
        merged.update(config)
        self.train_bias = bool(merged.get("train_bias", False))
        bias_modules = merged.get("bias_modules", None)
        self.bias_modules = None if bias_modules is None else list(bias_modules)
        self._bias_params: List[nn.Parameter] = []
        super().__init__(merged)

    def inject(self, backbone: nn.Module) -> nn.Module:
        backbone = super().inject(backbone)
        self._collect_bias_params(backbone)
        return backbone

    def _collect_bias_params(self, backbone: nn.Module) -> None:
        self._bias_params = []
        if not self.train_bias:
            return

        seen = {id(param) for param in super().get_trainable_params()}
        for name, param in backbone.named_parameters():
            if not (name.endswith(".bias") or name == "bias"):
                continue
            if id(param) in seen or not self._should_bias(name):
                continue
            param.requires_grad_(True)
            self._bias_params.append(param)
            seen.add(id(param))

    def _should_bias(self, name: str) -> bool:
        if self.bias_modules is None:
            return True
        return any(target in name for target in self.bias_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()
        for param in super().get_trainable_params() + self._bias_params:
            ident = id(param)
            if ident not in seen:
                params.append(param)
                seen.add(ident)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
