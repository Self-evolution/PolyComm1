"""PolySSF: polynomial attention with SSF calibration.

PolyAttn improves attention expressivity, while SSF gives a very stable
per-channel calibration signal on small VTAB-style datasets. This hybrid keeps
the two parts deliberately compositional: inject PolyAttn first, then wrap the
selected linear / LayerNorm modules with SSF so the wrapped attention still
uses the calibrated qkv and projection paths.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.baselines.ssf import SSFMethod
from src.methods.novel.polyattn import PolyAttnMethod
from src.methods.registry import register_method


@register_method("polyssf")
class PolySSFMethod(MethodModule):
    """Compose PolyAttn with SSF scale/shift calibration.

    Config keys are the union of PolyAttn's keys plus:
        ssf_target_modules (list[str]): module-name patterns for SSF injection.
            Defaults to ["qkv", "proj", "norm"].
        inject_norm (bool): whether to apply SSF to matching LayerNorm modules.
            Defaults to True.
    """

    def __init__(self, config: dict) -> None:
        self.polyattn = PolyAttnMethod(config)
        ssf_config = {
            "target_modules": list(
                config.get(
                    "ssf_target_modules",
                    config.get("target_modules", ["qkv", "proj", "norm"]),
                )
            ),
            "inject_norm": config.get("inject_norm", True),
        }
        self.ssf = SSFMethod(ssf_config)

    def inject(self, backbone: nn.Module) -> nn.Module:
        backbone = self.polyattn.inject(backbone)
        backbone = self.ssf.inject(backbone)
        return backbone

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()
        for param in self.polyattn.get_trainable_params() + self.ssf.get_trainable_params():
            ident = id(param)
            if ident not in seen:
                params.append(param)
                seen.add(ident)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
