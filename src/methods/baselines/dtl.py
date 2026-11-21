"""DTL — Decomposed Transfer Learning for ViT-style backbones.

Decomposes the weight update into a low-rank component applied to the
attention projection layers. DTL learns a shared basis across layers and
per-layer coefficients, reducing the total parameter count compared to
per-layer LoRA.

The update is: W' = W + U @ diag(s) @ V^T
where U and V are shared across layers and s is a per-layer diagonal.

Reference: Inspired by "Parameter-Efficient Transfer Learning" benchmarks;
DTL uses a decomposed low-rank structure with shared basis matrices.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


class DTLLinear(nn.Module):
    """Linear layer with DTL decomposed low-rank update.

    W' x = W x + (U @ diag(s) @ V^T) x
         = W x + U @ (s * (V^T x))
    """

    def __init__(
        self,
        linear: nn.Linear,
        U: nn.Parameter,  # shared basis (out_features, rank)
        V: nn.Parameter,  # shared basis (in_features, rank)
        rank: int,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Shared bases (not owned here, registered in parent)
        self.U = U
        self.V = V

        # Per-layer diagonal coefficients
        self.s = nn.Parameter(torch.zeros(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        # low-rank update: U @ diag(s) @ V^T @ x
        vx = x @ self.V[:self.in_features]          # (..., rank)
        svx = vx * self.s                            # (..., rank)
        update = svx @ self.U[:self.out_features].t()  # (..., out_features)
        return base + update


@register_method("dtl")
class DTLMethod(MethodModule):
    """DTL PEFT method — Decomposed Transfer Learning.

    Learns shared basis matrices U, V across all injected layers, with
    per-layer diagonal scaling vectors s.

    Config keys:
        rank (int): decomposition rank. Default 8.
        target_modules (list[str]): layer name patterns to inject.
            Default ["qkv", "proj"].
    """

    def __init__(self, config: dict) -> None:
        self.rank: int = int(config.get("rank", 8))
        self.target_modules: List[str] = list(
            config.get("target_modules", ["qkv", "proj"])
        )
        self._dtl_layers: List[DTLLinear] = []
        self._shared_U: Optional[nn.Parameter] = None
        self._shared_V: Optional[nn.Parameter] = None

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._dtl_layers = []
        # First pass: find all target linears to determine max dims
        targets = self._collect_targets(backbone)
        if not targets:
            return backbone

        max_out = max(m.out_features for _, m in targets)
        max_in = max(m.in_features for _, m in targets)

        # Shared basis matrices
        self._shared_U = nn.Parameter(torch.empty(max_out, self.rank))
        self._shared_V = nn.Parameter(torch.empty(max_in, self.rank))
        nn.init.normal_(self._shared_U, std=0.02)
        nn.init.zeros_(self._shared_V)

        # Register on backbone so they appear in state_dict
        backbone.register_parameter("_dtl_U", self._shared_U)
        backbone.register_parameter("_dtl_V", self._shared_V)

        # Second pass: replace layers
        self._inject_recursive(backbone)
        return backbone

    def _collect_targets(self, module: nn.Module) -> List:
        results = []
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_inject(name):
                results.append((name, child))
            else:
                results.extend(self._collect_targets(child))
        return results

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and self._should_inject(name):
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                dtl_layer = DTLLinear(child, self._shared_U, self._shared_V, self.rank)
                setattr(module, name, dtl_layer)
                self._dtl_layers.append(dtl_layer)
            else:
                self._inject_recursive(child)

    def _should_inject(self, name: str) -> bool:
        return any(t in name for t in self.target_modules)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self._shared_U is not None:
            params.append(self._shared_U)
        if self._shared_V is not None:
            params.append(self._shared_V)
        for layer in self._dtl_layers:
            params.append(layer.s)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
