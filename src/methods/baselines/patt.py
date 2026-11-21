"""PATT — Patch-Aware Token Tuning for video ViT backbones.

PATT introduces learnable temporal-aware tokens that are injected into
the attention of each transformer block. It is designed for video action
recognition, where temporal context across frames is important.

Each layer receives a set of learnable "temporal anchor" tokens that
attend to the full sequence (spatial + temporal patches) and whose
outputs are added back to the CLS token representation, providing a
lightweight temporal aggregation mechanism.

Reference: Inspired by temporal prompt tuning methods used in
VideoActionRecognition benchmarks (PATT-style temporal token injection).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class PATTBlock(nn.Module):
    """Wraps a ViT transformer block with PATT temporal anchor tokens.

    The anchor tokens are concatenated to the sequence, processed through
    the block, and their output is used to update the CLS token via a
    learned aggregation. The anchor tokens are then removed from the output.
    """

    def __init__(
        self,
        block: nn.Module,
        anchor_tokens: nn.Parameter,   # (1, num_anchors, hidden_dim)
        aggregator: nn.Linear,          # hidden_dim -> hidden_dim
        dropout: nn.Dropout,
        num_anchors: int,
    ) -> None:
        super().__init__()
        self.block = block
        self.anchor_tokens = anchor_tokens
        self.aggregator = aggregator
        self.dropout = dropout
        self.num_anchors = num_anchors

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        # Expand anchor tokens for the batch
        anchors = self.dropout(self.anchor_tokens.expand(B, -1, -1))

        # Append anchor tokens to the sequence (after all patch tokens)
        x_aug = torch.cat([x, anchors], dim=1)  # (B, N+num_anchors, C)

        # Run through the transformer block
        out = self.block(x_aug, **kwargs)
        if isinstance(out, tuple):
            hidden, *rest = out
        else:
            hidden = out
            rest = []

        # Extract anchor outputs and aggregate into CLS token
        anchor_out = hidden[:, N:, :]                    # (B, num_anchors, C)
        anchor_agg = anchor_out.mean(dim=1)              # (B, C)
        cls_update = self.aggregator(anchor_agg)         # (B, C)

        # Update CLS token with temporal anchor information
        result = hidden[:, :N, :].clone()
        result[:, 0, :] = result[:, 0, :] + cls_update

        if rest:
            return (result, *rest)
        return result


@register_method("patt")
class PATTMethod(MethodModule):
    """PATT — Patch-Aware Token Tuning PEFT method.

    Injects learnable temporal anchor tokens at each transformer block.
    The anchor tokens aggregate temporal context and update the CLS token,
    making it suitable for video action recognition tasks.

    Config keys:
        num_anchors (int): number of temporal anchor tokens per layer. Default 4.
        dropout (float): dropout on anchor token embeddings. Default 0.0.
        inject_layers (str): "all" or "last_N" (e.g. "last_6"). Default "all".
        aggregation (str): "mean" (default) — how anchor outputs are pooled.
    """

    def __init__(self, config: dict) -> None:
        self.num_anchors: int = int(config.get("num_anchors", 4))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.inject_layers: str = config.get("inject_layers", "all")
        self.aggregation: str = config.get("aggregation", "mean")

        self._anchor_params: List[nn.Parameter] = []
        self._aggregators: List[nn.Linear] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._anchor_params = []
        self._aggregators = []

        blocks, hidden_dim = self._find_blocks(backbone)
        if blocks is None:
            raise RuntimeError(
                "PATTMethod.inject: could not locate transformer blocks. "
                "Backbone must expose 'blocks', 'layers', or 'encoder.layer'."
            )

        num_layers = len(blocks)
        inject_set = self._resolve_inject_layers(num_layers)

        val = math.sqrt(6.0 / (hidden_dim + self.num_anchors))
        dropout = nn.Dropout(p=self.dropout)

        new_blocks = nn.Sequential()
        for i, blk in enumerate(blocks):
            if i not in inject_set:
                new_blocks.append(blk)
                continue

            anchor_tokens = nn.Parameter(
                torch.empty(1, self.num_anchors, hidden_dim).uniform_(-val, val)
            )
            aggregator = nn.Linear(hidden_dim, hidden_dim, bias=True)
            # Near-zero init so initial contribution is small
            nn.init.normal_(aggregator.weight, std=1e-3)
            nn.init.zeros_(aggregator.bias)

            self._anchor_params.append(anchor_tokens)
            self._aggregators.append(aggregator)

            new_blocks.append(PATTBlock(
                block=blk,
                anchor_tokens=anchor_tokens,
                aggregator=aggregator,
                dropout=dropout,
                num_anchors=self.num_anchors,
            ))

        self._set_blocks(backbone, new_blocks)
        return backbone

    def _resolve_inject_layers(self, num_layers: int) -> set:
        """Parse inject_layers config into a set of layer indices."""
        spec = self.inject_layers.strip()
        if spec == "all":
            return set(range(num_layers))
        if spec.startswith("last_"):
            try:
                n = int(spec[5:])
                return set(range(max(0, num_layers - n), num_layers))
            except ValueError:
                pass
        # Try comma-separated indices
        try:
            return {int(i.strip()) for i in spec.split(",")}
        except ValueError:
            raise ValueError(
                f"PATTMethod: invalid inject_layers '{spec}'. "
                "Use 'all', 'last_N', or comma-separated indices."
            )

    def _find_blocks(self, backbone: nn.Module):
        """Return (blocks, hidden_dim)."""
        for attr in ("blocks", "layers"):
            container = getattr(backbone, attr, None)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                hidden_dim = self._infer_hidden_dim(container[0])
                return container, hidden_dim
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            container = backbone.encoder.layer
            hidden_dim = self._infer_hidden_dim(container[0])
            return container, hidden_dim
        return None, None

    @staticmethod
    def _infer_hidden_dim(block: nn.Module) -> int:
        for attr in ("norm1", "norm2", "norm"):
            ln = getattr(block, attr, None)
            if isinstance(ln, nn.LayerNorm):
                return ln.normalized_shape[0]
        for m in block.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        raise RuntimeError("Cannot infer hidden_dim from transformer block.")

    def _set_blocks(self, backbone: nn.Module, new_blocks: nn.Module) -> None:
        if hasattr(backbone, "blocks") and isinstance(backbone.blocks, (nn.ModuleList, nn.Module)):
            backbone.blocks = new_blocks
        elif hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
            backbone.layers = new_blocks
        elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            backbone.encoder.layer = new_blocks
        else:
            raise RuntimeError("PATTMethod: cannot set blocks on backbone.")

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = list(self._anchor_params)
        for agg in self._aggregators:
            params.extend(agg.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
