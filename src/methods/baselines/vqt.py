"""VQT — Visual Query Tuning for ViT-style backbones.

Inserts learnable query tokens at each transformer layer. Unlike VPT which
prepends prompt tokens to the full sequence, VQT injects dedicated query
tokens that interact with the existing sequence via cross-attention-style
injection at every layer.

Reference: Tu et al., "Visual Query Tuning: Towards Effective Usage of
Intermediate Representations for Parameter and Memory Efficient Transfer
Learning", CVPR 2023.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class VQTBlock(nn.Module):
    """Wraps a ViT block to inject learnable query tokens at this layer.

    The query tokens are prepended to the sequence (after CLS) before the
    block forward pass, then stripped from the output so downstream layers
    see the original sequence length.
    """

    def __init__(
        self,
        block: nn.Module,
        query_tokens: nn.Parameter,
        dropout: nn.Dropout,
        num_tokens: int,
    ) -> None:
        super().__init__()
        self.block = block
        self.query_tokens = query_tokens
        self.dropout = dropout
        self.num_tokens = num_tokens

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        queries = self.dropout(self.query_tokens.expand(B, -1, -1))
        # Prepend queries after CLS token
        x_aug = torch.cat([x[:, :1], queries, x[:, 1:]], dim=1)
        out = self.block(x_aug, **kwargs)
        if isinstance(out, tuple):
            h, *rest = out
            # Remove query tokens from output
            h = torch.cat([h[:, :1], h[:, 1 + self.num_tokens:]], dim=1)
            return (h, *rest)
        # Remove query tokens
        return torch.cat([out[:, :1], out[:, 1 + self.num_tokens:]], dim=1)


@register_method("vqt")
class VQTMethod(MethodModule):
    """VQT PEFT method — Visual Query Tuning.

    Config keys:
        num_tokens (int): number of query tokens per layer. Default 10.
        dropout (float): dropout on query embeddings. Default 0.0.
        patch_size (int): patch size for Xavier init scale. Default 16.
    """

    def __init__(self, config: dict) -> None:
        self.num_tokens: int = int(config.get("num_tokens", 10))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.patch_size: int = int(config.get("patch_size", 16))
        self._query_params: List[nn.Parameter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._query_params = []
        blocks, hidden_dim = self._find_blocks(backbone)
        if blocks is None:
            raise RuntimeError(
                "VQTMethod.inject: could not locate transformer blocks. "
                "Backbone must expose 'blocks', 'layers', or 'encoder.layer'."
            )

        val = math.sqrt(6.0 / float(3 * self.patch_size * self.patch_size + hidden_dim))
        dropout = nn.Dropout(p=self.dropout)

        new_blocks = nn.Sequential()
        for blk in blocks:
            query_tokens = nn.Parameter(
                torch.empty(1, self.num_tokens, hidden_dim).uniform_(-val, val)
            )
            self._query_params.append(query_tokens)
            new_blocks.append(VQTBlock(blk, query_tokens, dropout, self.num_tokens))

        self._set_blocks(backbone, new_blocks)
        return backbone

    def _find_blocks(self, backbone: nn.Module):
        if hasattr(backbone, "blocks") and isinstance(backbone.blocks, (nn.ModuleList, nn.Sequential)):
            blocks = backbone.blocks
            hidden_dim = self._infer_hidden_dim(blocks[0])
            return blocks, hidden_dim
        if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
            blocks = backbone.layers
            hidden_dim = self._infer_hidden_dim(blocks[0])
            return blocks, hidden_dim
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            blocks = backbone.encoder.layer
            hidden_dim = self._infer_hidden_dim(blocks[0])
            return blocks, hidden_dim
        return None, None

    @staticmethod
    def _infer_hidden_dim(block: nn.Module) -> int:
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
            raise RuntimeError("VQTMethod: cannot set blocks on backbone.")

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._query_params)

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self._query_params)
        return {"total": trainable, "trainable": trainable}
