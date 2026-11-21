"""SCT — Selective Channel Tuning for ViT-style backbones.

Adds a lightweight ChannelAdapter after the attention output in each
transformer block. The adapter operates only on a selected subset of
channels (topN), applying a small linear transform:
    x[..., selected_channels] += Linear(x[..., selected_channels])

Channel selection is based on the L2 norm of pretrained weight rows.

Reference: "SCT: A Simple Baseline for Parameter-Efficient Fine-Tuning
via Salient Channel Tuning" — PETL benchmark baseline.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class ChannelAdapter(nn.Module):
    """Lightweight adapter that transforms only selected channels.

    forward: x[..., channel_index] += adapt_linear(x[..., channel_index])
    """

    def __init__(self, num_selection: int, channel_index: torch.Tensor) -> None:
        super().__init__()
        self.channel_index = channel_index[:num_selection]
        self.adapt_linear = nn.Linear(num_selection, num_selection)
        nn.init.zeros_(self.adapt_linear.weight)
        nn.init.zeros_(self.adapt_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[..., self.channel_index] = (
            x[..., self.channel_index]
            + self.adapt_linear(x[..., self.channel_index])
        )
        return x


class SCTBlock(nn.Module):
    """Wraps a ViT block to insert a ChannelAdapter after attention."""

    def __init__(self, block: nn.Module, adapter: ChannelAdapter) -> None:
        super().__init__()
        self.block = block
        self.sct_adapter = adapter

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.block(x, **kwargs)
        out = self.sct_adapter(out)
        return out


@register_method("sct")
class SCTMethod(MethodModule):
    """SCT PEFT method — Selective Channel Tuning.

    Inserts a ChannelAdapter (nn.Linear(topN, topN)) after attention in
    each transformer block. Only the adapter parameters are trainable.

    Config keys:
        topN (int): number of channels to select per block. Default 96.
    """

    def __init__(self, config: dict) -> None:
        self.topN: int = int(config.get("topN", 96))
        self._adapters: List[ChannelAdapter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._adapters = []

        # Freeze everything
        for param in backbone.parameters():
            param.requires_grad_(False)

        # Find transformer blocks
        blocks = self._find_blocks(backbone)
        if blocks is None:
            raise RuntimeError(
                "SCTMethod.inject: could not locate transformer blocks."
            )

        embed_dim = self._infer_embed_dim(blocks[0])

        # Select channels based on weight magnitude of the attention output proj
        channel_indices = self._select_channels(blocks, embed_dim)

        # Wrap each block with a ChannelAdapter
        new_blocks = nn.Sequential()
        for i, blk in enumerate(blocks):
            adapter = ChannelAdapter(self.topN, channel_indices[i])
            self._adapters.append(adapter)
            new_blocks.append(SCTBlock(blk, adapter))

        self._set_blocks(backbone, new_blocks)
        return backbone

    def _select_channels(
        self, blocks, embed_dim: int
    ) -> List[torch.Tensor]:
        """Select top-N channels per block by L2 norm of attention proj weight."""
        indices = []
        for blk in blocks:
            # Try to get attention projection weight
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "proj") and hasattr(attn.proj, "weight"):
                w = attn.proj.weight.data  # (embed_dim, embed_dim)
                norms = w.norm(dim=0)  # L2 norm per input channel
            else:
                # Fallback: use norm1 weight if available
                norm1 = getattr(blk, "norm1", None)
                if norm1 is not None and hasattr(norm1, "weight"):
                    norms = norm1.weight.data.abs()
                else:
                    norms = torch.randn(embed_dim).abs()

            topn = min(self.topN, len(norms))
            _, idx = torch.topk(norms, topn)
            indices.append(idx.sort().values)
        return indices

    @staticmethod
    def _find_blocks(backbone: nn.Module):
        for attr in ("blocks", "layers"):
            obj = getattr(backbone, attr, None)
            if isinstance(obj, (nn.ModuleList, nn.Sequential)) and len(obj) > 0:
                return obj
        return None

    @staticmethod
    def _infer_embed_dim(block: nn.Module) -> int:
        for m in block.modules():
            if isinstance(m, nn.LayerNorm):
                return m.normalized_shape[0]
        return 768

    @staticmethod
    def _set_blocks(backbone: nn.Module, new_blocks: nn.Module) -> None:
        for attr in ("blocks", "layers"):
            if hasattr(backbone, attr):
                setattr(backbone, attr, new_blocks)
                return

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = []
        for adapter in self._adapters:
            params.extend(adapter.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
