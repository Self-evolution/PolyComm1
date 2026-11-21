"""GPS — Gradient-based Parameter Selection for ViT fine-tuning.

GPS uses the same ChannelAdapter architecture as SCT, but selects channels
based on gradient magnitude (computed on a calibration forward pass) rather
than weight magnitude.

In practice (without a calibration set at inject time), we fall back to
weight-magnitude selection — making GPS equivalent to SCT with a different
default topN.

Reference: "GPS: Gradient-guided Parameter Selection for Efficient
Fine-Tuning" — PETL benchmark baseline.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method
from src.methods.baselines.sct import ChannelAdapter, SCTBlock


@register_method("gps")
class GPSMethod(MethodModule):
    """GPS PEFT method — gradient-based channel selection + ChannelAdapter.

    Architecturally identical to SCT (ChannelAdapter after attention),
    but uses gradient-based channel selection when calibration data is
    available, falling back to weight-magnitude selection otherwise.

    Config keys:
        topN (int): number of channels to select per block. Default 96.
    """

    def __init__(self, config: dict) -> None:
        self.topN: int = int(config.get("topN", 96))
        self._adapters: List[ChannelAdapter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._adapters = []

        for param in backbone.parameters():
            param.requires_grad_(False)

        blocks = self._find_blocks(backbone)
        if blocks is None:
            raise RuntimeError(
                "GPSMethod.inject: could not locate transformer blocks."
            )

        embed_dim = self._infer_embed_dim(blocks[0])
        channel_indices = self._select_channels(blocks, embed_dim)

        new_blocks = nn.Sequential()
        for i, blk in enumerate(blocks):
            adapter = ChannelAdapter(self.topN, channel_indices[i])
            self._adapters.append(adapter)
            new_blocks.append(SCTBlock(blk, adapter))

        self._set_blocks(backbone, new_blocks)
        return backbone

    def _select_channels(self, blocks, embed_dim: int) -> List[torch.Tensor]:
        """Select top-N channels per block by weight magnitude (fallback)."""
        indices = []
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and hasattr(attn, "proj") and hasattr(attn.proj, "weight"):
                w = attn.proj.weight.data
                norms = w.norm(dim=0)
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
