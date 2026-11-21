"""LAST — Layer-wise Adaptive Sparse Tuning for ViT backbones.

LAST fine-tunes only the last N transformer blocks of the backbone, keeping
all earlier blocks frozen. This is a simple but effective baseline that
exploits the observation that later layers are more task-specific.

Optionally, LAST also unfreezes the patch embedding and positional embedding.

Reference: "LAST: Layer-wise Adaptive Sparse Tuning" — used as a partial
fine-tuning baseline in PETL benchmarks.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


@register_method("last")
class LASTMethod(MethodModule):
    """LAST PEFT method — fine-tunes only the last N transformer blocks.

    Config keys:
        num_last_blocks (int): number of last blocks to unfreeze. Default 2.
        unfreeze_norm (bool): also unfreeze the final LayerNorm. Default True.
        unfreeze_patch_embed (bool): unfreeze patch embedding. Default False.
    """

    def __init__(self, config: dict) -> None:
        self.num_last_blocks: int = int(config.get("num_last_blocks", 2))
        self.unfreeze_norm: bool = bool(config.get("unfreeze_norm", True))
        self.unfreeze_patch_embed: bool = bool(config.get("unfreeze_patch_embed", False))
        self._trainable_params: List[nn.Parameter] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._trainable_params = []

        # Freeze everything first
        for param in backbone.parameters():
            param.requires_grad_(False)

        # Find transformer blocks
        blocks = self._find_blocks(backbone)
        if blocks is not None and len(blocks) > 0:
            n = min(self.num_last_blocks, len(blocks))
            for block in list(blocks)[-n:]:
                for param in block.parameters():
                    param.requires_grad_(True)
                    self._trainable_params.append(param)

        # Optionally unfreeze final norm
        if self.unfreeze_norm:
            for attr in ("norm", "norm_f", "ln_f", "layernorm"):
                norm = getattr(backbone, attr, None)
                if norm is not None and isinstance(norm, nn.LayerNorm):
                    for param in norm.parameters():
                        param.requires_grad_(True)
                        self._trainable_params.append(param)
                    break

        # Optionally unfreeze patch embedding
        if self.unfreeze_patch_embed:
            for attr in ("patch_embed", "embeddings", "patch_embedding"):
                embed = getattr(backbone, attr, None)
                if embed is not None:
                    for param in embed.parameters():
                        param.requires_grad_(True)
                        self._trainable_params.append(param)
                    break

        return backbone

    @staticmethod
    def _find_blocks(backbone: nn.Module):
        for attr in ("blocks", "layers", "encoder"):
            obj = getattr(backbone, attr, None)
            if obj is None:
                continue
            if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                return obj
            # encoder.layer pattern
            layer = getattr(obj, "layer", None)
            if isinstance(layer, (nn.ModuleList, nn.Sequential)):
                return layer
        return None

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self._trainable_params)

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self._trainable_params)
        return {"total": trainable, "trainable": trainable}
