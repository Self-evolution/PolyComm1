"""VPT — Visual Prompt Tuning for ViT-style backbones.

Prepends learnable prompt tokens to the input sequence of each transformer block.

Modes:
  shallow: prompts only at the first layer (input embedding level).
  deep:    prompts prepended at every transformer block.

Reference: Jia et al., "Visual Prompt Tuning", ECCV 2022.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class ShallowPromptWrapper(nn.Module):
    """Wraps a ViT-style block sequence; prepends prompt tokens only at layer 0."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        prompt_embeddings: nn.Parameter,
        prompt_dropout: nn.Dropout,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.prompt_embeddings = prompt_embeddings
        self.prompt_dropout = prompt_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        prompts = self.prompt_dropout(
            self.prompt_embeddings.expand(B, -1, -1)
        )
        # Insert prompts after CLS token
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return x

    # Allow timm's checkpoint_seq to iterate over this wrapper
    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)


class DeepPromptWrapper(nn.Module):
    """Wraps a ViT-style block sequence; prepends fresh prompt tokens at every layer."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        prompt_embeddings: nn.Parameter,       # (1, num_tokens, dim)  — layer 0
        deep_prompt_embeddings: nn.Parameter,  # (num_layers-1, num_tokens, dim)
        prompt_dropout: nn.Dropout,
        num_tokens: int,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.prompt_embeddings = prompt_embeddings
        self.deep_prompt_embeddings = deep_prompt_embeddings
        self.prompt_dropout = prompt_dropout
        self.num_tokens = num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Layer 0: prepend shallow prompts
        prompts = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
        x = self.blocks[0](x)

        for i, blk in enumerate(self.blocks[1:], start=1):
            # Replace previous prompt tokens with fresh ones for this layer
            deep_prompts = self.prompt_dropout(
                self.deep_prompt_embeddings[i - 1].unsqueeze(0).expand(B, -1, -1)
            )
            x = torch.cat([
                x[:, :1],                          # CLS
                deep_prompts,                       # new prompts
                x[:, 1 + self.num_tokens:],         # patch tokens (skip old prompts)
            ], dim=1)
            x = blk(x)

        return x

    # Allow timm's checkpoint_seq to iterate over this wrapper
    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)


@register_method("vpt")
class VPTMethod(MethodModule):
    """Visual Prompt Tuning (VPT) PEFT method.

    Config keys:
        num_tokens (int): number of prompt tokens per layer. Default 10.
        mode (str): "shallow" or "deep". Default "shallow".
        dropout (float): dropout on prompt embeddings. Default 0.0.
        patch_size (int): patch size used for Xavier init scale. Default 16.
    """

    def __init__(self, config: dict) -> None:
        self.num_tokens: int = int(config.get("num_tokens", 10))
        self.mode: str = config.get("mode", "shallow")
        self.dropout: float = float(config.get("dropout", 0.0))
        self.patch_size: int = int(config.get("patch_size", 16))

        if self.mode not in ("shallow", "deep"):
            raise ValueError(f"VPT mode must be 'shallow' or 'deep', got '{self.mode}'")

        self._prompt_embeddings: Optional[nn.Parameter] = None
        self._deep_prompt_embeddings: Optional[nn.Parameter] = None

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Locate the transformer block list and wrap it with prompt injection."""
        blocks, hidden_dim = self._find_blocks(backbone)
        if blocks is None:
            raise RuntimeError(
                "VPTMethod.inject: could not locate a transformer block list "
                "(looked for 'blocks', 'layers', 'encoder.layer'). "
                "Ensure the backbone exposes its block list as one of these attributes."
            )

        # Xavier-uniform init scale (matches original VPT paper)
        val = math.sqrt(6.0 / float(3 * self.patch_size * self.patch_size + hidden_dim))

        self._prompt_embeddings = nn.Parameter(
            torch.empty(1, self.num_tokens, hidden_dim).uniform_(-val, val)
        )
        prompt_dropout = nn.Dropout(p=self.dropout)

        if self.mode == "shallow":
            wrapper = ShallowPromptWrapper(blocks, self._prompt_embeddings, prompt_dropout)
        else:
            num_deep = len(blocks) - 1
            self._deep_prompt_embeddings = nn.Parameter(
                torch.empty(num_deep, self.num_tokens, hidden_dim).uniform_(-val, val)
            )
            wrapper = DeepPromptWrapper(
                blocks,
                self._prompt_embeddings,
                self._deep_prompt_embeddings,
                prompt_dropout,
                self.num_tokens,
            )

        # Replace the block container on the backbone
        self._set_blocks(backbone, wrapper)
        return backbone

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self._prompt_embeddings is not None:
            params.append(self._prompt_embeddings)
        if self._deep_prompt_embeddings is not None:
            params.append(self._deep_prompt_embeddings)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_blocks(self, backbone: nn.Module):
        """Return (block_list, hidden_dim) by probing common attribute names."""
        # timm ViT: backbone.blocks
        if hasattr(backbone, "blocks") and isinstance(backbone.blocks, (nn.ModuleList, nn.Sequential)):
            blocks = backbone.blocks
            hidden_dim = self._infer_hidden_dim(blocks[0])
            return blocks, hidden_dim
        # timm Swin / some ViTs: backbone.layers
        if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
            blocks = backbone.layers
            hidden_dim = self._infer_hidden_dim(blocks[0])
            return blocks, hidden_dim
        # HuggingFace-style: backbone.encoder.layer
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            blocks = backbone.encoder.layer
            hidden_dim = self._infer_hidden_dim(blocks[0])
            return blocks, hidden_dim
        return None, None

    @staticmethod
    def _infer_hidden_dim(block: nn.Module) -> int:
        """Infer hidden dimension from the first Linear found in a block."""
        for m in block.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        raise RuntimeError("Cannot infer hidden_dim from transformer block.")

    def _set_blocks(self, backbone: nn.Module, wrapper: nn.Module) -> None:
        """Replace the block container attribute with *wrapper*."""
        if hasattr(backbone, "blocks") and isinstance(backbone.blocks, (nn.ModuleList, nn.Sequential)):
            backbone.blocks = wrapper
        elif hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
            backbone.layers = wrapper
        elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            backbone.encoder.layer = wrapper
        else:
            raise RuntimeError("VPTMethod: cannot set block wrapper on backbone.")
