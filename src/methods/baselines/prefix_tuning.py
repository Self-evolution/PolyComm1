"""Prefix Tuning — prepends learnable key/value prefix tokens to every
transformer attention layer.

Unlike VPT which prepends tokens to the input sequence, Prefix Tuning
directly prepends learnable key and value vectors to the attention
computation at every layer, without modifying the input sequence length.

Reference: Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for
Generation", ACL 2021. Adapted for ViT-style vision backbones as used in
VideoActionRecognition benchmarks.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


class PrefixAttention(nn.Module):
    """Wraps a ViT attention module to prepend prefix key/value tokens.

    The prefix K and V tensors are prepended to the projected K and V
    before the attention softmax, so the attention can attend to both
    the original tokens and the prefix tokens.
    """

    def __init__(
        self,
        attn: nn.Module,
        prefix_key: nn.Parameter,    # (num_heads, prefix_len, head_dim)
        prefix_value: nn.Parameter,  # (num_heads, prefix_len, head_dim)
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.prefix_key = prefix_key
        self.prefix_value = prefix_value
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_dropout = nn.Dropout(p=dropout)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        attn = self.attn

        # Try to use the original attention's qkv projection
        if hasattr(attn, "qkv"):
            qkv = attn.qkv(x)  # (B, N, 3*C)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
            q, k, v = qkv.unbind(0)             # each (B, H, N, D)
        else:
            # Fallback: run original attention and return
            return attn(x, **kwargs)

        # Prepend prefix K and V
        pk = self.prefix_dropout(self.prefix_key).unsqueeze(0).expand(B, -1, -1, -1)
        pv = self.prefix_dropout(self.prefix_value).unsqueeze(0).expand(B, -1, -1, -1)
        # pk, pv: (B, H, prefix_len, D)
        k = torch.cat([pk, k], dim=2)   # (B, H, prefix_len+N, D)
        v = torch.cat([pv, v], dim=2)   # (B, H, prefix_len+N, D)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, prefix_len+N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if hasattr(attn, "attn_drop"):
            attn_weights = attn.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Apply output projection if present
        if hasattr(attn, "proj"):
            out = attn.proj(out)
            if hasattr(attn, "proj_drop"):
                out = attn.proj_drop(out)

        return out


class PrefixBlock(nn.Module):
    """Wraps a ViT transformer block, replacing its attention with PrefixAttention."""

    def __init__(self, block: nn.Module, prefix_attn: PrefixAttention) -> None:
        super().__init__()
        self.block = block
        self.prefix_attn = prefix_attn
        # Patch the block's attn attribute so the block's forward uses prefix_attn
        self.block.attn = prefix_attn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.block(x, **kwargs)


@register_method("prefix_tuning")
class PrefixTuningMethod(MethodModule):
    """Prefix Tuning PEFT method for ViT-style backbones.

    Prepends learnable key/value prefix tokens to the attention of every
    transformer block. The input sequence length is unchanged.

    Config keys:
        prefix_len (int): number of prefix tokens per layer. Default 10.
        dropout (float): dropout on prefix embeddings. Default 0.0.
        reparameterize (bool): use MLP reparameterization for prefix init
            (helps optimization). Default False.
        mid_dim (int): hidden dim for reparameterization MLP. Default 512.
    """

    def __init__(self, config: dict) -> None:
        self.prefix_len: int = int(config.get("prefix_len", 10))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.reparameterize: bool = bool(config.get("reparameterize", False))
        self.mid_dim: int = int(config.get("mid_dim", 512))

        self._prefix_keys: List[nn.Parameter] = []
        self._prefix_values: List[nn.Parameter] = []
        self._reparam_nets: List[nn.Module] = []

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._prefix_keys = []
        self._prefix_values = []
        self._reparam_nets = []

        blocks, hidden_dim, num_heads = self._find_blocks(backbone)
        if blocks is None:
            raise RuntimeError(
                "PrefixTuningMethod.inject: could not locate transformer blocks. "
                "Backbone must expose 'blocks', 'layers', or 'encoder.layer'."
            )

        head_dim = hidden_dim // num_heads
        val = math.sqrt(6.0 / (hidden_dim + self.prefix_len))

        new_blocks = nn.Sequential()
        for blk in blocks:
            if not hasattr(blk, "attn"):
                new_blocks.append(blk)
                continue

            if self.reparameterize:
                # MLP reparameterization: embed_idx -> prefix K and V
                reparam = nn.Sequential(
                    nn.Embedding(self.prefix_len, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, num_heads * head_dim * 2),
                )
                self._reparam_nets.append(reparam)
                idx = torch.arange(self.prefix_len)
                kv = reparam(idx)  # (prefix_len, num_heads*head_dim*2)
                kv = kv.view(self.prefix_len, num_heads, head_dim * 2)
                k_init = kv[..., :head_dim].permute(1, 0, 2)   # (H, P, D)
                v_init = kv[..., head_dim:].permute(1, 0, 2)   # (H, P, D)
                prefix_key = nn.Parameter(k_init.detach().clone())
                prefix_val = nn.Parameter(v_init.detach().clone())
            else:
                prefix_key = nn.Parameter(
                    torch.empty(num_heads, self.prefix_len, head_dim).uniform_(-val, val)
                )
                prefix_val = nn.Parameter(
                    torch.empty(num_heads, self.prefix_len, head_dim).uniform_(-val, val)
                )

            self._prefix_keys.append(prefix_key)
            self._prefix_values.append(prefix_val)

            prefix_attn = PrefixAttention(
                attn=blk.attn,
                prefix_key=prefix_key,
                prefix_value=prefix_val,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=self.dropout,
            )
            new_blocks.append(PrefixBlock(blk, prefix_attn))

        self._set_blocks(backbone, new_blocks)
        return backbone

    def _find_blocks(self, backbone: nn.Module):
        """Return (blocks, hidden_dim, num_heads)."""
        for attr in ("blocks", "layers"):
            container = getattr(backbone, attr, None)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                blk = container[0]
                hidden_dim = self._infer_hidden_dim(blk)
                num_heads = self._infer_num_heads(blk)
                return container, hidden_dim, num_heads
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            container = backbone.encoder.layer
            blk = container[0]
            hidden_dim = self._infer_hidden_dim(blk)
            num_heads = self._infer_num_heads(blk)
            return container, hidden_dim, num_heads
        return None, None, None

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

    @staticmethod
    def _infer_num_heads(block: nn.Module) -> int:
        attn = getattr(block, "attn", None)
        if attn is not None:
            for attr in ("num_heads", "heads"):
                v = getattr(attn, attr, None)
                if isinstance(v, int):
                    return v
        # Fallback: assume 12 heads (ViT-B default)
        return 12

    def _set_blocks(self, backbone: nn.Module, new_blocks: nn.Module) -> None:
        if hasattr(backbone, "blocks") and isinstance(backbone.blocks, (nn.ModuleList, nn.Module)):
            backbone.blocks = new_blocks
        elif hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
            backbone.layers = new_blocks
        elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            backbone.encoder.layer = new_blocks
        else:
            raise RuntimeError("PrefixTuningMethod: cannot set blocks on backbone.")

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = list(self._prefix_keys) + list(self._prefix_values)
        for net in self._reparam_nets:
            params.extend(net.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
