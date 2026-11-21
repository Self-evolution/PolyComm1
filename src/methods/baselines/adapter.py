"""Adapter — Bottleneck Adapter variants for ViT-style backbones.

Supports three variants:
  - Pfeiffer: single adapter after FFN (post-FFN placement)
  - Houlsby:  two adapters per block — after attention and after FFN
  - AdaptFormer: parallel adapter alongside FFN (adds to FFN output)

All variants use a down-project → nonlinearity → up-project bottleneck.
Initialized with near-zero weights so initial output ≈ original output.

References:
  Pfeiffer et al., "AdapterFusion", EACL 2021
  Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019
  Chen et al., "AdaptFormer: Adapting Vision Transformers for Scalable
    Visual Recognition", NeurIPS 2022
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.methods.base import MethodModule
from src.methods.registry import register_method


class AdapterLayer(nn.Module):
    """Bottleneck adapter: down → act → up, with residual."""

    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, in_features)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Near-zero init so adapter starts as identity
        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=init_scale)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.dropout(x))))


# ---------------------------------------------------------------------------
# Block wrappers — one per variant
# ---------------------------------------------------------------------------

class PfeifferBlock(nn.Module):
    """Wraps a transformer block to inject a Pfeiffer adapter after FFN."""

    def __init__(self, block: nn.Module, adapter: AdapterLayer) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.block(x, **kwargs)
        # block may return a tensor or a tuple; handle both
        if isinstance(out, tuple):
            hidden, *rest = out
            return (self.adapter(hidden), *rest)
        return self.adapter(out)


class HoulsbyBlock(nn.Module):
    """Wraps a transformer block to inject Houlsby adapters after attn and FFN.

    Requires the block to expose `attn` and `mlp` (or `ffn`) sub-modules
    and a two-stage forward (attn → adapter1 → mlp → adapter2).
    Falls back to a single post-block adapter when the block structure
    is not recognised.
    """

    def __init__(
        self,
        block: nn.Module,
        adapter_attn: AdapterLayer,
        adapter_ffn: AdapterLayer,
    ) -> None:
        super().__init__()
        self.block = block
        self.adapter_attn = adapter_attn
        self.adapter_ffn = adapter_ffn
        # Detect timm-style ViT block attributes
        self._has_split_forward = (
            hasattr(block, "attn") and hasattr(block, "mlp")
            and hasattr(block, "norm1") and hasattr(block, "norm2")
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._has_split_forward:
            # Replicate timm ViT block forward with adapter injections
            b = self.block
            # Attention sub-block
            x = x + self.adapter_attn(b.attn(b.norm1(x)))
            # FFN sub-block
            x = x + self.adapter_ffn(b.mlp(b.norm2(x)))
            return x
        # Fallback: apply both adapters after the full block
        out = self.block(x, **kwargs)
        if isinstance(out, tuple):
            hidden, *rest = out
            hidden = self.adapter_attn(hidden)
            hidden = self.adapter_ffn(hidden)
            return (hidden, *rest)
        out = self.adapter_attn(out)
        out = self.adapter_ffn(out)
        return out


class AdaptFormerBlock(nn.Module):
    """Wraps a transformer block to inject a parallel AdaptFormer adapter.

    The adapter runs in parallel with the FFN branch and its output is
    added (scaled by `scale`) to the FFN output before the residual.
    """

    def __init__(
        self,
        block: nn.Module,
        adapter: AdapterLayer,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter
        self.scale = scale
        self._has_split_forward = (
            hasattr(block, "attn") and hasattr(block, "mlp")
            and hasattr(block, "norm1") and hasattr(block, "norm2")
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._has_split_forward:
            b = self.block
            # Attention sub-block (unchanged)
            x = x + b.drop_path(b.attn(b.norm1(x))) if hasattr(b, "drop_path") else x + b.attn(b.norm1(x))
            # Parallel FFN + adapter
            normed = b.norm2(x)
            ffn_out = b.mlp(normed)
            adapt_out = self.adapter(normed) * self.scale
            x = x + ffn_out + adapt_out
            return x
        # Fallback: post-block parallel adapter (approximate)
        out = self.block(x, **kwargs)
        if isinstance(out, tuple):
            hidden, *rest = out
            return (hidden + self.adapter(x) * self.scale, *rest)
        return out + self.adapter(x) * self.scale


# ---------------------------------------------------------------------------
# MethodModule
# ---------------------------------------------------------------------------

@register_method("adapter")
class AdapterMethod(MethodModule):
    """Adapter PEFT method supporting Pfeiffer, Houlsby, and AdaptFormer variants.

    Config keys:
        variant (str): "pfeiffer" | "houlsby" | "adaptformer". Default "pfeiffer".
        bottleneck_dim (int): hidden size of the adapter bottleneck. Default 64.
        dropout (float): dropout inside the adapter. Default 0.0.
        init_scale (float): std for near-zero weight init. Default 1e-3.
        scale (float): output scale for AdaptFormer parallel branch. Default 1.0.
        block_name_pattern (str): substring to match transformer block names.
            Default "blocks".
    """

    def __init__(self, config: dict) -> None:
        self.variant: str = config.get("variant", "pfeiffer")
        self.bottleneck_dim: int = int(config.get("bottleneck_dim", 64))
        self.dropout: float = float(config.get("dropout", 0.0))
        self.init_scale: float = float(config.get("init_scale", 1e-3))
        self.scale: float = float(config.get("scale", 1.0))
        self.block_name_pattern: str = config.get("block_name_pattern", "blocks")

        self._adapters: List[AdapterLayer] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        self._adapters = []
        self._inject_recursive(backbone)
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self.block_name_pattern in name or self._is_transformer_block(child):
                hidden_dim = self._infer_hidden_dim(child)
                if hidden_dim is None:
                    self._inject_recursive(child)
                    continue

                if self.variant == "pfeiffer":
                    adapter = self._make_adapter(hidden_dim)
                    wrapped = PfeifferBlock(child, adapter)
                    setattr(module, name, wrapped)

                elif self.variant == "houlsby":
                    adapter_attn = self._make_adapter(hidden_dim)
                    adapter_ffn = self._make_adapter(hidden_dim)
                    wrapped = HoulsbyBlock(child, adapter_attn, adapter_ffn)
                    setattr(module, name, wrapped)

                elif self.variant == "adaptformer":
                    adapter = self._make_adapter(hidden_dim)
                    wrapped = AdaptFormerBlock(child, adapter, scale=self.scale)
                    setattr(module, name, wrapped)

                else:
                    raise ValueError(
                        f"Unknown adapter variant '{self.variant}'. "
                        "Choose from: pfeiffer, houlsby, adaptformer."
                    )
            else:
                self._inject_recursive(child)

    def _make_adapter(self, hidden_dim: int) -> AdapterLayer:
        adapter = AdapterLayer(
            in_features=hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
            init_scale=self.init_scale,
        )
        self._adapters.append(adapter)
        return adapter

    @staticmethod
    def _is_transformer_block(module: nn.Module) -> bool:
        """Heuristic: a module with both 'attn' and 'mlp' children is a ViT block."""
        children = {n for n, _ in module.named_children()}
        return "attn" in children and ("mlp" in children or "ffn" in children)

    @staticmethod
    def _infer_hidden_dim(module: nn.Module) -> Optional[int]:
        """Try to infer the hidden dimension from the block's norm or mlp layers."""
        # timm ViT: norm1 is LayerNorm(hidden_dim)
        for attr in ("norm1", "norm2", "norm"):
            ln = getattr(module, attr, None)
            if isinstance(ln, nn.LayerNorm):
                return ln.normalized_shape[0]
        # Fallback: look for a Linear layer in mlp/ffn
        for attr in ("mlp", "ffn"):
            sub = getattr(module, attr, None)
            if sub is not None:
                for m in sub.modules():
                    if isinstance(m, nn.Linear):
                        return m.in_features
        return None

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for adapter in self._adapters:
            params.extend(adapter.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
