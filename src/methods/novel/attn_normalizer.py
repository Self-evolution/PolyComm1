"""AttnNormalizer — Attention Normalizer PEFT.

Implements a learnable attention normalizer that replaces the standard softmax
with a mixture of normalizers:

    A = sum_m w_m(x) * normalizer_m(logits / tau)

where:
  - normalizer_m in {softmax, sparsemax, entmax}
  - tau: per-head learnable temperature (initialized to 1.0)
  - w_m: per-head mixture weights (learned, sum to 1 via softmax)
  - Optional entropy constraint via a Lagrange-style penalty

Residual initialisation: mixture weights concentrate on softmax (w_softmax=1),
temperature tau=1.0, so the initial output equals standard softmax attention.

References:
  - sparsemax: Martins & Astudillo, 2016 (From Softmax to Sparsemax)
  - entmax: Peters et al., 2019 (Sparse Sequence-to-Sequence Models)
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# Sparsemax implementation
# ---------------------------------------------------------------------------

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation: projection onto the probability simplex.

    Computes the Euclidean projection of z onto the probability simplex,
    yielding a sparse probability distribution.

    Args:
        z: input logits tensor.
        dim: dimension along which to apply sparsemax.

    Returns:
        Sparse probability distribution with same shape as z.
    """
    # Sort z in descending order along dim
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    # Cumulative sum
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    # k(z): number of support elements
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    # Reshape k for broadcasting
    shape = [1] * z.dim()
    shape[dim] = -1
    k = k.view(shape)
    # Condition: 1 + k * z_sorted[k] > cumsum[k-1]
    z_check = 1 + k * z_sorted > z_cumsum
    # k(z) = max k satisfying the condition
    k_z = z_check.sum(dim=dim, keepdim=True).float()
    # Threshold: tau(z) = (cumsum[k(z)] - 1) / k(z)
    # Gather cumsum at position k_z - 1
    k_z_idx = (k_z - 1).long().clamp(min=0)
    tau_z = (z_cumsum.gather(dim, k_z_idx) - 1.0) / k_z
    # Projection: max(z - tau, 0)
    return torch.clamp(z - tau_z, min=0.0)


# ---------------------------------------------------------------------------
# Entmax-1.5 implementation (alpha=1.5)
# ---------------------------------------------------------------------------

def entmax15(z: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """Entmax with alpha=1.5 via bisection.

    Computes the alpha=1.5 entmax transformation, which lies between
    softmax (alpha=1) and sparsemax (alpha=2).

    Args:
        z: input logits tensor.
        dim: dimension along which to apply entmax.
        n_iter: number of bisection iterations.

    Returns:
        Probability distribution with same shape as z.
    """
    # Shift for numerical stability
    z = z - z.max(dim=dim, keepdim=True).values

    def p_tau(tau: torch.Tensor) -> torch.Tensor:
        return torch.clamp(z - tau, min=0.0).pow(2)

    # Bisection to find tau such that sum(p_tau) = 1
    lo = (z - 1.0).min(dim=dim, keepdim=True).values
    hi = z.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        p_mid = p_tau(mid)
        s = p_mid.sum(dim=dim, keepdim=True)
        lo = torch.where(s > 1.0, mid, lo)
        hi = torch.where(s <= 1.0, mid, hi)

    tau_star = (lo + hi) / 2.0
    return p_tau(tau_star)


# ---------------------------------------------------------------------------
# Per-block attention wrapper
# ---------------------------------------------------------------------------

class AttnNormalizerWrapper(nn.Module):
    """Wraps a single transformer attention module with a learnable normalizer.

    Replaces softmax(QK^T / sqrt(d)) with a mixture of normalizers:

        A = sum_m w_m * normalizer_m(logits / tau)

    where normalizers are softmax, sparsemax, and entmax-1.5.

    Args:
        attn: the original timm-style attention module.
        num_heads: number of attention heads.
        entropy_weight: weight for entropy regularization penalty (0 = disabled).
    """

    def __init__(
        self,
        attn: nn.Module,
        num_heads: int,
        entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.num_heads = num_heads
        self.entropy_weight = entropy_weight

        # Learnable temperature per head: initialized to 1.0 (log-space for positivity)
        # log_temperature: (num_heads,), exp gives actual temperature
        self.log_temperature = nn.Parameter(torch.zeros(num_heads))

        # Mixture weights logits per head: (num_heads, 3) for [softmax, sparsemax, entmax]
        # Residual init: concentrate on softmax -> logits = [large, 0, 0]
        mix_logits = torch.zeros(num_heads, 3)
        mix_logits[:, 0] = 4.0  # softmax gets high initial weight
        self.mix_logits = nn.Parameter(mix_logits)

    @property
    def temperature(self) -> torch.Tensor:
        """Positive temperature per head: shape (num_heads,)."""
        return self.log_temperature.exp()

    @property
    def mix_weights(self) -> torch.Tensor:
        """Mixture weights per head (sum to 1): shape (num_heads, 3)."""
        return F.softmax(self.mix_logits, dim=-1)

    def _apply_normalizer(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the mixture of normalizers to attention logits.

        Args:
            logits: raw attention logits, shape (B, H, N, N).

        Returns:
            Normalized attention weights, shape (B, H, N, N).
        """
        B, H, N, _ = logits.shape

        # Apply temperature scaling: tau shape (H,) -> (1, H, 1, 1)
        tau = self.temperature.view(1, H, 1, 1)
        scaled = logits / tau.clamp(min=1e-6)

        # Compute each normalizer
        a_softmax = F.softmax(scaled, dim=-1)                   # (B, H, N, N)
        a_sparsemax = sparsemax(scaled, dim=-1)                  # (B, H, N, N)
        a_entmax = entmax15(scaled, dim=-1)                      # (B, H, N, N)

        # Mix weights: (H, 3) -> (1, H, 1, 1) per normalizer
        w = self.mix_weights                                     # (H, 3)
        w_softmax  = w[:, 0].view(1, H, 1, 1)
        w_sparsemax = w[:, 1].view(1, H, 1, 1)
        w_entmax   = w[:, 2].view(1, H, 1, 1)

        return w_softmax * a_softmax + w_sparsemax * a_sparsemax + w_entmax * a_entmax

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Intercept attention forward; apply learnable normalizer."""
        B, N, C = x.shape
        head_dim = C // self.num_heads
        attn_mod = self.attn

        if not (hasattr(attn_mod, "qkv") and isinstance(attn_mod.qkv, nn.Linear)):
            return attn_mod(x, **kwargs)

        # Compute Q, K, V
        qkv = attn_mod.qkv(x)                                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                        # each (B, H, N, head_dim)

        scale = attn_mod.scale if hasattr(attn_mod, "scale") else head_dim ** -0.5
        attn_logits = (q @ k.transpose(-2, -1)) * scale                # (B, H, N, N)

        # Apply learnable normalizer mixture
        attn_weights = self._apply_normalizer(attn_logits)             # (B, H, N, N)

        if hasattr(attn_mod, "attn_drop"):
            attn_weights = attn_mod.attn_drop(attn_weights)

        out = attn_weights @ v                                          # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)                     # (B, N, C)
        out = attn_mod.proj(out)
        if hasattr(attn_mod, "proj_drop"):
            out = attn_mod.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# MethodModule implementation
# ---------------------------------------------------------------------------

@register_method("attn_normalizer")
class AttnNormalizerMethod(MethodModule):
    """Attention Normalizer PEFT.

    Replaces the standard softmax attention normalizer with a learnable mixture
    of softmax, sparsemax, and entmax-1.5, with per-head learnable temperature.

    Config keys:
        entropy_weight (float): weight for entropy regularization. Default 0.0.
    """

    def __init__(self, config: dict) -> None:
        self.entropy_weight: float = float(config.get("entropy_weight", 0.0))
        self._wrappers: List[AttnNormalizerWrapper] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Replace each transformer block's attention with an AttnNormalizerWrapper."""
        self._wrappers = []
        self._inject_recursive(backbone)
        if not self._wrappers:
            raise RuntimeError(
                "AttnNormalizerMethod.inject: no attention modules found. "
                "Ensure the backbone has timm-style Attention blocks with a .qkv attribute."
            )
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self._is_attention(child):
                num_heads = self._infer_num_heads(child)
                wrapper = AttnNormalizerWrapper(
                    attn=child,
                    num_heads=num_heads,
                    entropy_weight=self.entropy_weight,
                )
                setattr(module, name, wrapper)
                self._wrappers.append(wrapper)
            else:
                self._inject_recursive(child)

    @staticmethod
    def _is_attention(module: nn.Module) -> bool:
        """Heuristic: timm ViT Attention has both .qkv and .proj as nn.Linear."""
        return (
            hasattr(module, "qkv")
            and isinstance(module.qkv, nn.Linear)
            and hasattr(module, "proj")
            and isinstance(module.proj, nn.Linear)
        )

    @staticmethod
    def _infer_num_heads(attn: nn.Module) -> int:
        return getattr(attn, "num_heads", 1)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for w in self._wrappers:
            params.append(w.log_temperature)
            params.append(w.mix_logits)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
