"""SpectralAttn — Spectral Attention PEFT.

Implements a spectral graph filter on the attention graph using Chebyshev
polynomial approximation:

    Y = g_theta(L_A) V

where L_A = I - D^{-1/2} A D^{-1/2} is the symmetric normalized attention
Laplacian, and g_theta is approximated via Chebyshev polynomials:

    g_theta(L_A) V = sum_{k=0}^{K} theta_k * T_k(L_tilde_A) * V

with the scaled Laplacian L_tilde_A = 2*L_A/lambda_max - I mapped to [-1, 1]
(lambda_max ≈ 2 for normalized Laplacian), and Chebyshev recurrence:

    T_0(x) = 1  (identity)
    T_1(x) = x
    T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)

Chebyshev coefficients theta are per-layer, per-head learnable parameters.
Residual initialisation: theta_1 = 1, all others = 0, so the initial output
equals the standard attention output A*V.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# Per-block attention wrapper
# ---------------------------------------------------------------------------

class SpectralAttnWrapper(nn.Module):
    """Wraps a single transformer attention module to apply spectral filtering.

    After computing the standard softmax attention weights A, constructs the
    symmetric normalized Laplacian L_A = I - D^{-1/2} A D^{-1/2}, scales it
    to L_tilde in [-1, 1], then applies the Chebyshev filter:

        Y_h = sum_{k=0}^{K} theta_{h,k} * T_k(L_tilde) * V_h

    Args:
        attn: the original timm-style attention module.
        cheby_order: Chebyshev polynomial order K (number of terms = K+1).
        num_heads: number of attention heads.
        lambda_max: approximation of the largest eigenvalue of L_A (default 2.0).
    """

    def __init__(
        self,
        attn: nn.Module,
        cheby_order: int,
        num_heads: int,
        lambda_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.cheby_order = cheby_order   # K
        self.num_heads = num_heads
        self.lambda_max = lambda_max

        # theta: (num_heads, K+1) — per-head Chebyshev coefficients
        # Residual init: theta[:, 1] = 1, rest = 0
        theta = torch.zeros(num_heads, cheby_order + 1)
        theta[:, 1] = 1.0
        self.theta = nn.Parameter(theta)

    def _compute_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """Compute symmetric normalized Laplacian L_A = I - D^{-1/2} A D^{-1/2}.

        Args:
            A: attention weights, shape (B, H, N, N), rows sum to 1 (non-negative).

        Returns:
            L_A: shape (B, H, N, N)
        """
        # Degree vector: d_i = sum_j A_{ij}  (= 1 for row-stochastic A)
        d = A.sum(dim=-1)                                   # (B, H, N)
        # D^{-1/2}: avoid division by zero
        d_inv_sqrt = d.pow(-0.5).clamp(max=1e4)            # (B, H, N)
        # D^{-1/2} A D^{-1/2}: broadcast via einsum-style multiply
        # A_norm[b,h,i,j] = d_inv_sqrt[b,h,i] * A[b,h,i,j] * d_inv_sqrt[b,h,j]
        A_norm = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)  # (B, H, N, N)
        # L_A = I - A_norm
        N = A.shape[-1]
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        return I - A_norm                                   # (B, H, N, N)

    def _scale_laplacian(self, L: torch.Tensor) -> torch.Tensor:
        """Scale Laplacian to [-1, 1]: L_tilde = 2*L/lambda_max - I."""
        N = L.shape[-1]
        I = torch.eye(N, device=L.device, dtype=L.dtype).unsqueeze(0).unsqueeze(0)
        return (2.0 / self.lambda_max) * L - I             # (B, H, N, N)

    def _chebyshev_filter(
        self, L_tilde: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Apply Chebyshev filter: sum_{k=0}^{K} theta_k * T_k(L_tilde) * V.

        Args:
            L_tilde: scaled Laplacian, shape (B, H, N, N).
            V: value matrix, shape (B, H, N, head_dim).

        Returns:
            Filtered output, shape (B, H, N, head_dim).
        """
        # theta: (num_heads, K+1) -> broadcast as (1, H, 1, 1) per coefficient
        # T_0(L_tilde) * V = I * V = V
        T_prev = V                                          # T_{k-1}: (B, H, N, head_dim)
        T_curr = torch.matmul(L_tilde, V)                  # T_1 * V = L_tilde * V

        # theta[:, k] shape: (H,) -> (1, H, 1, 1)
        def _theta_k(k: int) -> torch.Tensor:
            return self.theta[:, k].view(1, self.num_heads, 1, 1)

        out = _theta_k(0) * T_prev + _theta_k(1) * T_curr

        for k in range(2, self.cheby_order + 1):
            # T_k(L_tilde) * V = 2 * L_tilde * T_{k-1} * V - T_{k-2} * V
            T_next = 2.0 * torch.matmul(L_tilde, T_curr) - T_prev
            out = out + _theta_k(k) * T_next
            T_prev = T_curr
            T_curr = T_next

        return out

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Intercept attention forward; apply spectral Chebyshev filter."""
        B, N, C = x.shape
        head_dim = C // self.num_heads
        attn_mod = self.attn

        if not (hasattr(attn_mod, "qkv") and isinstance(attn_mod.qkv, nn.Linear)):
            # Fallback: run original attention unchanged
            return attn_mod(x, **kwargs)

        # ------------------------------------------------------------------
        # 1. Compute Q, K, V and softmax attention weights A
        # ------------------------------------------------------------------
        qkv = attn_mod.qkv(x)                                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                        # each (B, H, N, head_dim)

        scale = attn_mod.scale if hasattr(attn_mod, "scale") else head_dim ** -0.5
        attn_logits = (q @ k.transpose(-2, -1)) * scale                # (B, H, N, N)
        A = F.softmax(attn_logits, dim=-1)                             # (B, H, N, N)
        if hasattr(attn_mod, "attn_drop"):
            A = attn_mod.attn_drop(A)

        # ------------------------------------------------------------------
        # 2. Build scaled Laplacian L_tilde from A
        # ------------------------------------------------------------------
        L_A = self._compute_laplacian(A)                               # (B, H, N, N)
        L_tilde = self._scale_laplacian(L_A)                           # (B, H, N, N)

        # ------------------------------------------------------------------
        # 3. Apply Chebyshev spectral filter
        # ------------------------------------------------------------------
        out = self._chebyshev_filter(L_tilde, v)                       # (B, H, N, head_dim)

        # ------------------------------------------------------------------
        # 4. Reshape and project
        # ------------------------------------------------------------------
        out = out.transpose(1, 2).reshape(B, N, C)                     # (B, N, C)
        out = attn_mod.proj(out)
        if hasattr(attn_mod, "proj_drop"):
            out = attn_mod.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# MethodModule implementation
# ---------------------------------------------------------------------------

@register_method("spectral_attn")
class SpectralAttnMethod(MethodModule):
    """Spectral Attention PEFT via Chebyshev polynomial approximation.

    Implements the spectral graph filter Y = g_theta(L_A) V where L_A is the
    symmetric normalized attention Laplacian and g_theta is approximated by
    Chebyshev polynomials of order K.

    Config keys:
        cheby_order (int): Chebyshev polynomial order K. Default 3.
        lambda_max (float): approximation of max eigenvalue of L_A. Default 2.0.
    """

    def __init__(self, config: dict) -> None:
        self.cheby_order: int = int(config.get("cheby_order", 3))
        self.lambda_max: float = float(config.get("lambda_max", 2.0))

        if self.cheby_order < 1:
            raise ValueError(
                f"SpectralAttn cheby_order must be >= 1, got {self.cheby_order}"
            )

        self._wrappers: List[SpectralAttnWrapper] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Replace each transformer block's attention with a SpectralAttnWrapper."""
        self._wrappers = []
        self._inject_recursive(backbone)
        if not self._wrappers:
            raise RuntimeError(
                "SpectralAttnMethod.inject: no attention modules found. "
                "Ensure the backbone has timm-style Attention blocks with a .qkv attribute."
            )
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self._is_attention(child):
                num_heads = self._infer_num_heads(child)
                wrapper = SpectralAttnWrapper(
                    attn=child,
                    cheby_order=self.cheby_order,
                    num_heads=num_heads,
                    lambda_max=self.lambda_max,
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
        """Return num_heads from a timm Attention module."""
        return getattr(attn, "num_heads", 1)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [w.theta for w in self._wrappers]

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
