"""PolyAttn — Polynomial / Multi-hop Attention PEFT.

Replaces the standard attention output with a polynomial combination of
multi-hop attention maps applied to the value matrix:

    Y_h = sum_{k=0}^{K} gamma_{l,h,k}(x) * A_h^k * V_h

where A_h^k is computed via recurrence (A_h^0 = I, A_h^{k+1} = A_h * A_h^k)
to avoid explicit matrix powers.

Two coefficient modes:
  static:  gamma is a directly trainable parameter tensor, shape (K+1,) per head/layer.
  dynamic: a small CoeffGenerator MLP produces gamma conditioned on the input.

Residual initialisation: gamma_1 = 1, all others = 0, so the initial output
equals the standard attention output A*V.

A norm constraint is applied to higher-order coefficients (k >= 2) to keep
the polynomial well-conditioned.

Optional head-wise coefficients let different heads learn different propagation
depths, and residual-safe dropout can regularise non-residual hops without
randomly disabling the identity initialisation path.

For small-data regimes, coefficients can also be learned as a bounded delta
around the residual initialisation instead of as unconstrained direct values.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _optional_float(config: dict, key: str) -> Optional[float]:
    if key not in config or config.get(key) is None:
        return None
    return float(config[key])


def _optional_int(config: dict, key: str) -> Optional[int]:
    if key not in config or config.get(key) is None:
        return None
    return int(config[key])


def _prob_to_logit(prob: float) -> float:
    eps = 1e-6
    prob = min(max(float(prob), eps), 1.0 - eps)
    return math.log(prob / (1.0 - prob))


# ---------------------------------------------------------------------------
# Coefficient generator (dynamic mode)
# ---------------------------------------------------------------------------

class CoeffGenerator(nn.Module):
    """Input-conditioned coefficient generator.

    Pools Q, K, V representations and maps them to K+1 gamma coefficients
    via a small 2-layer MLP.

    Args:
        hidden_dim: dimension of Q/K/V projections.
        num_hops: K (polynomial degree); output size is K+1.
        mlp_ratio: hidden size of MLP = hidden_dim * mlp_ratio.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_hops: int,
        num_heads: int = 1,
        headwise: bool = False,
        mlp_ratio: float = 0.25,
        parameterization: str = "direct",
        delta_scale: float = 1.0,
        coeff_activation: str = "none",
    ) -> None:
        super().__init__()
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.headwise = headwise
        self.parameterization = parameterization
        self.delta_scale = delta_scale
        self.coeff_activation = coeff_activation

        mid = max(4, int(hidden_dim * mlp_ratio))
        out_dim = (num_hops + 1) * num_heads if headwise else num_hops + 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Linear(mid, out_dim),
        )
        base = torch.zeros(out_dim)
        if headwise:
            base = base.view(num_heads, num_hops + 1)
            base[:, 1] = 1.0
            base = base.reshape(out_dim)
        else:
            base[1] = 1.0
        self.register_buffer("base_gamma", base)

        # Init: direct mode outputs residual gamma; delta mode outputs zero
        # delta around the residual gamma.
        nn.init.zeros_(self.mlp[-1].weight)
        bias = base.clone() if parameterization == "direct" else torch.zeros(out_dim)
        self.mlp[-1].bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, hidden_dim) -> gamma shaped (B, H|1, K+1)."""
        pooled = x.mean(dim=1)          # (B, hidden_dim)
        gamma = self.mlp(pooled)
        if self.parameterization == "delta":
            gamma = self.base_gamma.unsqueeze(0) + self.delta_scale * self._activate(gamma)
        if self.headwise:
            return gamma.view(x.shape[0], self.num_heads, self.num_hops + 1)
        return gamma.unsqueeze(1)

    def _activate(self, gamma: torch.Tensor) -> torch.Tensor:
        if self.coeff_activation == "tanh":
            return torch.tanh(gamma)
        return gamma


# ---------------------------------------------------------------------------
# Value residual adapter
# ---------------------------------------------------------------------------

class ValueResidualAdapter(nn.Module):
    """Low-rank residual adapter for the value stream.

    The adapter is zero-initialized on the up projection, so enabling it keeps
    the initial wrapped attention identical to the original residual-init path.
    During training it learns a compact ΔV(x) that is propagated by the
    polynomial attention recurrence.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.down = nn.Parameter(torch.empty(rank, hidden_dim))
        self.up = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.down.t() @ self.up.t()) * self.scaling


class OutputResidualAdapter(ValueResidualAdapter):
    """Low-rank output adapter with deterministic down initialization.

    Adding an output adapter should not perturb the random stream used by the
    task head initialization. The grouped averaging basis is simple but
    trainable; once the zero-initialized up projection moves, the down basis
    also receives gradients.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)
        self.rank = rank
        self.scaling = alpha / rank
        self.down = nn.Parameter(torch.zeros(rank, hidden_dim))
        self.up = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        with torch.no_grad():
            for row in range(rank):
                indices = torch.arange(row, hidden_dim, rank)
                if indices.numel() > 0:
                    self.down[row, indices] = indices.numel() ** -0.5


class PackedQKVResidualAdapter(nn.Module):
    """LoRA-style low-rank residual adapter for a combined QKV projection."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.down = nn.Parameter(torch.empty(rank, hidden_dim))
        self.up = nn.Parameter(torch.zeros(3 * hidden_dim, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.down.t() @ self.up.t()) * self.scaling


class PackedQKVDoRAAdapter(nn.Module):
    """DoRA-style low-rank direction adapter for a combined QKV projection."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        train_bias: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("base_weight", linear.weight.detach().clone())
        base_bias = None if linear.bias is None else linear.bias.detach().clone()
        if train_bias and base_bias is not None:
            self.bias = nn.Parameter(base_bias.clone())
            self.register_buffer("base_bias", None)
        else:
            self.bias = None
            self.register_buffer("base_bias", base_bias)
        self.rank = rank
        self.scaling = alpha / rank
        self.down = nn.Parameter(torch.empty(rank, linear.in_features))
        self.up = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        with torch.no_grad():
            row_norms = self.base_weight.norm(p=2, dim=1)
        self.magnitude = nn.Parameter(row_norms.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up @ self.down
        weight = self.base_weight + self.scaling * delta
        direction = weight / (weight.norm(p=2, dim=1, keepdim=True) + 1e-8)
        adapted = self.magnitude.unsqueeze(1) * direction
        bias = self.bias if self.bias is not None else self.base_bias
        return F.linear(self.dropout(x), adapted, bias)


class QKVCommunicationGate(nn.Module):
    """Sigmoid router for the calibrated DoRA-QKV communication branch."""

    def __init__(
        self,
        hidden_dim: int,
        mode: str,
        init: float,
        mlp_ratio: float = 0.03125,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"sigmoid", "dynamic"}:
            raise ValueError(
                "QKVCommunicationGate mode must be 'sigmoid' or "
                f"'dynamic', got '{mode}'"
            )
        self.mode = mode
        self.temperature = max(float(temperature), 1e-6)
        init_logit = _prob_to_logit(init)

        if mode == "sigmoid":
            self.logit: Optional[nn.Parameter] = nn.Parameter(torch.tensor(init_logit))
            self.mlp: Optional[nn.Module] = None
        else:
            self.logit = None
            mid = max(4, int(hidden_dim * mlp_ratio))
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mid),
                nn.GELU(),
                nn.Linear(mid, 1),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            self.mlp[-1].bias = nn.Parameter(torch.tensor([init_logit]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.logit is not None:
            gate = torch.sigmoid(self.logit / self.temperature).view(1, 1, 1)
            return gate.to(dtype=x.dtype)

        assert self.mlp is not None
        pooled = x.mean(dim=1)
        gate = torch.sigmoid(self.mlp(pooled) / self.temperature)
        return gate.to(dtype=x.dtype).view(x.shape[0], 1, 1)


# ---------------------------------------------------------------------------
# Per-block attention wrapper
# ---------------------------------------------------------------------------

class PolyAttnWrapper(nn.Module):
    """Wraps a single transformer attention module to apply polynomial attention.

    Hooks into the attention *output* by replacing the block's forward so that
    after the standard softmax attention weights A are computed, the output is:

        Y = sum_{k=0}^{K} gamma_k * A^k * V

    This wrapper stores a reference to the original attention module and
    intercepts its forward call.
    """

    def __init__(
        self,
        attn: nn.Module,
        num_hops: int,
        mode: str,
        coeff_dropout: float,
        norm_constraint: float,
        hidden_dim: int,
        num_heads: int,
        headwise: bool = False,
        dropout_target: str = "all",
        mlp_ratio: float = 0.25,
        parameterization: str = "direct",
        delta_scale: float = 1.0,
        coeff_activation: str = "none",
        value_adapter_rank: int = 0,
        value_adapter_alpha: float = 1.0,
        value_adapter_dropout: float = 0.0,
        value_adapter_gate_init: Optional[float] = None,
        support_adapter_rank: int = 0,
        support_adapter_alpha: float = 1.0,
        support_adapter_dropout: float = 0.0,
        support_adapter_gate_init: Optional[float] = None,
        support_adapter_tied: bool = False,
        output_adapter_rank: int = 0,
        output_adapter_alpha: float = 1.0,
        output_adapter_dropout: float = 0.0,
        output_adapter_gate_init: Optional[float] = None,
        qkv_adapter_rank: int = 0,
        qkv_adapter_alpha: float = 1.0,
        qkv_adapter_dropout: float = 0.0,
        qkv_adapter_gate_init: Optional[float] = None,
        dora_qkv_rank: int = 0,
        dora_qkv_alpha: float = 1.0,
        dora_qkv_dropout: float = 0.0,
        dora_qkv_ssf: bool = False,
        dora_qkv_train_bias: bool = False,
        dora_qkv_gate_init: Optional[float] = None,
        dora_qkv_gate_mode: str = "linear",
        dora_qkv_gate_scope: str = "dora",
        dora_qkv_gate_mlp_ratio: float = 0.03125,
        dora_qkv_gate_temperature: float = 1.0,
        coeff_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.num_hops = num_hops          # K
        self.mode = mode
        self.num_heads = num_heads
        self.headwise = headwise
        self.norm_constraint = norm_constraint
        self.coeff_dropout = coeff_dropout
        self.dropout_target = dropout_target
        self.parameterization = parameterization
        self.delta_scale = delta_scale
        self.coeff_activation = coeff_activation
        self.coeff_trainable = coeff_trainable
        self.value_adapter_rank = value_adapter_rank
        self.support_adapter_rank = support_adapter_rank
        self.support_adapter_tied = support_adapter_tied
        self.output_adapter_rank = output_adapter_rank
        self.qkv_adapter_rank = qkv_adapter_rank
        self.dora_qkv_rank = dora_qkv_rank
        self.dora_qkv_ssf = dora_qkv_ssf
        self.dora_qkv_train_bias = dora_qkv_train_bias
        self.dora_qkv_gate_init = dora_qkv_gate_init
        self.dora_qkv_gate_mode = dora_qkv_gate_mode
        self.dora_qkv_gate_scope = dora_qkv_gate_scope
        self.dora_qkv_gate_mlp_ratio = dora_qkv_gate_mlp_ratio
        self.dora_qkv_gate_temperature = dora_qkv_gate_temperature

        if dora_qkv_gate_mode not in {"linear", "sigmoid", "dynamic"}:
            raise ValueError(
                "dora_qkv_gate_mode must be 'linear', 'sigmoid', or "
                f"'dynamic', got '{dora_qkv_gate_mode}'"
            )
        if dora_qkv_gate_scope not in {"dora", "calibrated"}:
            raise ValueError(
                "dora_qkv_gate_scope must be 'dora' or "
                f"'calibrated', got '{dora_qkv_gate_scope}'"
            )

        frozen_base = torch.zeros(num_heads, num_hops + 1) if headwise else torch.zeros(num_hops + 1)
        if headwise:
            frozen_base[:, 1] = 1.0
        else:
            frozen_base[1] = 1.0
        self.register_buffer("frozen_base_gamma", frozen_base)

        if mode == "static":
            if headwise:
                base = torch.zeros(num_heads, num_hops + 1)
                base[:, 1] = 1.0        # residual init per head
            else:
                # Shape: (K+1,) — shared across heads for parameter efficiency
                base = torch.zeros(num_hops + 1)
                base[1] = 1.0           # residual init
            self.register_buffer("base_gamma", base)
            init = base.clone() if parameterization == "direct" else torch.zeros_like(base)
            self.gamma = nn.Parameter(init)
            self.coeff_gen: Optional[CoeffGenerator] = None
        else:
            self.gamma = None            # type: ignore[assignment]
            self.coeff_gen = CoeffGenerator(
                hidden_dim,
                num_hops,
                num_heads,
                headwise,
                mlp_ratio=mlp_ratio,
                parameterization=parameterization,
                delta_scale=delta_scale,
                coeff_activation=coeff_activation,
            )

        self.value_adapter: Optional[ValueResidualAdapter]
        if value_adapter_rank > 0:
            self.value_adapter = ValueResidualAdapter(
                hidden_dim=hidden_dim,
                rank=value_adapter_rank,
                alpha=value_adapter_alpha,
                dropout=value_adapter_dropout,
            )
        else:
            self.value_adapter = None

        self.value_adapter_gate: Optional[nn.Parameter]
        if value_adapter_gate_init is not None and self.value_adapter is not None:
            self.value_adapter_gate = nn.Parameter(torch.tensor(float(value_adapter_gate_init)))
        else:
            self.value_adapter_gate = None

        self.q_adapter: Optional[ValueResidualAdapter]
        self.k_adapter: Optional[ValueResidualAdapter]
        if support_adapter_rank > 0:
            self.q_adapter = ValueResidualAdapter(
                hidden_dim=hidden_dim,
                rank=support_adapter_rank,
                alpha=support_adapter_alpha,
                dropout=support_adapter_dropout,
            )
            if support_adapter_tied:
                self.k_adapter = self.q_adapter
            else:
                self.k_adapter = ValueResidualAdapter(
                    hidden_dim=hidden_dim,
                    rank=support_adapter_rank,
                    alpha=support_adapter_alpha,
                    dropout=support_adapter_dropout,
                )
        else:
            self.q_adapter = None
            self.k_adapter = None

        self.support_adapter_gate: Optional[nn.Parameter]
        if support_adapter_gate_init is not None and self.q_adapter is not None:
            self.support_adapter_gate = nn.Parameter(torch.tensor(float(support_adapter_gate_init)))
        else:
            self.support_adapter_gate = None

        self.output_adapter: Optional[ValueResidualAdapter]
        if output_adapter_rank > 0:
            self.output_adapter = OutputResidualAdapter(
                hidden_dim=hidden_dim,
                rank=output_adapter_rank,
                alpha=output_adapter_alpha,
                dropout=output_adapter_dropout,
            )
        else:
            self.output_adapter = None

        self.output_adapter_gate: Optional[nn.Parameter]
        if output_adapter_gate_init is not None and self.output_adapter is not None:
            self.output_adapter_gate = nn.Parameter(torch.tensor(float(output_adapter_gate_init)))
        else:
            self.output_adapter_gate = None

        self.qkv_adapter: Optional[PackedQKVResidualAdapter]
        if qkv_adapter_rank > 0:
            self.qkv_adapter = PackedQKVResidualAdapter(
                hidden_dim=hidden_dim,
                rank=qkv_adapter_rank,
                alpha=qkv_adapter_alpha,
                dropout=qkv_adapter_dropout,
            )
        else:
            self.qkv_adapter = None

        self.qkv_adapter_gate: Optional[nn.Parameter]
        if qkv_adapter_gate_init is not None and self.qkv_adapter is not None:
            self.qkv_adapter_gate = nn.Parameter(torch.tensor(float(qkv_adapter_gate_init)))
        else:
            self.qkv_adapter_gate = None

        self.dora_qkv_adapter: Optional[PackedQKVDoRAAdapter]
        if dora_qkv_rank > 0:
            if qkv_adapter_rank > 0:
                raise ValueError("Use either qkv_adapter_rank or dora_qkv_rank, not both")
            self.dora_qkv_adapter = PackedQKVDoRAAdapter(
                linear=attn.qkv,
                rank=dora_qkv_rank,
                alpha=dora_qkv_alpha,
                dropout=dora_qkv_dropout,
                train_bias=dora_qkv_train_bias,
            )
        else:
            self.dora_qkv_adapter = None

        self.dora_qkv_gate: Optional[nn.Parameter]
        self.dora_qkv_gate_router: Optional[QKVCommunicationGate]
        if dora_qkv_gate_init is not None and self.dora_qkv_adapter is not None:
            if dora_qkv_gate_mode == "linear":
                self.dora_qkv_gate = nn.Parameter(torch.tensor(float(dora_qkv_gate_init)))
                self.dora_qkv_gate_router = None
            else:
                self.dora_qkv_gate = None
                self.dora_qkv_gate_router = QKVCommunicationGate(
                    hidden_dim=hidden_dim,
                    mode=dora_qkv_gate_mode,
                    init=float(dora_qkv_gate_init),
                    mlp_ratio=dora_qkv_gate_mlp_ratio,
                    temperature=dora_qkv_gate_temperature,
                )
        else:
            self.dora_qkv_gate = None
            self.dora_qkv_gate_router = None

        if self.dora_qkv_adapter is not None and self.dora_qkv_ssf:
            self.dora_qkv_scale = nn.Parameter(torch.ones(3 * hidden_dim))
            self.dora_qkv_shift = nn.Parameter(torch.zeros(3 * hidden_dim))
        else:
            self.dora_qkv_scale = None
            self.dora_qkv_shift = None

    def _get_gamma(self, x: torch.Tensor) -> torch.Tensor:
        """Return gamma coefficients, shape (B|1, H|1, K+1)."""
        if not self.coeff_trainable:
            gamma = self.frozen_base_gamma.to(dtype=x.dtype)
            if self.headwise:
                return gamma.unsqueeze(0)
            return gamma.view(1, 1, -1)

        if self.mode == "static":
            gamma = self._static_gamma()
            if self.headwise:
                gamma = gamma.unsqueeze(0)           # (1, H, K+1)
            else:
                gamma = gamma.view(1, 1, -1)         # (1, 1, K+1)
        else:
            gamma = self.coeff_gen(x)                # (B, H|1, K+1)

        gamma = self._apply_coeff_dropout(gamma)

        # Norm constraint on higher-order coefficients (k >= 2)
        if self.norm_constraint > 0.0 and self.num_hops >= 2:
            high = gamma[..., 2:]
            norm = high.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = (norm / self.norm_constraint).clamp(min=1.0)
            gamma = torch.cat([gamma[..., :2], high / scale], dim=-1)

        return gamma

    def _get_dora_qkv_gate(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.dora_qkv_gate_router is not None:
            return self.dora_qkv_gate_router(x)
        if self.dora_qkv_gate is not None:
            return self.dora_qkv_gate
        return None

    def _static_gamma(self) -> torch.Tensor:
        if self.parameterization == "direct":
            return self.gamma
        return self.base_gamma + self.delta_scale * self._activate(self.gamma)

    def _activate(self, gamma: torch.Tensor) -> torch.Tensor:
        if self.coeff_activation == "tanh":
            return torch.tanh(gamma)
        return gamma

    def _apply_coeff_dropout(self, gamma: torch.Tensor) -> torch.Tensor:
        if self.coeff_dropout <= 0.0 or not self.training:
            return gamma
        if self.dropout_target == "all":
            return F.dropout(gamma, p=self.coeff_dropout, training=True)

        if self.dropout_target == "high_order":
            indices = range(2, self.num_hops + 1)
        else:
            indices = [idx for idx in range(self.num_hops + 1) if idx != 1]

        indices = list(indices)
        if not indices:
            return gamma

        dropped = gamma.clone()
        selected = dropped[..., indices]
        dropped[..., indices] = F.dropout(selected, p=self.coeff_dropout, training=True)
        return dropped

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Intercept attention forward; compute polynomial output."""
        # ------------------------------------------------------------------
        # 1. Obtain Q, K, V and attention weights from the wrapped module.
        #    We support timm-style ViT attention (Attention class with .qkv).
        # ------------------------------------------------------------------
        B, N, C = x.shape
        head_dim = getattr(self.attn, "head_dim", C // self.num_heads)
        attn_dim = getattr(self.attn, "attn_dim", head_dim * self.num_heads)

        attn_mod = self.attn

        # timm ViT Attention: has .qkv (nn.Linear), .proj, .attn_drop, .scale
        if hasattr(attn_mod, "qkv"):
            if self.dora_qkv_adapter is not None:
                dora_qkv = self.dora_qkv_adapter(x)
                if self.dora_qkv_scale is not None and self.dora_qkv_shift is not None:
                    calibrated_dora_qkv = self.dora_qkv_scale * dora_qkv + self.dora_qkv_shift
                else:
                    calibrated_dora_qkv = dora_qkv

                dora_gate = self._get_dora_qkv_gate(x)
                if dora_gate is not None:
                    base_qkv = attn_mod.qkv(x)
                    if self.dora_qkv_gate_scope == "calibrated":
                        qkv = base_qkv + dora_gate * (calibrated_dora_qkv - base_qkv)
                    else:
                        qkv = base_qkv + dora_gate * (dora_qkv - base_qkv)
                        if self.dora_qkv_scale is not None and self.dora_qkv_shift is not None:
                            qkv = self.dora_qkv_scale * qkv + self.dora_qkv_shift
                else:
                    qkv = calibrated_dora_qkv
            else:
                qkv = attn_mod.qkv(x)                      # (B, N, 3*C)
            if self.qkv_adapter is not None:
                delta_qkv = self.qkv_adapter(x)
                if self.qkv_adapter_gate is not None:
                    delta_qkv = delta_qkv * self.qkv_adapter_gate
                qkv = qkv + delta_qkv
            qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)                        # each (B, H, N, head_dim)
            if self.q_adapter is not None and self.k_adapter is not None:
                delta_q = self.q_adapter(x)
                delta_k = self.k_adapter(x)
                if self.support_adapter_gate is not None:
                    delta_q = delta_q * self.support_adapter_gate
                    delta_k = delta_k * self.support_adapter_gate
                delta_q = delta_q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
                delta_k = delta_k.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
                q = q + delta_q
                k = k + delta_k
            if self.value_adapter is not None:
                delta_v = self.value_adapter(x)
                if self.value_adapter_gate is not None:
                    delta_v = delta_v * self.value_adapter_gate
                delta_v = delta_v.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
                v = v + delta_v

            if hasattr(attn_mod, "q_norm"):
                q = attn_mod.q_norm(q)
            if hasattr(attn_mod, "k_norm"):
                k = attn_mod.k_norm(k)

            scale = attn_mod.scale if hasattr(attn_mod, "scale") else head_dim ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale   # (B, H, N, N)

            if hasattr(attn_mod, "attn_drop"):
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = attn_mod.attn_drop(attn_weights)
            else:
                attn_weights = F.softmax(attn_weights, dim=-1)

        else:
            # Fallback: run original attention and return as-is
            return attn_mod(x, **kwargs)

        # ------------------------------------------------------------------
        # 2. Polynomial multi-hop: Y = sum_k gamma_k * A^k * V
        #    Computed via recurrence to avoid explicit matrix powers.
        # ------------------------------------------------------------------
        gamma = self._get_gamma(x)      # (B, K+1) or (1, K+1)

        # A_power[:, :, k] tracks A^k applied to V cumulatively
        # Start: hop_0 = I * V = V  (k=0 term uses identity)
        # We accumulate the weighted sum directly.

        # A^0 * V = V  (identity hop)
        A_pow_V = v.clone()             # (B, H, N, head_dim) — this is A^0 * V = V

        # gamma shape broadcast: (B|1, H|1, 1, 1) per coefficient
        def _g(b_idx: int) -> torch.Tensor:
            return gamma[..., b_idx].unsqueeze(-1).unsqueeze(-1)

        out = _g(0) * A_pow_V           # k=0 term

        prev = v                        # A^0 * V
        for k in range(1, self.num_hops + 1):
            # A^k * V = A * (A^{k-1} * V)
            A_pow_V = attn_weights @ prev   # (B, H, N, head_dim)
            out = out + _g(k) * A_pow_V
            prev = A_pow_V

        # ------------------------------------------------------------------
        # 3. Reshape and project
        # ------------------------------------------------------------------
        out = out.transpose(1, 2).reshape(B, N, attn_dim)      # (B, N, C)
        if hasattr(attn_mod, "norm"):
            out = attn_mod.norm(out)
        pre_proj = out
        out = attn_mod.proj(pre_proj)
        if self.output_adapter is not None:
            delta_out = self.output_adapter(pre_proj)
            if self.output_adapter_gate is not None:
                delta_out = delta_out * self.output_adapter_gate
            out = out + delta_out
        if hasattr(attn_mod, "proj_drop"):
            out = attn_mod.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# MethodModule implementation
# ---------------------------------------------------------------------------

@register_method("polyattn")
class PolyAttnMethod(MethodModule):
    """Polynomial / Multi-hop Attention PEFT.

    Config keys:
        max_hops (int): polynomial degree K. Default 3.
        mode (str): "static" or "dynamic". Default "dynamic".
        headwise (bool): learn separate coefficients per attention head. Default False.
        mlp_ratio (float): hidden ratio for the dynamic coefficient MLP. Default 0.25.
        parameterization (str): "direct" gamma or residual "delta". Default "direct".
        delta_scale (float): scale applied to residual delta coefficients. Default 1.0.
        coeff_activation (str): "none" or "tanh" for delta coefficients. Default "none".
        coeff_dropout (float): dropout on gamma coefficients. Default 0.1.
        dropout_target (str): "all", "non_residual", or "high_order". Default "all".
        norm_constraint (float): max L2 norm for high-order (k>=2) coefficients.
            Set to 0 to disable. Default 1.0.
        value_adapter_rank (int): low-rank ΔV adapter rank. Default 0 disables it.
        value_adapter_alpha (float): ΔV adapter scaling alpha. Default 2 * rank.
        value_adapter_dropout (float): dropout before ΔV adapter. Default 0.0.
        value_adapter_gate_init (float): optional trainable scalar gate for ΔV.
            Omit to preserve the ungated adapter behavior. Default disabled.
        support_adapter_rank (int): low-rank ΔQ/ΔK adapter rank. Default 0 disables it.
        support_adapter_alpha (float): ΔQ/ΔK adapter scaling alpha. Default 2 * rank.
        support_adapter_dropout (float): dropout before ΔQ/ΔK adapters. Default 0.0.
        support_adapter_gate_init (float): optional shared trainable scalar gate for ΔQ/ΔK.
            Omit to preserve the ungated adapter behavior. Default disabled.
        support_adapter_tied (bool): share one residual adapter between Q and K.
            Default False.
        output_adapter_rank (int): low-rank residual adapter rank for the attention
            output projection. Default 0 disables it.
        output_adapter_alpha (float): ΔO adapter scaling alpha. Default 2 * rank.
        output_adapter_dropout (float): dropout before ΔO adapter. Default 0.0.
        output_adapter_gate_init (float): optional trainable scalar gate for ΔO.
            Default disabled.
        qkv_adapter_rank (int): LoRA-style low-rank adapter rank for the packed
            QKV projection. Default 0 disables it.
        qkv_adapter_alpha (float): packed ΔQKV adapter scaling alpha.
            Default 2 * rank.
        qkv_adapter_dropout (float): dropout before packed ΔQKV adapter.
            Default 0.0.
        qkv_adapter_gate_init (float): optional trainable scalar gate for ΔQKV.
            Default disabled.
        dora_qkv_rank (int): DoRA-style packed QKV adapter rank. Default 0.
        dora_qkv_alpha (float): packed DoRA scaling alpha. Default 2 * rank.
        dora_qkv_dropout (float): dropout before packed DoRA QKV adapter.
            Default 0.0.
        dora_qkv_ssf (bool): add trainable scale/shift after packed DoRA QKV.
            Default False.
        dora_qkv_train_bias (bool): make the packed QKV bias trainable inside
            the DoRA path. Default False.
        dora_qkv_gate_init (float): optional trainable scalar gate for blending
            base QKV with DoRA QKV. Default disabled.
        dora_qkv_gate_mode (str): "linear" keeps the legacy unconstrained scalar
            gate; "sigmoid" uses a bounded scalar gate; "dynamic" predicts a
            bounded per-sample gate from pooled tokens. Default "linear".
        dora_qkv_gate_scope (str): "dora" keeps the legacy behavior where
            scale/shift are applied after the gate; "calibrated" gates the
            whole calibrated DoRA-QKV branch. Default "dora".
        dora_qkv_gate_mlp_ratio (float): hidden ratio for dynamic QKV routing.
            Default 0.03125.
        dora_qkv_gate_temperature (float): sigmoid temperature for bounded
            QKV gates. Default 1.0.
    """

    def __init__(self, config: dict) -> None:
        self.max_hops: int = int(config.get("max_hops", 3))
        self.mode: str = config.get("mode", "dynamic")
        self.headwise: bool = _as_bool(config.get("headwise", False))
        self.mlp_ratio: float = float(config.get("mlp_ratio", 0.25))
        self.parameterization: str = config.get("parameterization", "direct")
        self.delta_scale: float = float(config.get("delta_scale", 1.0))
        self.coeff_activation: str = config.get("coeff_activation", "none")
        self.coeff_dropout: float = float(config.get("coeff_dropout", 0.1))
        self.dropout_target: str = config.get("dropout_target", "all")
        self.norm_constraint: float = float(config.get("norm_constraint", 1.0))
        self.value_adapter_rank: int = int(config.get("value_adapter_rank", 0))
        default_adapter_alpha = 2 * self.value_adapter_rank if self.value_adapter_rank > 0 else 1.0
        self.value_adapter_alpha: float = float(
            config.get("value_adapter_alpha", default_adapter_alpha)
        )
        self.value_adapter_dropout: float = float(config.get("value_adapter_dropout", 0.0))
        self.value_adapter_gate_init: Optional[float] = _optional_float(
            config, "value_adapter_gate_init"
        )
        self.support_adapter_rank: int = int(config.get("support_adapter_rank", 0))
        default_support_alpha = 2 * self.support_adapter_rank if self.support_adapter_rank > 0 else 1.0
        self.support_adapter_alpha: float = float(
            config.get("support_adapter_alpha", default_support_alpha)
        )
        self.support_adapter_dropout: float = float(config.get("support_adapter_dropout", 0.0))
        self.support_adapter_gate_init: Optional[float] = _optional_float(
            config, "support_adapter_gate_init"
        )
        self.support_adapter_tied: bool = _as_bool(config.get("support_adapter_tied", False))
        self.output_adapter_rank: int = int(config.get("output_adapter_rank", 0))
        default_output_alpha = 2 * self.output_adapter_rank if self.output_adapter_rank > 0 else 1.0
        self.output_adapter_alpha: float = float(
            config.get("output_adapter_alpha", default_output_alpha)
        )
        self.output_adapter_dropout: float = float(config.get("output_adapter_dropout", 0.0))
        self.output_adapter_gate_init: Optional[float] = _optional_float(
            config, "output_adapter_gate_init"
        )
        self.qkv_adapter_rank: int = int(config.get("qkv_adapter_rank", 0))
        default_qkv_alpha = 2 * self.qkv_adapter_rank if self.qkv_adapter_rank > 0 else 1.0
        self.qkv_adapter_alpha: float = float(
            config.get("qkv_adapter_alpha", default_qkv_alpha)
        )
        self.qkv_adapter_dropout: float = float(config.get("qkv_adapter_dropout", 0.0))
        self.qkv_adapter_gate_init: Optional[float] = _optional_float(
            config, "qkv_adapter_gate_init"
        )
        self.dora_qkv_rank: int = int(config.get("dora_qkv_rank", 0))
        default_dora_qkv_alpha = 2 * self.dora_qkv_rank if self.dora_qkv_rank > 0 else 1.0
        self.dora_qkv_alpha: float = float(
            config.get("dora_qkv_alpha", default_dora_qkv_alpha)
        )
        self.dora_qkv_dropout: float = float(config.get("dora_qkv_dropout", 0.0))
        self.dora_qkv_ssf: bool = _as_bool(config.get("dora_qkv_ssf", False))
        self.dora_qkv_train_bias: bool = _as_bool(
            config.get("dora_qkv_train_bias", False)
        )
        self.dora_qkv_gate_init: Optional[float] = _optional_float(
            config, "dora_qkv_gate_init"
        )
        self.dora_qkv_gate_mode: str = config.get("dora_qkv_gate_mode", "linear")
        self.dora_qkv_gate_scope: str = config.get("dora_qkv_gate_scope", "dora")
        self.dora_qkv_gate_mlp_ratio: float = float(
            config.get("dora_qkv_gate_mlp_ratio", 0.03125)
        )
        self.dora_qkv_gate_temperature: float = float(
            config.get("dora_qkv_gate_temperature", 1.0)
        )
        self.coeff_trainable: bool = _as_bool(config.get("coeff_trainable", True))
        self.first_n_blocks: Optional[int] = _optional_int(config, "first_n_blocks")
        self.last_n_blocks: Optional[int] = _optional_int(config, "last_n_blocks")
        self.target_block_indices = self._parse_target_block_indices(
            config.get("target_block_indices")
        )

        if self.mode not in ("static", "dynamic"):
            raise ValueError(f"PolyAttn mode must be 'static' or 'dynamic', got '{self.mode}'")
        if self.parameterization not in ("direct", "delta"):
            raise ValueError(
                "PolyAttn parameterization must be 'direct' or "
                f"'delta', got '{self.parameterization}'"
            )
        if self.coeff_activation not in ("none", "tanh"):
            raise ValueError(
                "PolyAttn coeff_activation must be 'none' or "
                f"'tanh', got '{self.coeff_activation}'"
            )
        if self.dropout_target not in ("all", "non_residual", "high_order"):
            raise ValueError(
                "PolyAttn dropout_target must be 'all', 'non_residual', or "
                f"'high_order', got '{self.dropout_target}'"
            )
        if self.value_adapter_rank < 0:
            raise ValueError("PolyAttn value_adapter_rank must be non-negative")
        if self.support_adapter_rank < 0:
            raise ValueError("PolyAttn support_adapter_rank must be non-negative")
        if self.output_adapter_rank < 0:
            raise ValueError("PolyAttn output_adapter_rank must be non-negative")
        if self.qkv_adapter_rank < 0:
            raise ValueError("PolyAttn qkv_adapter_rank must be non-negative")
        if self.dora_qkv_rank < 0:
            raise ValueError("PolyAttn dora_qkv_rank must be non-negative")
        if self.dora_qkv_gate_mode not in ("linear", "sigmoid", "dynamic"):
            raise ValueError(
                "PolyAttn dora_qkv_gate_mode must be 'linear', 'sigmoid', "
                f"or 'dynamic', got '{self.dora_qkv_gate_mode}'"
            )
        if self.dora_qkv_gate_scope not in ("dora", "calibrated"):
            raise ValueError(
                "PolyAttn dora_qkv_gate_scope must be 'dora' or "
                f"'calibrated', got '{self.dora_qkv_gate_scope}'"
            )
        if self.dora_qkv_gate_mlp_ratio <= 0.0:
            raise ValueError("PolyAttn dora_qkv_gate_mlp_ratio must be positive")
        if self.dora_qkv_gate_temperature <= 0.0:
            raise ValueError("PolyAttn dora_qkv_gate_temperature must be positive")
        if self.qkv_adapter_rank > 0 and self.dora_qkv_rank > 0:
            raise ValueError("PolyAttn cannot enable both qkv_adapter_rank and dora_qkv_rank")
        if self.first_n_blocks is not None and self.first_n_blocks < 0:
            raise ValueError("PolyAttn first_n_blocks must be non-negative")
        if self.last_n_blocks is not None and self.last_n_blocks < 0:
            raise ValueError("PolyAttn last_n_blocks must be non-negative")

        self._wrappers: List[PolyAttnWrapper] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Replace each transformer block's attention with a PolyAttnWrapper."""
        self._wrappers = []
        candidates = []
        self._collect_attention_modules(backbone, candidates)
        if not candidates:
            raise RuntimeError(
                "PolyAttnMethod.inject: no attention modules found. "
                "Ensure the backbone has timm-style Attention blocks with a .qkv attribute."
            )

        selected = self._select_block_indices(len(candidates))
        for idx, (module, name, child) in enumerate(candidates):
            if idx not in selected:
                continue
            wrapper = self._make_wrapper(child)
            setattr(module, name, wrapper)
            self._wrappers.append(wrapper)

        if not self._wrappers:
            raise RuntimeError(
                "PolyAttnMethod.inject: block selection matched no attention modules."
            )
        return backbone

    def _collect_attention_modules(self, module: nn.Module, candidates: list) -> None:
        for name, child in module.named_children():
            if self._is_attention(child):
                candidates.append((module, name, child))
            else:
                self._collect_attention_modules(child, candidates)

    def _select_block_indices(self, total: int) -> set[int]:
        if (
            self.target_block_indices is None
            and self.first_n_blocks is None
            and self.last_n_blocks is None
        ):
            return set(range(total))

        selected: set[int] = set()
        if self.target_block_indices is not None:
            for idx in self.target_block_indices:
                normalized = idx if idx >= 0 else total + idx
                if 0 <= normalized < total:
                    selected.add(normalized)

        if self.first_n_blocks is not None:
            selected.update(range(min(self.first_n_blocks, total)))

        if self.last_n_blocks is not None:
            start = max(0, total - self.last_n_blocks)
            selected.update(range(start, total))

        return selected

    def _make_wrapper(self, attn: nn.Module) -> PolyAttnWrapper:
        hidden_dim, num_heads = self._infer_dims(attn)
        return PolyAttnWrapper(
            attn=attn,
            num_hops=self.max_hops,
            mode=self.mode,
            coeff_dropout=self.coeff_dropout,
            dropout_target=self.dropout_target,
            norm_constraint=self.norm_constraint,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            headwise=self.headwise,
            mlp_ratio=self.mlp_ratio,
            parameterization=self.parameterization,
            delta_scale=self.delta_scale,
            coeff_activation=self.coeff_activation,
            value_adapter_rank=self.value_adapter_rank,
            value_adapter_alpha=self.value_adapter_alpha,
            value_adapter_dropout=self.value_adapter_dropout,
            value_adapter_gate_init=self.value_adapter_gate_init,
            support_adapter_rank=self.support_adapter_rank,
            support_adapter_alpha=self.support_adapter_alpha,
            support_adapter_dropout=self.support_adapter_dropout,
            support_adapter_gate_init=self.support_adapter_gate_init,
            support_adapter_tied=self.support_adapter_tied,
            output_adapter_rank=self.output_adapter_rank,
            output_adapter_alpha=self.output_adapter_alpha,
            output_adapter_dropout=self.output_adapter_dropout,
            output_adapter_gate_init=self.output_adapter_gate_init,
            qkv_adapter_rank=self.qkv_adapter_rank,
            qkv_adapter_alpha=self.qkv_adapter_alpha,
            qkv_adapter_dropout=self.qkv_adapter_dropout,
            qkv_adapter_gate_init=self.qkv_adapter_gate_init,
            dora_qkv_rank=self.dora_qkv_rank,
            dora_qkv_alpha=self.dora_qkv_alpha,
            dora_qkv_dropout=self.dora_qkv_dropout,
            dora_qkv_ssf=self.dora_qkv_ssf,
            dora_qkv_train_bias=self.dora_qkv_train_bias,
            dora_qkv_gate_init=self.dora_qkv_gate_init,
            dora_qkv_gate_mode=self.dora_qkv_gate_mode,
            dora_qkv_gate_scope=self.dora_qkv_gate_scope,
            dora_qkv_gate_mlp_ratio=self.dora_qkv_gate_mlp_ratio,
            dora_qkv_gate_temperature=self.dora_qkv_gate_temperature,
            coeff_trainable=self.coeff_trainable,
        )

    @staticmethod
    def _parse_target_block_indices(value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return [int(part.strip()) for part in stripped.split(",") if part.strip()]
        return [int(item) for item in value]

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
    def _infer_dims(attn: nn.Module):
        """Return (hidden_dim, num_heads) from a timm Attention module."""
        hidden_dim = attn.proj.in_features
        # timm stores num_heads as .num_heads
        num_heads = getattr(attn, "num_heads", 1)
        return hidden_dim, num_heads

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for w in self._wrappers:
            if w.coeff_trainable:
                if w.mode == "static":
                    params.append(w.gamma)
                else:
                    params.extend(w.coeff_gen.parameters())
            if w.value_adapter is not None:
                params.extend(w.value_adapter.parameters())
            if w.value_adapter_gate is not None:
                params.append(w.value_adapter_gate)
            if w.q_adapter is not None and w.k_adapter is not None:
                params.extend(w.q_adapter.parameters())
                if w.k_adapter is not w.q_adapter:
                    params.extend(w.k_adapter.parameters())
            if w.support_adapter_gate is not None:
                params.append(w.support_adapter_gate)
            if w.output_adapter is not None:
                params.extend(w.output_adapter.parameters())
            if w.output_adapter_gate is not None:
                params.append(w.output_adapter_gate)
            if w.qkv_adapter is not None:
                params.extend(w.qkv_adapter.parameters())
            if w.qkv_adapter_gate is not None:
                params.append(w.qkv_adapter_gate)
            if w.dora_qkv_adapter is not None:
                params.extend(w.dora_qkv_adapter.parameters())
            if w.dora_qkv_gate is not None:
                params.append(w.dora_qkv_gate)
            if w.dora_qkv_gate_router is not None:
                params.extend(w.dora_qkv_gate_router.parameters())
            if w.dora_qkv_scale is not None:
                params.append(w.dora_qkv_scale)
            if w.dora_qkv_shift is not None:
                params.append(w.dora_qkv_shift)
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}
