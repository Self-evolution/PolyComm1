"""HeadInteraction — Nonlinear Head Interaction PEFT.

Modifies the attention output by mixing attention maps across heads using
either quadratic (second-order) interactions or a low-dimensional head
embedding mixer.

Two modes:

  quadratic:
    output_h = sum_i alpha_{h,i} * A_i + sum_{i<j} beta_{h,ij} * (A_i ⊙ A_j)

    where A_i is the attention map of head i, ⊙ is element-wise product,
    alpha has shape (num_heads, num_heads) and beta has shape
    (num_heads, num_heads*(num_heads-1)//2).

  mixer:
    Each head has a low-dimensional embedding e_i ∈ R^{embed_dim}.
    A tiny MLP takes [e_i; e_j] (concat) and outputs a scalar mixing weight
    w_{ij}. The output is:
      output_h = sum_i w_{h,i} * A_i + sum_{i<j} w_{h,ij} * (A_i ⊙ A_j)

    where w_{h,i} and w_{h,ij} are produced by the MLP from head embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodModule
from src.methods.registry import register_method


# ---------------------------------------------------------------------------
# Tiny MLP for mixer mode
# ---------------------------------------------------------------------------

class HeadMixerMLP(nn.Module):
    """Tiny MLP: embed_dim*2 -> hidden -> 1.

    Used in mixer mode to compute scalar mixing weights from pairs of
    head embeddings.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        hidden = max(4, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Init output near zero so initial mixing is near-identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ei: torch.Tensor, ej: torch.Tensor) -> torch.Tensor:
        """Compute scalar weight from two head embeddings.

        Args:
            ei: (embed_dim,) or (B, embed_dim)
            ej: (embed_dim,) or (B, embed_dim)
        Returns:
            scalar weight, shape () or (B,)
        """
        return self.net(torch.cat([ei, ej], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Per-block attention wrapper
# ---------------------------------------------------------------------------

class HeadInteractionWrapper(nn.Module):
    """Wraps a single transformer attention module to apply head interaction.

    Intercepts the attention forward and modifies the output by mixing
    attention maps across heads.

    quadratic mode:
        output_h = sum_i alpha_{h,i} * A_i * V_i
                 + sum_{i<j} beta_{h,ij} * (A_i ⊙ A_j) * V_i

    mixer mode:
        Mixing weights are computed from head embeddings via a tiny MLP.
        output_h = sum_i w(e_h, e_i) * A_i * V_i
                 + sum_{i<j} w(e_h, e_i, e_j) * (A_i ⊙ A_j) * V_i
    """

    def __init__(
        self,
        attn: nn.Module,
        mode: str,
        num_heads: int,
        hidden_dim: int,
        embed_dim: int,
        use_quadratic: bool,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mode = mode
        self.num_heads = num_heads
        self.use_quadratic = use_quadratic
        num_pairs = num_heads * (num_heads - 1) // 2

        if mode == "quadratic":
            # Linear mixing coefficients: alpha[h, i] = weight of head i for output h
            self.alpha = nn.Parameter(torch.eye(num_heads))  # (H, H) residual init
            # Second-order coefficients (upper triangle pairs only)
            self.beta = nn.Parameter(torch.zeros(num_heads, num_pairs))
            self.head_embed: Optional[nn.Parameter] = None
            self.mixer: Optional[HeadMixerMLP] = None
        elif mode == "mixer":
            # Low-dimensional head embeddings
            self.head_embed = nn.Parameter(
                torch.randn(num_heads, embed_dim) * 0.01
            )
            self.mixer = HeadMixerMLP(embed_dim)
            self.alpha = None  # type: ignore[assignment]
            self.beta = None   # type: ignore[assignment]
        else:
            raise ValueError(
                f"HeadInteraction mode must be 'quadratic' or 'mixer', got '{mode}'"
            )

        # Precompute upper-triangle index pairs (i < j)
        pairs = [(i, j) for i in range(num_heads) for j in range(i + 1, num_heads)]
        self.register_buffer(
            "pair_i", torch.tensor([p[0] for p in pairs], dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor([p[1] for p in pairs], dtype=torch.long)
        )

    def _compute_mixing_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (alpha_w, beta_w) mixing weights for quadratic and mixer modes.

        Returns:
            alpha_w: (H, H) — linear mixing weights
            beta_w:  (H, num_pairs) — quadratic mixing weights
        """
        if self.mode == "quadratic":
            return self.alpha, self.beta
        else:
            # mixer mode: compute weights from head embeddings via MLP
            H = self.num_heads
            num_pairs = len(self.pair_i)
            embeds = self.head_embed  # (H, embed_dim)

            # Linear weights: w(e_h, e_i) for each (h, i) pair
            alpha_w = torch.zeros(H, H, device=embeds.device, dtype=embeds.dtype)
            for h in range(H):
                for i in range(H):
                    alpha_w[h, i] = self.mixer(embeds[h], embeds[i])

            # Quadratic weights: w(e_h, e_i + e_j) for each (h, pair) combination
            beta_w = torch.zeros(H, num_pairs, device=embeds.device, dtype=embeds.dtype)
            for h in range(H):
                for p_idx in range(num_pairs):
                    i = self.pair_i[p_idx].item()
                    j = self.pair_j[p_idx].item()
                    # Use average of pair embeddings as the "pair" representation
                    e_pair = (embeds[i] + embeds[j]) * 0.5
                    beta_w[h, p_idx] = self.mixer(embeds[h], e_pair)

            return alpha_w, beta_w

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Intercept attention forward; apply head interaction mixing."""
        B, N, C = x.shape
        head_dim = C // self.num_heads
        attn_mod = self.attn

        if not (hasattr(attn_mod, "qkv") and isinstance(attn_mod.qkv, nn.Linear)):
            return attn_mod(x, **kwargs)

        # Compute Q, K, V
        qkv = attn_mod.qkv(x)                                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                        # each (B, H, N, head_dim)
        del qkv

        scale = attn_mod.scale if hasattr(attn_mod, "scale") else head_dim ** -0.5
        attn_logits = (q @ k.transpose(-2, -1)) * scale                # (B, H, N, N)
        del q, k

        if hasattr(attn_mod, "attn_drop"):
            A = F.softmax(attn_logits, dim=-1)
            del attn_logits
            A = attn_mod.attn_drop(A)
        else:
            A = F.softmax(attn_logits, dim=-1)                         # (B, H, N, N)
            del attn_logits

        # Get mixing weights
        alpha_w, beta_w = self._compute_mixing_weights()               # (H,H), (H,P)

        # Linear mixing: output_h = sum_i alpha_{h,i} * A_i * V_i
        # A: (B, H, N, N), v: (B, H, N, head_dim)
        # alpha_w: (H, H) -> broadcast over batch and spatial dims
        # out_linear[b, h, n, d] = sum_i alpha_w[h, i] * (A[b,i] @ v[b,i])[n, d]
        AV = A @ v                                                      # (B, H, N, head_dim)
        # alpha_w[h, i]: weight of head i for output head h
        # out_linear[b, h] = sum_i alpha_w[h, i] * AV[b, i]
        # Use einsum: bhnd, hi -> bhnd (sum over i)
        out = torch.einsum("hi,bind->bhnd", alpha_w, AV)               # (B, H, N, head_dim)

        # Quadratic mixing: sum_{i<j} beta_{h,ij} * (A_i ⊙ A_j) * V_i
        if self.use_quadratic and len(self.pair_i) > 0:
            pair_i = self.pair_i  # (P,)
            pair_j = self.pair_j  # (P,)

            # Process one pair at a time to minimize peak memory
            out_quad = torch.zeros_like(out)  # (B, H, N, head_dim)
            for p in range(len(pair_i)):
                pi, pj = pair_i[p], pair_j[p]
                A_cross = A[:, pi, :, :] * A[:, pj, :, :]  # (B, N, N)
                AV_c = A_cross @ v[:, pi, :, :]             # (B, N, head_dim)
                # beta_w[:, p] is (H,) — weight for each output head
                out_quad += beta_w[:, p].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * AV_c.unsqueeze(1)
                del A_cross, AV_c

            out = out + out_quad

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)                     # (B, N, C)
        out = attn_mod.proj(out)
        if hasattr(attn_mod, "proj_drop"):
            out = attn_mod.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# MethodModule implementation
# ---------------------------------------------------------------------------

@register_method("head_interaction")
class HeadInteractionMethod(MethodModule):
    """Nonlinear Head Interaction PEFT.

    Modifies attention output by mixing attention maps across heads using
    either quadratic (second-order) interactions or a low-dimensional head
    embedding mixer.

    Config keys:
        mode (str): "quadratic" or "mixer". Default "quadratic".
        embed_dim (int): head embedding dimension for mixer mode. Default 8.
        use_quadratic (bool): whether to include second-order (A_i ⊙ A_j) terms.
            Default True.
    """

    def __init__(self, config: dict) -> None:
        self.mode: str = config.get("mode", "quadratic")
        self.embed_dim: int = int(config.get("embed_dim", 8))
        self.use_quadratic: bool = bool(config.get("use_quadratic", True))

        if self.mode not in ("quadratic", "mixer"):
            raise ValueError(
                f"HeadInteraction mode must be 'quadratic' or 'mixer', got '{self.mode}'"
            )

        self._wrappers: List[HeadInteractionWrapper] = []

    # ------------------------------------------------------------------
    # MethodModule interface
    # ------------------------------------------------------------------

    def inject(self, backbone: nn.Module) -> nn.Module:
        """Replace each transformer block's attention with a HeadInteractionWrapper."""
        self._wrappers = []
        self._inject_recursive(backbone)
        if not self._wrappers:
            raise RuntimeError(
                "HeadInteractionMethod.inject: no attention modules found. "
                "Ensure the backbone has timm-style Attention blocks with a .qkv attribute."
            )
        return backbone

    def _inject_recursive(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if self._is_attention(child):
                hidden_dim, num_heads = self._infer_dims(child)
                wrapper = HeadInteractionWrapper(
                    attn=child,
                    mode=self.mode,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    embed_dim=self.embed_dim,
                    use_quadratic=self.use_quadratic,
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
    def _infer_dims(attn: nn.Module):
        """Return (hidden_dim, num_heads) from a timm Attention module."""
        hidden_dim = attn.proj.in_features
        num_heads = getattr(attn, "num_heads", 1)
        return hidden_dim, num_heads

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for w in self._wrappers:
            if w.mode == "quadratic":
                params.append(w.alpha)
                params.append(w.beta)
            else:
                params.append(w.head_embed)
                params.extend(w.mixer.parameters())
        return params

    def get_param_count(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": trainable, "trainable": trainable}

    def is_compatible_with(self, other: "MethodModule") -> bool:
        return True
