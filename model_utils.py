"""Original TMM model construction, attention execution and parameter IO."""
from pathlib import Path
import types

import torch
from torch import nn
import torch.nn.functional as F
import timm

from src.methods import create_method
from src.methods.novel.polyattn import PolyAttnWrapper

ROOT = Path(__file__).resolve().parent


def fast_poly_forward(self, x, **kwargs):
    """Same polynomial recurrence, using SDPA when attention dropout is zero.

    Each recurrence evaluates the SAME A(q,k) times the preceding value. The
    frozen-coefficient configuration is exactly one hop, with no discarded hops.
    Nonzero attention dropout uses the original implementation/mask semantics.
    """
    attn = self.attn
    if getattr(getattr(attn, 'attn_drop', None), 'p', 0.0) != 0:
        return self._campaign_original_forward(x, **kwargs)
    b, n, c = x.shape
    hd = getattr(attn, 'head_dim', c // self.num_heads)
    ad = getattr(attn, 'attn_dim', hd * self.num_heads)
    if self.dora_qkv_adapter is not None:
        dq = self.dora_qkv_adapter(x)
        cq = dq
        if self.dora_qkv_scale is not None:
            cq = self.dora_qkv_scale * dq + self.dora_qkv_shift
        gate = self._get_dora_qkv_gate(x)
        if gate is None:
            qkv = cq
        else:
            base = attn.qkv(x)
            qkv = base + gate * ((cq if self.dora_qkv_gate_scope == 'calibrated' else dq) - base)
            if self.dora_qkv_gate_scope != 'calibrated' and self.dora_qkv_scale is not None:
                qkv = self.dora_qkv_scale * qkv + self.dora_qkv_shift
    else:
        qkv = attn.qkv(x)
    if self.qkv_adapter is not None:
        delta = self.qkv_adapter(x)
        qkv = qkv + delta * (self.qkv_adapter_gate if self.qkv_adapter_gate is not None else 1)
    q, k, v = qkv.reshape(b, n, 3, self.num_heads, hd).permute(2, 0, 3, 1, 4).unbind(0)
    if self.q_adapter is not None:
        gate = self.support_adapter_gate if self.support_adapter_gate is not None else 1
        q = q + (self.q_adapter(x) * gate).reshape(b, n, self.num_heads, hd).permute(0, 2, 1, 3)
        k = k + (self.k_adapter(x) * gate).reshape(b, n, self.num_heads, hd).permute(0, 2, 1, 3)
    if self.value_adapter is not None:
        delta = self.value_adapter(x)
        if self.value_adapter_gate is not None:
            delta = delta * self.value_adapter_gate
        v = v + delta.reshape(b, n, self.num_heads, hd).permute(0, 2, 1, 3)
    q, k = getattr(attn, 'q_norm', nn.Identity())(q), getattr(attn, 'k_norm', nn.Identity())(k)
    if not self.coeff_trainable:
        out = F.scaled_dot_product_attention(q, k, v, scale=getattr(attn, 'scale', hd ** -.5))
    else:
        gamma = self._get_gamma(x)
        out = gamma[..., 0, None, None] * v
        prev = v
        for hop in range(1, self.num_hops + 1):
            prev = F.scaled_dot_product_attention(q, k, prev, scale=getattr(attn, 'scale', hd ** -.5))
            out = out + gamma[..., hop, None, None] * prev
    out = out.transpose(1, 2).reshape(b, n, ad)
    out = getattr(attn, 'norm', nn.Identity())(out)
    projected = attn.proj(out)
    if self.output_adapter is not None:
        delta = self.output_adapter(out)
        if self.output_adapter_gate is not None:
            delta = delta * self.output_adapter_gate
        projected = projected + delta
    return getattr(attn, 'proj_drop', nn.Identity())(projected)


def build_model(job, classes, device):
    backbone_name = job.get('backbone', 'vit_base_patch16_224')
    model = timm.create_model(backbone_name, pretrained=False, num_classes=0, drop_path_rate=.1)
    from timm.models.vision_transformer import _load_weights
    weights = ROOT / 'checkpoints/pretrained' / job.get('weights', 'ViT-B_16_in21k.npz')
    _load_weights(model, str(weights))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    method_cfg = dict(job['method'])
    method = create_method(method_cfg['name'], method_cfg)
    model = method.inject(model)
    trainable_ids = {id(p) for p in method.get_trainable_params()}
    # A registered but unused branch must not accidentally unfreeze the backbone.
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    if job.get('fixed_gate') is not None:
        for module in model.modules():
            if isinstance(module, PolyAttnWrapper):
                if module.dora_qkv_gate is not None:
                    module.dora_qkv_gate.data.fill_(float(job['fixed_gate']))
                    module.dora_qkv_gate.requires_grad_(False)
                if module.dora_qkv_gate_router is not None:
                    raise ValueError('Fixed gate jobs must use linear gate mode')
    for module in model.modules():
        if isinstance(module, PolyAttnWrapper):
            module._campaign_original_forward = module.forward
            module.forward = types.MethodType(fast_poly_forward, module)
    head = nn.Linear(model.num_features, classes)
    return model.to(device), head.to(device)


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_adapter(model, head, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if expected != set(state['adapter']):
        raise ValueError('Adapter checkpoint parameter set does not match this configuration')
    model.load_state_dict(state['adapter'], strict=False)
    head.load_state_dict(state['head'], strict=True)
