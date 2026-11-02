from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class Globals:
    instructions: Optional[torch.Tensor] = None
    barriers: Optional[torch.Tensor] = None
    position_id: Optional[int] = None

    attn_norm_weights: Optional[torch.Tensor] = None
    qkv_weights: Optional[torch.Tensor] = None
    qkv_bias: Optional[torch.Tensor] = None
    rope_cos: Optional[torch.Tensor] = None
    rope_sin: Optional[torch.Tensor] = None
    o_proj_weights: Optional[torch.Tensor] = None
    mlp_norm_weights: Optional[torch.Tensor] = None
    up_proj_weights: Optional[torch.Tensor] = None
    gate_proj_weights: Optional[torch.Tensor] = None
    down_proj_weights: Optional[torch.Tensor] = None
    lm_head_norm_weights: Optional[torch.Tensor] = None
    lm_head_weights: Optional[torch.Tensor] = None

    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    q_post_rope: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    attn_out_intermediates: Optional[torch.Tensor] = None
    attn_lse_intermediates: Optional[torch.Tensor] = None
    silu_out: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    hidden_states_seq: Optional[torch.Tensor] = None
    prefill_q_seq: Optional[torch.Tensor] = None
    attn_out_seq: Optional[torch.Tensor] = None

    def diff(self, other: "Globals", skip_kv_cache: bool = False):
        for field in fields(self):
            name = field.name
            attr = getattr(self, name)
            other_attr = getattr(other, name)
            if not isinstance(attr, torch.Tensor):
                continue
            if "weights" in name or "bias" in name or name in {"rope_cos", "rope_sin"}:
                continue
            if skip_kv_cache and "cache" in name:
                continue
            diff_tensors(attr, other_attr, name)


def diff_tensors(a: torch.Tensor, b: torch.Tensor, name: str):
    a = a.float()
    b = b.float()
    diff = a - b
    adiff = diff.abs()
    rdiff = 2 * adiff / (a.abs() + b.abs() + 1e-6)
    print(f"{name}: max adiff: {adiff.max()}, mean rdiff: {rdiff.mean()}")
    return diff, adiff, rdiff
