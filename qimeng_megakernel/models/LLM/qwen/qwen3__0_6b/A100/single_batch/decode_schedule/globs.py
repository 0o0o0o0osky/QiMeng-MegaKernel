"""Globals dataclass for Qwen3-0.6B A100 decode.

Differences from Qwen2.5-3B Globals:
- Added q_norm_weights and k_norm_weights: per-layer per-head RMSNorm (shape [L, head_dim])
- Removed qkv_bias: Qwen3 has no attention bias
"""
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
    # Qwen3: no qkv_bias (attention_bias=False)
    q_norm_weights: Optional[torch.Tensor] = None   # [L, head_dim] — per-head RMSNorm for Q
    k_norm_weights: Optional[torch.Tensor] = None   # [L, head_dim] — per-head RMSNorm for K
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
    qkv_raw: Optional[torch.Tensor] = None   # [qkv_out_dim=4096] raw QKV before Q/K-norm and RoPE
    q_post_rope: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    attn_out_intermediates: Optional[torch.Tensor] = None
    attn_lse_intermediates: Optional[torch.Tensor] = None
    silu_out: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    # Prefill-only: sequence-sized buffers [seq_len, ...]. Set when running fused prefill.
    hidden_states_seq: Optional[torch.Tensor] = None  # [S, hidden_size]
    prefill_q_seq: Optional[torch.Tensor] = None    # [S, n_heads, head_dim] for causal SDPA
    attn_out_seq: Optional[torch.Tensor] = None    # [S, n_heads * head_dim] before o_proj

    def diff(self, other: "Globals", skip_kv_cache: bool = False):
        for field in fields(self):
            name = field.name
            attr = getattr(self, name)
            other_attr = getattr(other, name)
            if not isinstance(attr, torch.Tensor):
                continue
            if "weights" in name or name in {"rope_cos", "rope_sin"}:
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
