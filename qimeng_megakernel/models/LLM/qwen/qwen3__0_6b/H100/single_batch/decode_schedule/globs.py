from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Globals:
    instructions: Optional[torch.Tensor] = None
    barriers: Optional[torch.Tensor] = None
    position_id: Optional[int] = None

    attn_norm_weights: Optional[torch.Tensor] = None
    q_norm_weights: Optional[torch.Tensor] = None  # Qwen3: RMSNorm on Q per head_dim (L, head_dim)
    k_norm_weights: Optional[torch.Tensor] = None  # Qwen3: RMSNorm on K per head_dim (L, head_dim)
    qkv_weights: Optional[torch.Tensor] = None
    qkv_bias: Optional[torch.Tensor] = None  # Qwen3-0.6B: attention_bias=False, always None
    rope_cos: Optional[torch.Tensor] = None
    rope_sin: Optional[torch.Tensor] = None
    o_proj_weights: Optional[torch.Tensor] = None
    mlp_norm_weights: Optional[torch.Tensor] = None
    up_proj_weights: Optional[torch.Tensor] = None
    gate_proj_weights: Optional[torch.Tensor] = None
    down_proj_weights: Optional[torch.Tensor] = None
    lm_head_norm_weights: Optional[torch.Tensor] = None
    lm_head_weights: Optional[torch.Tensor] = None
    # Host-only embedding table retained from fused prefill for central decode.
    embed_tokens: Optional[torch.Tensor] = None

    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    qkv_raw: Optional[torch.Tensor] = None
    q_post_rope: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    attn_out_intermediates: Optional[torch.Tensor] = None
    attn_lse_intermediates: Optional[torch.Tensor] = None
    silu_out: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    # Prefill-only: sequence-sized buffers [seq_len, ...]. Set when running fused prefill.
    hidden_states_seq: Optional[torch.Tensor] = None
    prefill_qkv_raw: Optional[torch.Tensor] = None
    prefill_q_seq: Optional[torch.Tensor] = None
    attn_out_seq: Optional[torch.Tensor] = None
