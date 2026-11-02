from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Globals:
    # instructions / sync (megakernel-related)
    instructions: Optional[torch.Tensor] = None
    barriers: Optional[torch.Tensor] = None
    position_id: Optional[int] = None

    # weights (ordered by a layer's compute flow)
    attn_norm_weights: Optional[torch.Tensor] = None
    qkv_weights: Optional[torch.Tensor] = None
    rope_cos: Optional[torch.Tensor] = None
    rope_sin: Optional[torch.Tensor] = None
    o_proj_weights: Optional[torch.Tensor] = None
    mlp_norm_weights: Optional[torch.Tensor] = None
    up_proj_weights: Optional[torch.Tensor] = None
    gate_proj_weights: Optional[torch.Tensor] = None
    down_proj_weights: Optional[torch.Tensor] = None
    lm_head_norm_weights: Optional[torch.Tensor] = None
    lm_head_weights: Optional[torch.Tensor] = None

    # buffers (ordered by a layer's compute flow)
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    q_post_rope: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    attn_out_intermediates: Optional[torch.Tensor] = None
    attn_lse_intermediates: Optional[torch.Tensor] = None
    silu_out: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    # Prefill-only: sequence-sized buffers [seq_len, ...]. Set when running fused prefill (same as 1B H100).
    hidden_states_seq: Optional[torch.Tensor] = None  # [S, hidden_size]
    prefill_q_seq: Optional[torch.Tensor] = None    # [S, n_heads, head_dim] for causal SDPA
    attn_out_seq: Optional[torch.Tensor] = None    # [S, n_heads * head_dim] before o_proj
