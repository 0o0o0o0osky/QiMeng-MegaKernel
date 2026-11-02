"""Globals and ownership accounting for Qwen1.5-MoE-A2.7B H100 decode.

The flattened expert weights are zero-copy views that combine the layer and expert
axes for the routed-expert GEMV. ``preprocess_decode`` owns decode allocations, the
scheduler owns ``instructions``, and the sequence buffers belong to prefill.
"""

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
    qkv_bias: Optional[torch.Tensor] = None  # (L, qkv_dim) when q_proj/k_proj/v_proj have bias
    rope_cos: Optional[torch.Tensor] = None
    rope_sin: Optional[torch.Tensor] = None
    o_proj_weights: Optional[torch.Tensor] = None
    mlp_norm_weights: Optional[torch.Tensor] = None
    # MoE: shared expert (one MLP per layer)
    shared_expert_gate_proj_weights: Optional[torch.Tensor] = None
    shared_expert_up_proj_weights: Optional[torch.Tensor] = None
    shared_expert_down_proj_weights: Optional[torch.Tensor] = None
    shared_expert_gate_weights: Optional[torch.Tensor] = None  # (L, 1, hidden)
    # MoE: router
    router_weights: Optional[torch.Tensor] = None  # (L, num_experts, hidden)
    router_logits_buf: Optional[torch.Tensor] = None  # (num_experts,) float32, filled by Router_GEMV
    # MoE: routed experts
    expert_gate_up_proj_weights: Optional[torch.Tensor] = None  # (L, E, 2*moe_intermediate, hidden)
    expert_down_proj_weights: Optional[torch.Tensor] = None     # (L, E, hidden, moe_intermediate)
    # MoE: routed experts, zero-copy 3D views for ThunderKittens gl<> (max 4D)
    expert_gate_up_proj_weights_flat: Optional[torch.Tensor] = None  # (L*E, 2*moe_intermediate, hidden)
    expert_down_proj_weights_flat: Optional[torch.Tensor] = None     # (L*E, hidden, moe_intermediate)
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
    # MoE: RMS-normed hidden_states (written by shared UpGate when tile 0 runs), reused downstream
    normed_hidden_states: Optional[torch.Tensor] = None  # (hidden_size,) bf16
    # MoE: shared expert intermediate (gate*up after SiLU) and output
    shared_expert_silu_out: Optional[torch.Tensor] = None  # (shared_expert_intermediate_size,)
    shared_out: Optional[torch.Tensor] = None
    # MoE: routed experts weighted sum (accumulator)
    routed_out: Optional[torch.Tensor] = None
    routed_out_f32: Optional[torch.Tensor] = None       # float32 accumulator for atomic-add
    routed_silu_buffer: Optional[torch.Tensor] = None   # (num_experts_per_tok, moe_intermediate)
    # MoE: router output
    expert_indices: Optional[torch.Tensor] = None   # (num_experts_per_tok,) int64
    expert_weights: Optional[torch.Tensor] = None   # (num_experts_per_tok,) float32
    # MLP intermediate (routed expert upgate scratch)
    silu_out: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    # Prefill-only (optional)
    hidden_states_seq: Optional[torch.Tensor] = None
    prefill_q_seq: Optional[torch.Tensor] = None
    attn_out_seq: Optional[torch.Tensor] = None

#: Fields allocated by ``preprocess_decode`` for the decode route.
DECODE_ALLOCATED_FIELDS = (
    "barriers",
    "position_id",
    "attn_norm_weights",
    "qkv_weights",
    "qkv_bias",
    "rope_cos",
    "rope_sin",
    "o_proj_weights",
    "mlp_norm_weights",
    "shared_expert_gate_proj_weights",
    "shared_expert_up_proj_weights",
    "shared_expert_down_proj_weights",
    "shared_expert_gate_weights",
    "router_weights",
    "router_logits_buf",
    "expert_gate_up_proj_weights",
    "expert_down_proj_weights",
    "expert_gate_up_proj_weights_flat",
    "expert_down_proj_weights_flat",
    "lm_head_norm_weights",
    "lm_head_weights",
    "k_cache",
    "v_cache",
    "hidden_states",
    "q_post_rope",
    "attn_out",
    "attn_out_intermediates",
    "attn_lse_intermediates",
    "normed_hidden_states",
    "shared_expert_silu_out",
    "shared_out",
    "routed_out",
    "routed_out_f32",
    "routed_silu_buffer",
    "expert_indices",
    "expert_weights",
    "silu_out",
    "logits",
)

#: Fields owned by schedule construction or prefill.
NOT_PREPROCESS_OWNED_FIELDS = {
    "instructions": "scheduler (tensorize_instructions)",
    "hidden_states_seq": "prefill-only",
    "prefill_q_seq": "prefill-only",
    "attn_out_seq": "prefill-only",
}
