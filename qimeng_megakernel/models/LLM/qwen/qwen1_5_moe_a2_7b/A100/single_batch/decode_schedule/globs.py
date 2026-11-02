"""Globals and tensor ownership for Qwen1.5-MoE A100 decode."""

from dataclasses import dataclass, fields
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
    shared_expert_gate_weights: Optional[torch.Tensor] = None  # (L, hidden, 1)
    # MoE: router
    router_weights: Optional[torch.Tensor] = None  # (L, num_experts, hidden)
    router_logits_buf: Optional[torch.Tensor] = None  # (num_experts,) float32, filled by Router_GEMV
    # MoE: routed experts [layer_idx][expert_idx] -> (gate_up_proj, down_proj) or stacked
    expert_gate_up_proj_weights: Optional[torch.Tensor] = None  # (L, num_experts, 2*moe_intermediate, hidden)
    expert_down_proj_weights: Optional[torch.Tensor] = None    # (L, num_experts, hidden, moe_intermediate)
    expert_gate_up_proj_weights_flat: Optional[torch.Tensor] = None
    expert_down_proj_weights_flat: Optional[torch.Tensor] = None
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
    # MoE: RMS-normalized hidden state published by opcode 8.
    normed_hidden_states: Optional[torch.Tensor] = None  # (hidden_size,) bf16
    # MoE: shared expert intermediate (gate*up after SiLU, size 5632) and output
    shared_expert_silu_out: Optional[torch.Tensor] = None  # (5632,) for PyVM shared UpGate
    shared_out: Optional[torch.Tensor] = None
    # MoE: routed experts weighted sum (accumulator)
    routed_out: Optional[torch.Tensor] = None
    routed_out_f32: Optional[torch.Tensor] = None       # float32 accumulator for atomic-add
    routed_silu_buffer: Optional[torch.Tensor] = None   # (num_experts_per_tok, moe_intermediate)
    # MoE: router output
    expert_indices: Optional[torch.Tensor] = None   # (num_experts_per_tok,) int64
    expert_weights: Optional[torch.Tensor] = None  # (num_experts_per_tok,) float
    # MLP intermediate (for routed expert upgate: per-expert or shared buffer)
    silu_out: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    # Prefill-only (optional)
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

NOT_PREPROCESS_OWNED_FIELDS = {
    "instructions": "scheduler",
    "hidden_states_seq": "prefill-only",
    "prefill_q_seq": "prefill-only",
    "attn_out_seq": "prefill-only",
}
