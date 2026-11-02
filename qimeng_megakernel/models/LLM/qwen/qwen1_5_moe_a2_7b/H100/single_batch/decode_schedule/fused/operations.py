"""
Fused decode operations for Qwen1.5-MoE-A2.7B: attention (Llama-like) + MoE block.

Used for baseline_to_fused: HF forward vs these fused ops.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..globs import Globals
from ..model_config import QwenMoeConfig


def _rms_norm(
    inp: Tensor, weight: Tensor, eps: float, use_float32: bool = True
) -> Tensor:
    """RMSNorm; optionally keep variance and rsqrt in the input dtype."""
    input_dtype = inp.dtype
    if use_float32:
        inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)
    return weight * inp.to(input_dtype)


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    new_x1 = -x2
    new_x2 = x1
    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)


def _apply_rotary_pos_emb_interleaved(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def fused_rms_qkvproj_gemv_rope(globs: Globals, layer_idx: int) -> None:
    """Decode token path: RMS + QKV + bias + RoPE; aligned with Qwen2.5 rms_qkv_bias_rope."""
    pos_id = int(globs.position_id) if globs.position_id is not None else 0
    eps = QwenMoeConfig.rms_norm_eps
    n_heads = QwenMoeConfig.num_attention_heads
    n_kv_heads = QwenMoeConfig.num_key_value_heads
    head_dim = QwenMoeConfig.head_dim
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    post_ln = _rms_norm(
        globs.hidden_states,
        globs.attn_norm_weights[layer_idx],
        eps=eps,
    )
    post_ln_1 = post_ln.unsqueeze(0)
    qkv = F.linear(post_ln_1, globs.qkv_weights[layer_idx]).squeeze(0)
    if globs.qkv_bias is not None:
        qkv = qkv + globs.qkv_bias[layer_idx].to(qkv.dtype)
    q = qkv[:k_start].view(1, n_heads, head_dim)
    k = qkv[k_start:v_start].view(1, n_kv_heads, head_dim)
    v = qkv[v_start:].view(1, n_kv_heads, head_dim)

    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
        q, k, cos, sin, unsqueeze_dim=0
    )
    q_rope = q_rope.to(globs.q_post_rope.dtype)
    k_rope = k_rope.to(globs.k_cache.dtype)

    globs.q_post_rope.copy_(q_rope.view(-1))
    n_h = globs.k_cache.shape[3]
    n_d = globs.k_cache.shape[4]
    assert k_rope.numel() == n_h * n_d
    globs.k_cache[layer_idx, 0, pos_id, :, :].copy_(k_rope.reshape(n_h, n_d))
    globs.v_cache[layer_idx, 0, pos_id, :, :].copy_(
        v.reshape(n_h, n_d).to(globs.v_cache.dtype)
    )


def fused_decode_attention(globs: Globals, layer_idx: int) -> None:
    """Decode attention with cache read range [:position_id+1]."""
    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = QwenMoeConfig.num_attention_heads
    n_kv_heads = QwenMoeConfig.num_key_value_heads
    head_dim = QwenMoeConfig.head_dim
    q = globs.q_post_rope.view(1, n_heads, 1, head_dim).float()
    k = (
        globs.k_cache[layer_idx, 0, :seq_len, :, :]
        .permute(1, 0, 2)
        .unsqueeze(0)
        .float()
    )
    v = (
        globs.v_cache[layer_idx, 0, :seq_len, :, :]
        .permute(1, 0, 2)
        .unsqueeze(0)
        .float()
    )
    try:
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=True
        )
    except TypeError:
        gqa_ratio = n_heads // n_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).reshape(
            1, n_heads, seq_len, head_dim
        )
        v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).reshape(
            1, n_heads, seq_len, head_dim
        )
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    globs.attn_out.copy_(
        attn_out.squeeze(0).squeeze(1).reshape(-1).to(globs.attn_out.dtype)
    )


def fused_oproj_gemv_residual(globs: Globals, layer_idx: int) -> None:
    """Full layer: attn_out @ o_proj + residual -> hidden_states. Use F.linear to align with fused prefill."""
    attn_out_1 = globs.attn_out.unsqueeze(0)
    out = F.linear(attn_out_1, globs.o_proj_weights[layer_idx]).squeeze(0)
    globs.hidden_states += out.to(globs.hidden_states.dtype)


def fused_rms_lmhead_gemv(globs: Globals) -> None:
    """Final decode head: RMSNorm(hidden_states) -> lm_head projection."""
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.lm_head_norm_weights,
        eps=QwenMoeConfig.rms_norm_eps,
    )
    logits = F.linear(
        post_ln.unsqueeze(0).to(globs.lm_head_weights.dtype),
        globs.lm_head_weights,
    ).squeeze(0)
    globs.logits.copy_(logits.to(globs.logits.dtype))


# -------- MoE --------


def fused_shared_mlp(
    globs: Globals, layer_idx: int, normed: Tensor | None = None
) -> None:
    """Shared expert branch; optional precomputed normed hidden_states."""
    if normed is None:
        normed = _rms_norm(
            globs.hidden_states,
            globs.mlp_norm_weights[layer_idx],
            eps=QwenMoeConfig.rms_norm_eps,
        )
    normed_1 = normed.unsqueeze(0)
    gate = F.linear(
        normed_1, globs.shared_expert_gate_proj_weights[layer_idx]
    ).squeeze(0)
    up = F.linear(normed_1, globs.shared_expert_up_proj_weights[layer_idx]).squeeze(0)
    x = F.silu(gate.to(torch.float32)) * up.to(torch.float32)
    shared = F.linear(
        x.to(globs.shared_out.dtype).unsqueeze(0),
        globs.shared_expert_down_proj_weights[layer_idx],
    ).squeeze(0)
    globs.shared_out.copy_(shared.to(globs.shared_out.dtype))


def fused_router(
    globs: Globals, layer_idx: int, normed: Tensor | None = None
) -> None:
    """Router branch; optional precomputed normed hidden_states."""
    num_experts_per_tok = QwenMoeConfig.num_experts_per_tok
    if normed is None:
        normed = _rms_norm(
            globs.hidden_states,
            globs.mlp_norm_weights[layer_idx],
            eps=QwenMoeConfig.rms_norm_eps,
        )
    router_logits = F.linear(
        normed.float().unsqueeze(0),
        globs.router_weights[layer_idx].float(),
    ).squeeze(0)
    probs = F.softmax(router_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, num_experts_per_tok, dim=-1)
    globs.expert_indices.copy_(top_indices.to(globs.expert_indices.dtype))
    globs.expert_weights.copy_(top_probs.to(globs.expert_weights.dtype))


def fused_routed_expert_upgate(
    globs: Globals,
    layer_idx: int,
    expert_idx: int,
    normed: Tensor | None = None,
) -> None:
    """One routed expert up/gate branch; optional precomputed normed hidden_states."""
    mid = QwenMoeConfig.moe_intermediate_size
    if normed is None:
        normed = _rms_norm(
            globs.hidden_states,
            globs.mlp_norm_weights[layer_idx],
            eps=QwenMoeConfig.rms_norm_eps,
        )
    eid = globs.expert_indices[expert_idx].item()
    gate_up = F.linear(
        normed.unsqueeze(0),
        globs.expert_gate_up_proj_weights[layer_idx, eid],
    ).squeeze(0)
    gate = gate_up[:mid]
    up = gate_up[mid:]
    silu_out = F.silu(gate.to(torch.float32)) * up.to(torch.float32)
    globs.silu_out.copy_(silu_out.to(globs.silu_out.dtype))


def fused_routed_expert_downproj(
    globs: Globals,
    layer_idx: int,
    expert_idx: int,
    accum_f32: torch.Tensor | None = None,
) -> None:
    """One routed expert down branch multiplied by router weight."""
    eid = globs.expert_indices[expert_idx].item()
    w = globs.expert_weights[expert_idx]
    out = F.linear(
        globs.silu_out.unsqueeze(0),
        globs.expert_down_proj_weights[layer_idx, eid],
    ).squeeze(0)
    weighted = out.float() * w
    if accum_f32 is not None:
        accum_f32.add_(weighted)
    else:
        globs.routed_out.add_(weighted.to(globs.routed_out.dtype))


def fused_moe_block(globs: Globals, layer_idx: int) -> None:
    """Decode MoE block aligned with fused prefill and HF sparse block math."""
    num_experts_per_tok = QwenMoeConfig.num_experts_per_tok
    normed = _rms_norm(
        globs.hidden_states,
        globs.mlp_norm_weights[layer_idx],
        eps=QwenMoeConfig.rms_norm_eps,
    )
    fused_shared_mlp(globs, layer_idx, normed=normed)
    fused_router(globs, layer_idx, normed=normed)
    routed_out_f32 = torch.zeros(
        globs.routed_out.shape,
        dtype=torch.float32,
        device=globs.routed_out.device,
    )
    for expert_idx in range(num_experts_per_tok):
        fused_routed_expert_upgate(
            globs, layer_idx, expert_idx, normed=normed
        )
        fused_routed_expert_downproj(
            globs, layer_idx, expert_idx, accum_f32=routed_out_f32
        )
    globs.routed_out.copy_(routed_out_f32.to(globs.routed_out.dtype))
    gate_logit = (
        globs.shared_expert_gate_weights[layer_idx].squeeze(0).float()
        * normed.float()
    ).sum()
    gate_scale = torch.sigmoid(gate_logit).to(globs.hidden_states.dtype)
    globs.hidden_states.add_(globs.routed_out)
    globs.hidden_states.add_(globs.shared_out * gate_scale)


# -------- Prefill: full-sequence ops --------


def fused_prefill_rms_qkv_rope_write_cache(
    globs: Globals, layer_idx: int, seq_len: int
) -> None:
    """Prefill: RMS(hidden_states_seq[:seq_len]) -> QKV [S,H]@W -> RoPE -> write prefill_q_seq, k_cache, v_cache."""
    n_heads = QwenMoeConfig.num_attention_heads
    n_kv_heads = QwenMoeConfig.num_key_value_heads
    head_dim = QwenMoeConfig.head_dim
    eps = QwenMoeConfig.rms_norm_eps
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    hs = globs.hidden_states_seq[:seq_len]
    post_ln = _rms_norm(hs, globs.attn_norm_weights[layer_idx], eps=eps)
    qkv = F.linear(post_ln, globs.qkv_weights[layer_idx])
    if globs.qkv_bias is not None:
        qkv = qkv + globs.qkv_bias[layer_idx].to(qkv.dtype)
    q = qkv[:, :k_start].view(seq_len, n_heads, head_dim)
    k = qkv[:, k_start:v_start].view(seq_len, n_kv_heads, head_dim)
    v = qkv[:, v_start:].view(seq_len, n_kv_heads, head_dim)

    cos = globs.rope_cos[:seq_len]
    sin = globs.rope_sin[:seq_len]
    q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
        q, k, cos, sin, unsqueeze_dim=1
    )
    q_rope = q_rope.squeeze(0) if q_rope.dim() == 4 else q_rope
    k_rope = k_rope.squeeze(0) if k_rope.dim() == 4 else k_rope
    # Match megakernel/HF: cast rope outputs back to projection dtype (bf16/fp16).
    q_rope = q_rope.to(q.dtype)
    k_rope = k_rope.to(k.dtype)

    globs.prefill_q_seq[:seq_len].copy_(q_rope.to(globs.prefill_q_seq.dtype))
    kc, vc = globs.k_cache, globs.v_cache
    globs.k_cache[layer_idx, 0, :seq_len, :, :].copy_(k_rope.to(kc.dtype))
    globs.v_cache[layer_idx, 0, :seq_len, :, :].copy_(v.to(vc.dtype))


def fused_prefill_attention(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill attention aligned with megakernel/HF semantics.

    Use SDPA with grouped-query attention directly (enable_gqa=True) instead of
    manually repeating KV heads.
    """
    n_heads = QwenMoeConfig.num_attention_heads
    n_kv_heads = QwenMoeConfig.num_key_value_heads
    head_dim = QwenMoeConfig.head_dim

    q = globs.prefill_q_seq[:seq_len].float()
    # [S, H, D] -> [1, H, S, D]
    q_bhld = q.unsqueeze(0).transpose(1, 2)
    # cache [S, Hkv, D] -> [1, Hkv, S, D]
    k_bhld = (
        globs.k_cache[layer_idx, 0, :seq_len, :, :]
        .permute(1, 0, 2)
        .unsqueeze(0)
        .float()
    )
    v_bhld = (
        globs.v_cache[layer_idx, 0, :seq_len, :, :]
        .permute(1, 0, 2)
        .unsqueeze(0)
        .float()
    )
    try:
        attn_out = F.scaled_dot_product_attention(
            q_bhld, k_bhld, v_bhld, is_causal=True, enable_gqa=True
        )
    except TypeError:
        # Fallback for runtimes without enable_gqa support.
        gqa_ratio = n_heads // n_kv_heads
        k_exp = k_bhld.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).reshape(
            1, n_heads, seq_len, head_dim
        )
        v_exp = v_bhld.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).reshape(
            1, n_heads, seq_len, head_dim
        )
        attn_out = F.scaled_dot_product_attention(
            q_bhld, k_exp, v_exp, is_causal=True
        )
    globs.attn_out_seq[:seq_len].copy_(
        attn_out.transpose(1, 2)
        .squeeze(0)
        .reshape(seq_len, -1)
        .to(globs.attn_out_seq.dtype)
    )


def fused_prefill_oproj_residual(
    globs: Globals, layer_idx: int, seq_len: int
) -> None:
    """Prefill: attn_out_seq @ o_proj + residual -> hidden_states_seq."""
    o_proj = globs.o_proj_weights[layer_idx]
    attn_out = globs.attn_out_seq[:seq_len]
    if attn_out.dtype != o_proj.dtype:
        o_proj = o_proj.to(attn_out.dtype)
    out = F.linear(attn_out, o_proj)
    globs.hidden_states_seq[:seq_len] += out.to(globs.hidden_states_seq.dtype)


def fused_prefill_moe_block(
    globs: Globals,
    layer_idx: int,
    seq_len: int,
) -> None:
    """Prefill MoE aligned with megakernel/HF sparse block semantics.

    Vectorize over sequence, then dispatch by expert using index_add_ accumulation:
    - norm -> router softmax/topk
    - per-expert routed MLP on selected tokens
    - shared expert with sigmoid gate
    - residual add back to hidden_states_seq
    """
    num_experts_per_tok = QwenMoeConfig.num_experts_per_tok
    num_experts = QwenMoeConfig.num_experts
    moe_mid = QwenMoeConfig.moe_intermediate_size
    eps = QwenMoeConfig.rms_norm_eps

    hs = globs.hidden_states_seq[:seq_len]
    normed = _rms_norm(hs, globs.mlp_norm_weights[layer_idx], eps=eps)

    # Router in float32; cast routing weights back to hidden dtype (megakernel style)
    router_logits = F.linear(normed.float(), globs.router_weights[layer_idx].float())
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    top_probs, selected_experts = torch.topk(
        routing_weights, num_experts_per_tok, dim=-1
    )
    if getattr(QwenMoeConfig, "norm_topk_prob", False):
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    top_probs = top_probs.to(normed.dtype)

    routed_out_f32 = torch.zeros(
        normed.shape, dtype=torch.float32, device=normed.device
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().view(-1)

    for expert_idx in expert_hit.tolist():
        mask = expert_mask[expert_idx]
        idx, top_x = torch.where(mask)
        if top_x.numel() == 0:
            continue
        current_state = normed[top_x]
        gate_up = F.linear(
            current_state, globs.expert_gate_up_proj_weights[layer_idx, expert_idx]
        )
        gate = gate_up[:, :moe_mid]
        up = gate_up[:, moe_mid:]
        prod = F.silu(gate) * up
        current_hidden = F.linear(
            prod, globs.expert_down_proj_weights[layer_idx, expert_idx]
        )
        current_hidden = current_hidden * top_probs[top_x, idx].unsqueeze(-1)
        routed_out_f32.index_add_(0, top_x, current_hidden.float())

    # Shared expert branch.
    shared_gate = F.linear(normed, globs.shared_expert_gate_proj_weights[layer_idx])
    shared_up = F.linear(normed, globs.shared_expert_up_proj_weights[layer_idx])
    shared_prod = F.silu(shared_gate.to(torch.float32)) * shared_up.to(torch.float32)
    shared_out = F.linear(
        shared_prod.to(normed.dtype), globs.shared_expert_down_proj_weights[layer_idx]
    )
    gate_logits = F.linear(
        normed.float(), globs.shared_expert_gate_weights[layer_idx].float()
    ).squeeze(-1)
    gate_scales = torch.sigmoid(gate_logits).to(normed.dtype)
    shared_out = shared_out * gate_scales.unsqueeze(-1)

    # The official framework reference and MegaKernel compute residual + routed + shared.
    # Faster path: keep shared/residual accumulation in model dtype.
    # For stricter numerical parity, use the stable alternative:
    #   shared_out_f32 = shared_out.float() * gate_scales.unsqueeze(-1).float()
    #   hs_out = hs.float() + routed_out_f32 + shared_out_f32
    hs_out = hs + routed_out_f32.to(hs.dtype) + shared_out
    globs.hidden_states_seq[:seq_len].copy_(hs_out.to(globs.hidden_states_seq.dtype))

    # Publish the final prefill token into the decode-state buffers.
    globs.hidden_states.copy_(hs_out[-1].to(globs.hidden_states.dtype))
    globs.routed_out.copy_(routed_out_f32[-1].to(globs.routed_out.dtype))
    globs.shared_out.copy_(shared_out[-1].to(globs.shared_out.dtype))
    globs.expert_indices.copy_(selected_experts[-1].to(globs.expert_indices.dtype))
    globs.expert_weights.copy_(top_probs[-1].to(globs.expert_weights.dtype))
