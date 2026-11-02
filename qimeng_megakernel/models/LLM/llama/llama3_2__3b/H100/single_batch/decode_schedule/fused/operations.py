"""
Fused decode operations: full-layer (no tiling) versions of the same math as python_vm.

Used for two-level verification:
  1. HF -> fused: compare HF forward vs these fused ops.
  2. Fused -> instruction schedule: compare fused result vs pyvm (tiled) result.

Each fused_* function takes (globs, layer_idx) and performs one logical op for one layer
in one shot (full matvec, full attention, etc.), matching the semantics of the
instruction handlers in python_vm.py but without tiling.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from ..globs import Globals
from ..model_config import (
    ATTENTION_REDUCTION_LIST_CAPACITY,
    H100_SM_COUNT,
    LlamaConfig,
    attention_num_partials,
)


def _rms_norm(inp: Tensor, weight: Tensor, eps: float) -> Tensor:
    input_dtype = inp.dtype
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


def _apply_rotary_pos_emb_interleaved(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def fused_rms_qkvproj_gemv_rope(globs: Globals, layer_idx: int) -> None:
    """Full layer: RMS(hidden_states) -> QKV projection GEMV -> RoPE on q,k -> write q_post_rope, k_cache, v_cache."""
    pos_id = globs.position_id
    eps = LlamaConfig.rms_norm_eps
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    post_ln = _rms_norm(
        globs.hidden_states,
        globs.attn_norm_weights[layer_idx],
        eps=eps,
    )
    qkv = einsum(globs.qkv_weights[layer_idx], post_ln, "o i, i -> o")
    q = qkv[:k_start].view(1, n_heads, head_dim)
    k = qkv[k_start:v_start].view(1, n_kv_heads, head_dim)
    v = qkv[v_start:].view(1, n_kv_heads, head_dim)

    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    q_rope, k_rope = _apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=0)

    globs.q_post_rope.copy_(q_rope.view(-1).to(globs.q_post_rope.dtype))
    # k/v_cache layout [L, B, S, H, D] (from preprocess; HF was converted from (B,H,S,D) to (B,S,H,D)).
    n_h = globs.k_cache.shape[3]
    n_d = globs.k_cache.shape[4]
    assert k_rope.numel() == n_h * n_d, (
        f"k_rope size {k_rope.numel()} != cache slot {n_h * n_d} (n_heads={n_h}, head_dim={n_d}); "
        "ensure preprocess converted HF cache to [L,B,S,H,D] and model config matches."
    )
    k_slot = k_rope.reshape(n_h, n_d).to(globs.k_cache.dtype)
    globs.k_cache[layer_idx, 0, pos_id, :, :].copy_(k_slot)
    globs.v_cache[layer_idx, 0, pos_id, :, :].copy_(
        v.reshape(n_h, n_d).to(globs.v_cache.dtype)
    )


def fused_decode_attention(globs: Globals, layer_idx: int) -> None:
    """
    Full layer attention using per-Q-head split-kv then reduction (same structure as PyVM/kernel).
    Fills attn_lse_intermediates and attn_out_intermediates per (head, partial), then reduces to attn_out.
    """
    pos_id = globs.position_id or 0
    seq_len = pos_id + 1
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads
    kv_block_size = LlamaConfig.attn_kv_block_size
    reduction_size = LlamaConfig.attn_reduction_size
    device = globs.hidden_states.device
    detected_sm_count = (
        torch.cuda.get_device_properties(device).multi_processor_count
        if device.type == "cuda"
        else 1
    )
    schedule_sm_count = min(detected_sm_count, H100_SM_COUNT)
    num_partials = attention_num_partials(
        schedule_sm_count,
        n_heads,
        reduction_list_capacity=ATTENTION_REDUCTION_LIST_CAPACITY,
    )

    q = globs.q_post_rope.view(n_heads, head_dim).float()
    k_full = globs.k_cache[layer_idx, 0, :seq_len, :, :].float()
    v_full = globs.v_cache[layer_idx, 0, :seq_len, :, :].float()
    if (
        globs.attn_lse_intermediates is None
        or globs.attn_lse_intermediates.dim() != 2
        or globs.attn_lse_intermediates.shape[0] != n_heads
        or globs.attn_lse_intermediates.shape[1] < num_partials
    ):
        globs.attn_lse_intermediates = torch.empty(
            (n_heads, num_partials), device=device, dtype=torch.float32
        )
    if (
        globs.attn_out_intermediates is None
        or globs.attn_out_intermediates.dim() != 3
        or globs.attn_out_intermediates.shape[0] != n_heads
        or globs.attn_out_intermediates.shape[1] < num_partials
        or globs.attn_out_intermediates.shape[2] != head_dim
    ):
        globs.attn_out_intermediates = torch.empty(
            (n_heads, num_partials, head_dim), device=device, dtype=torch.float32
        )
    scale = 1.0 / math.sqrt(head_dim)
    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / num_partials)

    for attention_head_idx in range(n_heads):
        kv_head_idx = attention_head_idx // gqa_ratio
        k = k_full[:, kv_head_idx, :]
        v = v_full[:, kv_head_idx, :]
        q_h = q[attention_head_idx : attention_head_idx + 1]
        for partial_idx in range(num_partials):
            start_block = partial_idx * blocks_per_partial
            end_block = min(start_block + blocks_per_partial, total_blocks)
            start_token = start_block * kv_block_size
            end_token = min(end_block * kv_block_size, seq_len)
            if start_token >= end_token:
                globs.attn_lse_intermediates[attention_head_idx, partial_idx] = -float("inf")
                globs.attn_out_intermediates[attention_head_idx, partial_idx, :] = 0.0
                continue
            k_slice = k[start_token:end_token]
            v_slice = v[start_token:end_token]
            qk = einsum(q_h, k_slice, "h i, k i -> h k")
            scaled_qk = qk * scale
            softmax = torch.softmax(scaled_qk, dim=-1)
            lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1) + 1e-12)
            out = einsum(softmax, v_slice, "h k, k o -> h o")
            globs.attn_lse_intermediates[attention_head_idx, partial_idx] = lse.squeeze(0).item()
            globs.attn_out_intermediates[attention_head_idx, partial_idx, :] = out.squeeze(0)

    for head_start in range(0, n_heads, reduction_size):
        head_end = min(head_start + reduction_size, n_heads)
        lses = globs.attn_lse_intermediates[
            head_start:head_end, :num_partials
        ]
        outs = globs.attn_out_intermediates[
            head_start:head_end, :num_partials, :
        ]
        max_lse = torch.max(lses, dim=1, keepdim=True).values
        adjusted = (lses - max_lse).exp2()
        denom = adjusted.sum(dim=1, keepdim=True)
        reduced = (outs * adjusted.unsqueeze(-1)).sum(dim=1) / denom
        globs.attn_out.view(n_heads, head_dim)[head_start:head_end, :] = reduced.to(globs.attn_out.dtype)


def fused_oproj_gemv_residual(globs: Globals, layer_idx: int) -> None:
    """Full layer: attn_out @ o_proj (GEMV) + residual -> hidden_states."""
    out = einsum(globs.o_proj_weights[layer_idx], globs.attn_out, "o i, i -> o")
    globs.hidden_states += out.to(globs.hidden_states.dtype)


def fused_mlp_upgate(globs: Globals, layer_idx: int) -> None:
    """MLP first half: RMSNorm(hidden_states) -> up/gate proj -> SiLU(gate)*up -> silu_out. Does not modify hidden_states."""
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.mlp_norm_weights[layer_idx],
        eps=LlamaConfig.rms_norm_eps,
    )
    up = einsum(globs.up_proj_weights[layer_idx], post_ln, "o i, i -> o")
    gate = einsum(globs.gate_proj_weights[layer_idx], post_ln, "o i, i -> o")
    silu_out = F.silu(gate) * up
    globs.silu_out.copy_(silu_out.to(globs.silu_out.dtype))


def fused_mlp_down_residual(globs: Globals, layer_idx: int) -> None:
    """MLP second half: down_proj(silu_out) + residual -> hidden_states. Reads globs.silu_out."""
    down = einsum(globs.down_proj_weights[layer_idx], globs.silu_out, "o i, i -> o")
    globs.hidden_states += down.to(globs.hidden_states.dtype)


def fused_mlp(globs: Globals, layer_idx: int) -> None:
    """Full MLP: upgate then down+residual (same as calling fused_mlp_upgate + fused_mlp_down_residual)."""
    fused_mlp_upgate(globs, layer_idx)
    fused_mlp_down_residual(globs, layer_idx)


def fused_rms_lmhead_gemv(globs: Globals) -> None:
    """Final: RMSNorm(hidden_states) -> lm_head GEMV -> logits."""
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.lm_head_norm_weights,
        eps=LlamaConfig.rms_norm_eps,
    )
    globs.logits.copy_(einsum(globs.lm_head_weights, post_ln, "o i, i -> o").to(globs.logits.dtype))


# -------- Prefill: full-sequence ops (one forward per layer, H100 layout) --------

def fused_prefill_rms_qkv_rope_write_cache(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: RMS(hidden_states_seq) -> QKV [S,H]@W -> RoPE on Q,K -> write K,V to cache, Q to prefill_q_seq."""
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    eps = LlamaConfig.rms_norm_eps
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    hs = globs.hidden_states_seq  # [S, H]
    post_ln = _rms_norm(hs, globs.attn_norm_weights[layer_idx], eps=eps)
    qkv = einsum(globs.qkv_weights[layer_idx], post_ln, "o i, s i -> s o")  # [S, qkv_dim]
    q = qkv[:, :k_start].view(seq_len, n_heads, head_dim)
    k = qkv[:, k_start:v_start].view(seq_len, n_kv_heads, head_dim)
    v = qkv[:, v_start:].view(seq_len, n_kv_heads, head_dim)

    cos = globs.rope_cos[:seq_len]
    sin = globs.rope_sin[:seq_len]
    if cos.dim() == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_rope = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_rope = (k * cos) + (_rotate_half_interleaved(k) * sin)
    globs.prefill_q_seq.copy_(q_rope.to(globs.prefill_q_seq.dtype))
    globs.k_cache[layer_idx, 0, :seq_len, :, :].copy_(k_rope.to(globs.k_cache.dtype))
    globs.v_cache[layer_idx, 0, :seq_len, :, :].copy_(v.to(globs.v_cache.dtype))


def fused_prefill_attention(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: one causal SDPA over full Q, K, V (from cache); write attn_out_seq [S, n_heads*d]."""
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads

    q = globs.prefill_q_seq[:seq_len].float()
    k = globs.k_cache[layer_idx, 0, :seq_len, :, :].float()
    v = globs.v_cache[layer_idx, 0, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    q_bhld = q.unsqueeze(0).transpose(1, 2)
    k_bhld = k.unsqueeze(0).transpose(1, 2)
    v_bhld = v.unsqueeze(0).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(
        q_bhld, k_bhld, v_bhld, is_causal=True
    )
    globs.attn_out_seq[:seq_len].copy_(
        attn_out.transpose(1, 2).squeeze(0).reshape(seq_len, -1).to(globs.attn_out_seq.dtype)
    )


def fused_prefill_oproj_residual(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: attn_out_seq @ o_proj + residual -> hidden_states_seq."""
    out = einsum(
        globs.o_proj_weights[layer_idx],
        globs.attn_out_seq[:seq_len],
        "o i, s i -> s o",
    )
    globs.hidden_states_seq[:seq_len] += out.to(globs.hidden_states_seq.dtype)


def fused_prefill_mlp(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: RMS -> up/gate -> SiLU*gate*up -> down -> residual over full sequence."""
    hs = globs.hidden_states_seq[:seq_len]
    post_ln = _rms_norm(hs, globs.mlp_norm_weights[layer_idx], eps=LlamaConfig.rms_norm_eps)
    up = einsum(globs.up_proj_weights[layer_idx], post_ln, "o i, s i -> s o")
    gate = einsum(globs.gate_proj_weights[layer_idx], post_ln, "o i, s i -> s o")
    silu_out = F.silu(gate) * up
    down = einsum(globs.down_proj_weights[layer_idx], silu_out, "o i, s i -> s o")
    globs.hidden_states_seq[:seq_len] += down.to(globs.hidden_states_seq.dtype)
