"""
Fused decode operations for Qwen3-0.6B H100.

This mirrors the A100 Python fused reference split:
1. RMS(hidden_states) -> QKV GEMV -> qkv_raw
2. Q/K norm + RoPE + cache write
3. Attention
4. O-proj + residual
5. MLP
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from ..globs import Globals
from ..model_config import Qwen3Config
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


def _per_head_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    # Native Op2 keeps the RMS-normalized Q/K values and BF16 weight product in
    # FP32 through RoPE, then casts only at the Q/K cache stores.
    return weight.to(torch.float32) * x_normed


def fused_rms_qkvproj_gemv(globs: Globals, layer_idx: int) -> None:
    """RMS(hidden) -> QKV GEMV -> writes qkv_raw."""
    h = globs.hidden_states
    if h.dim() > 1:
        h = h.view(-1)
    post_ln = _rms_norm(h, globs.attn_norm_weights[layer_idx], eps=Qwen3Config.rms_norm_eps)
    qkv = einsum(globs.qkv_weights[layer_idx], post_ln, "o i, i -> o")
    globs.qkv_raw.copy_(qkv.to(globs.qkv_raw.dtype))


def fused_qknorm_rope_writecache(globs: Globals, layer_idx: int) -> None:
    """Read qkv_raw -> split Q/K/V -> Q/K norm -> RoPE -> write q_post_rope, k_cache, v_cache."""
    pos_id = int(globs.position_id) if globs.position_id is not None else 0
    eps = Qwen3Config.rms_norm_eps
    n_heads = Qwen3Config.num_attention_heads
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    # Runtime/native storage is [1, 1, 1, QKV_DIM], while some CPU fixtures use
    # a flat vector.  The serialized ABI is linear in either representation.
    qkv = globs.qkv_raw.reshape(-1)
    q = qkv[:k_start].view(n_heads, head_dim)
    k = qkv[k_start:v_start].view(n_kv_heads, head_dim)
    v = qkv[v_start:]

    q = _per_head_rms_norm(q, globs.q_norm_weights[layer_idx], eps=eps)
    k = _per_head_rms_norm(k, globs.k_norm_weights[layer_idx], eps=eps)

    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
        q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=0
    )

    globs.q_post_rope.copy_(q_rope.view(-1).to(globs.q_post_rope.dtype))
    kc = globs.k_cache
    vc = globs.v_cache
    k_slot = k_rope.reshape(n_kv_heads, head_dim).to(kc.dtype)
    v_slot = v.reshape(n_kv_heads, head_dim).to(vc.dtype)
    if kc.dim() == 5:
        globs.k_cache[layer_idx, 0, pos_id, :n_kv_heads, :].copy_(k_slot)
        globs.v_cache[layer_idx, 0, pos_id, :n_kv_heads, :].copy_(v_slot)
    else:
        globs.k_cache[layer_idx, pos_id, :n_kv_heads, :].copy_(k_slot)
        globs.v_cache[layer_idx, pos_id, :n_kv_heads, :].copy_(v_slot)


def fused_decode_attention(globs: Globals, layer_idx: int) -> None:
    """Full layer attention (split-kv then reduction)."""
    pos_id = globs.position_id or 0
    seq_len = pos_id + 1
    n_heads = Qwen3Config.num_attention_heads
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    gqa_ratio = n_heads // n_kv_heads
    kv_block_size = Qwen3Config.attn_kv_block_size
    reduction_size = Qwen3Config.attn_reduction_size
    device = globs.hidden_states.device
    sm_count = (
        torch.cuda.get_device_properties(device).multi_processor_count
        if device.type == "cuda"
        else 1
    )
    MAX_REDUCTION_LIST = 27
    computed_num_partials = min(
        MAX_REDUCTION_LIST,
        max(1, (sm_count + n_kv_heads - 1) // n_kv_heads),
    )
    if globs.attn_lse_intermediates is not None and globs.attn_lse_intermediates.dim() == 2:
        num_partials = globs.attn_lse_intermediates.shape[1]
    else:
        num_partials = computed_num_partials

    q = globs.q_post_rope.view(n_heads, head_dim).float()
    kc = globs.k_cache
    vc = globs.v_cache
    if kc.dim() == 5:
        k_full = kc[layer_idx, 0, :seq_len, :, :].float()
        v_full = vc[layer_idx, 0, :seq_len, :, :].float()
    else:
        k_full = kc[layer_idx, :seq_len, :, :].float()
        v_full = vc[layer_idx, :seq_len, :, :].float()
    if globs.attn_out is not None:
        globs.attn_out.zero_()
    if globs.attn_lse_intermediates is None or globs.attn_lse_intermediates.shape != (n_heads, num_partials):
        globs.attn_lse_intermediates = torch.zeros(
            (n_heads, num_partials), device=device, dtype=torch.float32
        )
    if globs.attn_out_intermediates is None or globs.attn_out_intermediates.shape != (n_heads, num_partials, head_dim):
        globs.attn_out_intermediates = torch.zeros(
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
        lses = globs.attn_lse_intermediates[head_start:head_end, :]
        outs = globs.attn_out_intermediates[head_start:head_end, :, :]
        lses_safe = torch.where(
            torch.isposinf(lses),
            torch.full_like(lses, 1e10),
            lses,
        )
        max_lse = torch.max(lses_safe, dim=1, keepdim=True).values
        max_lse_safe = torch.clamp(max_lse, min=-1e10)
        adjusted = (lses_safe - max_lse_safe).exp2()
        denom = adjusted.sum(dim=1, keepdim=True).clamp(min=1e-12)
        reduced = (outs * adjusted.unsqueeze(-1)).sum(dim=1) / denom
        globs.attn_out.view(n_heads, head_dim)[head_start:head_end, :] = reduced.to(globs.attn_out.dtype)


def fused_oproj_gemv_residual(globs: Globals, layer_idx: int) -> None:
    """attn_out @ o_proj (GEMV) + residual -> hidden_states."""
    attn = globs.attn_out.view(-1) if globs.attn_out.dim() > 1 else globs.attn_out
    out = einsum(globs.o_proj_weights[layer_idx], attn, "o i, i -> o")
    globs.hidden_states.view(-1).add_(out.to(globs.hidden_states.dtype))


def fused_mlp_upgate(globs: Globals, layer_idx: int) -> None:
    """MLP first half: RMSNorm -> up/gate proj -> SiLU(gate)*up -> silu_out."""
    h = globs.hidden_states.view(-1) if globs.hidden_states.dim() > 1 else globs.hidden_states
    post_ln = _rms_norm(
        h,
        globs.mlp_norm_weights[layer_idx],
        eps=Qwen3Config.rms_norm_eps,
    )
    up = einsum(globs.up_proj_weights[layer_idx], post_ln, "o i, i -> o")
    gate = einsum(globs.gate_proj_weights[layer_idx], post_ln, "o i, i -> o")
    silu_out = F.silu(gate) * up
    globs.silu_out.view(-1).copy_(silu_out.to(globs.silu_out.dtype))


def fused_mlp_down_residual(globs: Globals, layer_idx: int) -> None:
    """MLP second half: down_proj(silu_out) + residual -> hidden_states."""
    silu = globs.silu_out.view(-1) if globs.silu_out.dim() > 1 else globs.silu_out
    down = einsum(globs.down_proj_weights[layer_idx], silu, "o i, i -> o")
    globs.hidden_states.view(-1).add_(down.to(globs.hidden_states.dtype))


def fused_mlp(globs: Globals, layer_idx: int) -> None:
    """Full MLP: upgate then down+residual."""
    fused_mlp_upgate(globs, layer_idx)
    fused_mlp_down_residual(globs, layer_idx)


def fused_rms_lmhead_gemv(globs: Globals) -> None:
    """Final: RMSNorm(hidden_states) -> lm_head GEMV -> logits."""
    h = globs.hidden_states.view(-1) if globs.hidden_states.dim() > 1 else globs.hidden_states
    post_ln = _rms_norm(
        h,
        globs.lm_head_norm_weights,
        eps=Qwen3Config.rms_norm_eps,
    )
    logits_out = einsum(globs.lm_head_weights, post_ln, "o i, i -> o").to(globs.logits.dtype)
    globs.logits.view(-1).copy_(logits_out)


def fused_prefill_rms_qkv_gemv(
    globs: Globals, layer_idx: int, seq_len: int
) -> None:
    """Prefill: RMS(hidden_states_seq) -> QKV GEMV -> prefill_qkv_raw."""
    hs = globs.hidden_states_seq[:seq_len]
    post_ln = _rms_norm(
        hs,
        globs.attn_norm_weights[layer_idx],
        eps=Qwen3Config.rms_norm_eps,
    )
    qkv = einsum(globs.qkv_weights[layer_idx], post_ln, "o i, s i -> s o")
    globs.prefill_qkv_raw[:seq_len].copy_(qkv.to(globs.prefill_qkv_raw.dtype))


def fused_prefill_qknorm_rope_writecache(
    globs: Globals, layer_idx: int, seq_len: int
) -> None:
    """Prefill: prefill_qkv_raw -> Q/K norm + RoPE -> Q/K/V cache outputs."""
    n_heads = Qwen3Config.num_attention_heads
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    eps = Qwen3Config.rms_norm_eps
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    qkv = globs.prefill_qkv_raw[:seq_len]
    q = qkv[:, :k_start].view(seq_len, n_heads, head_dim)
    k = qkv[:, k_start:v_start].view(seq_len, n_kv_heads, head_dim)
    v = qkv[:, v_start:].view(seq_len, n_kv_heads, head_dim)

    q_norm_w = globs.q_norm_weights[layer_idx]
    k_norm_w = globs.k_norm_weights[layer_idx]
    q = _per_head_rms_norm(
        q.reshape(seq_len * n_heads, head_dim), q_norm_w, eps=eps
    ).view(seq_len, n_heads, head_dim)
    k = _per_head_rms_norm(
        k.reshape(seq_len * n_kv_heads, head_dim), k_norm_w, eps=eps
    ).view(seq_len, n_kv_heads, head_dim)

    cos = globs.rope_cos[:seq_len]
    sin = globs.rope_sin[:seq_len]
    if cos.dim() == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_rope, k_rope = _apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=0)
    q_rope = q_rope.squeeze(0) if q_rope.dim() == 4 else q_rope
    k_rope = k_rope.squeeze(0) if k_rope.dim() == 4 else k_rope

    globs.prefill_q_seq[:seq_len].copy_(q_rope.to(globs.prefill_q_seq.dtype))

    kc = globs.k_cache
    vc = globs.v_cache
    if kc.dim() == 5:
        globs.k_cache[layer_idx, 0, :seq_len, :, :].copy_(k_rope.to(kc.dtype))
        globs.v_cache[layer_idx, 0, :seq_len, :, :].copy_(v.to(vc.dtype))
    else:
        globs.k_cache[layer_idx, :seq_len, :, :].copy_(k_rope.to(kc.dtype))
        globs.v_cache[layer_idx, :seq_len, :, :].copy_(v.to(vc.dtype))


def fused_prefill_attention(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: causal SDPA over full sequence."""
    n_heads = Qwen3Config.num_attention_heads
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    gqa_ratio = n_heads // n_kv_heads

    q = globs.prefill_q_seq[:seq_len].float()
    kc = globs.k_cache
    vc = globs.v_cache
    if kc.dim() == 5:
        k = kc[layer_idx, 0, :seq_len, :, :].float()
        v = vc[layer_idx, 0, :seq_len, :, :].float()
    else:
        k = kc[layer_idx, :seq_len, :, :].float()
        v = vc[layer_idx, :seq_len, :, :].float()

    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    q_bhld = q.unsqueeze(0).transpose(1, 2)
    k_bhld = k.unsqueeze(0).transpose(1, 2)
    v_bhld = v.unsqueeze(0).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld, is_causal=True)
    globs.attn_out_seq[:seq_len].copy_(
        attn_out.transpose(1, 2).squeeze(0).reshape(seq_len, -1).to(globs.attn_out_seq.dtype)
    )


def fused_prefill_oproj_residual(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: o_proj + residual over full sequence."""
    out = einsum(
        globs.o_proj_weights[layer_idx],
        globs.attn_out_seq[:seq_len],
        "o i, s i -> s o",
    )
    globs.hidden_states_seq[:seq_len] += out.to(globs.hidden_states_seq.dtype)


def fused_prefill_mlp(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: RMS -> up/gate -> SiLU*up -> down + residual over full sequence."""
    hs = globs.hidden_states_seq[:seq_len]
    post_ln = _rms_norm(hs, globs.mlp_norm_weights[layer_idx], eps=Qwen3Config.rms_norm_eps)
    up = einsum(globs.up_proj_weights[layer_idx], post_ln, "o i, s i -> s o")
    gate = einsum(globs.gate_proj_weights[layer_idx], post_ln, "o i, s i -> s o")
    silu_out = F.silu(gate) * up
    down = einsum(globs.down_proj_weights[layer_idx], silu_out, "o i, s i -> s o")
    globs.hidden_states_seq[:seq_len] += down.to(globs.hidden_states_seq.dtype)
