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
from ..model_config import LlamaConfig


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
    w = layer_idx
    pos_id = globs.position_id
    eps = LlamaConfig.rms_norm_eps
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    post_ln = _rms_norm(
        globs.hidden_states,
        globs.attn_norm_weights[w],
        eps=eps,
    )
    qkv = einsum(globs.qkv_weights[w], post_ln, "o i, i -> o")
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
    """Full layer: single-head attention over current q_post_rope and k/v cache; write attn_out."""
    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads

    q = globs.q_post_rope.view(n_heads, head_dim).float()
    k = globs.k_cache[layer_idx, 0, :seq_len, :, :].float()
    v = globs.v_cache[layer_idx, 0, :seq_len, :, :].float()
    # k, v: (seq_len, n_kv_heads, head_dim) -> repeat to (seq_len, n_heads, head_dim)
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    globs.attn_out.copy_(out.reshape(-1).to(globs.attn_out.dtype))


def fused_oproj_gemv_residual(globs: Globals, layer_idx: int) -> None:
    """Full layer: attn_out @ o_proj (GEMV) + residual -> hidden_states."""
    w = layer_idx
    out = einsum(globs.o_proj_weights[w], globs.attn_out, "o i, i -> o")
    globs.hidden_states += out.to(globs.hidden_states.dtype)


def fused_mlp_upgate(globs: Globals, layer_idx: int) -> None:
    """MLP first half: RMSNorm(hidden_states) -> up/gate proj -> SiLU(gate)*up -> silu_out. Does not modify hidden_states."""
    w = layer_idx
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.mlp_norm_weights[w],
        eps=LlamaConfig.rms_norm_eps,
    )
    up = einsum(globs.up_proj_weights[w], post_ln, "o i, i -> o")
    gate = einsum(globs.gate_proj_weights[w], post_ln, "o i, i -> o")
    silu_out = F.silu(gate) * up
    globs.silu_out.copy_(silu_out.to(globs.silu_out.dtype))


def fused_mlp_down_residual(globs: Globals, layer_idx: int) -> None:
    """MLP second half: down_proj(silu_out) + residual -> hidden_states. Reads globs.silu_out."""
    w = layer_idx
    down = einsum(globs.down_proj_weights[w], globs.silu_out, "o i, i -> o")
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


# -------- Prefill: full-sequence ops (one forward per layer) --------

def fused_prefill_rms_qkv_rope_write_cache(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: RMS(hidden_states_seq) -> QKV [S,H]@W -> RoPE on Q,K -> write K,V to cache, Q to prefill_q_seq."""
    w = layer_idx
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    eps = LlamaConfig.rms_norm_eps
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    hs = globs.hidden_states_seq  # [S, H]
    post_ln = _rms_norm(hs, globs.attn_norm_weights[w], eps=eps)
    qkv = einsum(globs.qkv_weights[w], post_ln, "o i, s i -> s o")  # [S, qkv_dim]
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
    w = layer_idx
    out = einsum(
        globs.o_proj_weights[w],
        globs.attn_out_seq[:seq_len],
        "o i, s i -> s o",
    )
    globs.hidden_states_seq[:seq_len] += out.to(globs.hidden_states_seq.dtype)


def fused_prefill_mlp(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: RMS -> up/gate -> SiLU*gate*up -> down -> residual over full sequence."""
    w = layer_idx
    hs = globs.hidden_states_seq[:seq_len]
    post_ln = _rms_norm(hs, globs.mlp_norm_weights[w], eps=LlamaConfig.rms_norm_eps)
    up = einsum(globs.up_proj_weights[w], post_ln, "o i, s i -> s o")
    gate = einsum(globs.gate_proj_weights[w], post_ln, "o i, s i -> s o")
    silu_out = F.silu(gate) * up
    down = einsum(globs.down_proj_weights[w], silu_out, "o i, s i -> s o")
    globs.hidden_states_seq[:seq_len] += down.to(globs.hidden_states_seq.dtype)
