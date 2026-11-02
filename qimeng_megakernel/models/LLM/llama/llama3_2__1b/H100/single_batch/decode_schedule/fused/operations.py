"""
Fused decode operations: full-layer (no tiling) versions of the same math as python_vm.

Used for two-level verification:
  1. HF -> fused: compare HF forward vs these fused ops.
  2. Fused -> instruction schedule: compare fused result vs pyvm (tiled) result.

Each fused_* function takes (globs, layer_idx) and performs one logical op for one layer
in one shot (full matvec, full attention, etc.), matching the semantics of the
instruction handlers in python_vm.py but without tiling.

Multi-batch (decode): globs.hidden_states is [B, H] when B>1, else [H] (python_vm-compatible).
Prefill: hidden_states_seq is [B, S, H] when B>1, else [S, H].

Branch reduction: compute on a logical [B, ...] view (unsqueeze when storage is 1D / [S,H]),
then write back with small ndim checks — storage layout for B=1 is unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from qimeng_megakernel.models.LLM.decode_state import (
    decode_pos_id,
    row_active,
)

from ..globs import Globals
from ..model_config import LlamaConfig


def _decode_batch_n(globs: Globals) -> int:
    h = globs.hidden_states
    if h is None or h.ndim == 1:
        return 1
    return int(h.shape[0])


def _prefill_batch_n(globs: Globals) -> int:
    hs = globs.hidden_states_seq
    if hs is None or hs.ndim == 2:
        return 1
    return int(hs.shape[0])


def _as_decode_BH(x: Tensor) -> Tensor:
    """[H] -> [1,H]; [B,H] unchanged."""
    return x.unsqueeze(0) if x.ndim == 1 else x


def _decode_write_bh_flat(dst: Tensor, src_bh: Tensor) -> None:
    """Write [B, Hflat] to dst ([Hflat] or [B, Hflat])."""
    if dst.ndim == 1:
        dst.copy_(src_bh.reshape(-1).to(dst.dtype))
    else:
        dst.copy_(src_bh.reshape(dst.shape[0], -1).to(dst.dtype))


def _decode_add_residual(hidden: Tensor, delta_bh: Tensor) -> None:
    """delta_bh is [B, H]; hidden is [H] or [B, H]."""
    if hidden.ndim == 1:
        hidden.add_(delta_bh.squeeze(0).to(hidden.dtype))
    else:
        hidden.add_(delta_bh.to(hidden.dtype))


def _as_prefill_BSH(globs: Globals, seq_len: int) -> Tensor:
    """[S,H] -> [1,S,H]; [B,S,H] -> [:seq_len] slice keeps [B,seq_len,H]."""
    hs = globs.hidden_states_seq
    if hs.ndim == 2:
        return hs[:seq_len].unsqueeze(0)
    return hs[:, :seq_len, :]


def _prefill_rope_cos_sin(globs: Globals, seq_len: int) -> tuple[Tensor, Tensor]:
    """Broadcast-friendly cos/sin for q,k shape [B, S, n_heads, head_dim]."""
    cos = globs.rope_cos[:seq_len]
    sin = globs.rope_sin[:seq_len]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)
    return cos, sin


def _prefill_write_q_seq(globs: Globals, seq_len: int, q_rope: Tensor) -> None:
    """q_rope: [B, S, n_h, d]. prefill_q_seq: [S,...] or [B,S,...]."""
    pq = globs.prefill_q_seq
    if pq.ndim == 3:
        pq[:seq_len].copy_(q_rope[0].to(pq.dtype))
    else:
        pq[:, :seq_len].copy_(q_rope.to(pq.dtype))


def _prefill_attn_q_slice(globs: Globals, seq_len: int, b: int) -> Tensor:
    pq = globs.prefill_q_seq
    if pq.ndim == 3:
        return pq[:seq_len]
    return pq[b, :seq_len]


def _prefill_write_attn_out_seq(
    globs: Globals, seq_len: int, b: int, out_flat: Tensor
) -> None:
    a = globs.attn_out_seq
    row = out_flat.to(a.dtype)
    if a.ndim == 2:
        a[:seq_len].copy_(row)
    else:
        a[b, :seq_len].copy_(row)


def _prefill_add_hs_residual(globs: Globals, seq_len: int, delta: Tensor) -> None:
    """delta: [B, S, H]; hidden_states_seq: [S,H] or [B,S,H]."""
    hs = globs.hidden_states_seq
    if hs.ndim == 2:
        hs[:seq_len] += delta[0].to(hs.dtype)
    else:
        hs[:, :seq_len] += delta.to(hs.dtype)


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


def _apply_rotary_pos_emb_interleaved(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def fused_rms_qkvproj_gemv_rope(globs: Globals, layer_idx: int) -> None:
    """Full layer: RMS(hidden_states) -> QKV projection GEMV -> RoPE on q,k -> write q_post_rope, k_cache, v_cache."""
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
    post_b = _as_decode_BH(post_ln)
    B = post_b.shape[0]
    W = globs.qkv_weights[layer_idx]
    qkv = einsum(W, post_b, "o i, b i -> b o")
    q = qkv[:, :k_start].view(B, n_heads, head_dim)
    k = qkv[:, k_start:v_start].view(B, n_kv_heads, head_dim)
    v = qkv[:, v_start:].view(B, n_kv_heads, head_dim)

    if getattr(globs, "position_ids", None) is None:
        pos_id = decode_pos_id(globs, 0)
        cos = globs.rope_cos[pos_id]
        sin = globs.rope_sin[pos_id]
        cos_b = cos.view(1, 1, -1)
        sin_b = sin.view(1, 1, -1)
        q_rope = (q * cos_b) + (_rotate_half_interleaved(q) * sin_b)
        k_rope = (k * cos_b) + (_rotate_half_interleaved(k) * sin_b)
        if not row_active(globs, 0):
            return
    else:
        q_rope = torch.empty_like(q)
        k_rope = torch.empty_like(k)
        for b in range(B):
            pos_id_b = decode_pos_id(globs, b)
            cos = globs.rope_cos[pos_id_b]
            sin = globs.rope_sin[pos_id_b]
            cos_b = cos.view(1, 1, -1)
            sin_b = sin.view(1, 1, -1)
            qb = q[b : b + 1]
            kb = k[b : b + 1]
            q_rope[b : b + 1] = (qb * cos_b) + (_rotate_half_interleaved(qb) * sin_b)
            k_rope[b : b + 1] = (kb * cos_b) + (_rotate_half_interleaved(kb) * sin_b)
    if globs.q_post_rope.ndim == 1:
        if row_active(globs, 0):
            _decode_write_bh_flat(globs.q_post_rope, q_rope)
    else:
        for b in range(B):
            if row_active(globs, b):
                globs.q_post_rope[b].copy_(
                    q_rope[b].reshape(-1).to(globs.q_post_rope.dtype)
                )

    n_h = globs.k_cache.shape[3]
    n_d = globs.k_cache.shape[4]
    for b in range(B):
        if not row_active(globs, b):
            continue
        pos_id_b = decode_pos_id(globs, b)
        globs.k_cache[layer_idx, b, pos_id_b, :, :].copy_(
            k_rope[b].reshape(n_h, n_d).to(globs.k_cache.dtype)
        )
        globs.v_cache[layer_idx, b, pos_id_b, :, :].copy_(
            v[b].reshape(n_h, n_d).to(globs.v_cache.dtype)
        )


def fused_decode_attention(globs: Globals, layer_idx: int) -> None:
    """Full layer: single-head attention over current q_post_rope and k/v cache; write attn_out."""
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads

    B = _decode_batch_n(globs)
    for b in range(B):
        if not row_active(globs, b):
            if globs.attn_out.ndim > 1:
                globs.attn_out[b].zero_()
            else:
                globs.attn_out.zero_()
            continue
        seq_len = decode_pos_id(globs, b) + 1
        q_src = (
            globs.q_post_rope if globs.q_post_rope.ndim == 1 else globs.q_post_rope[b]
        )
        q = q_src.view(n_heads, head_dim).float()
        k = globs.k_cache[layer_idx, b, :seq_len, :, :].float()
        v = globs.v_cache[layer_idx, b, :seq_len, :, :].float()
        k = (
            k.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1)
            .reshape(seq_len, n_heads, head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1)
            .reshape(seq_len, n_heads, head_dim)
        )
        qk = einsum(q, k, "h i, s h i -> h s")
        scaled_qk = qk * (1 / math.sqrt(head_dim))
        softmax = torch.softmax(scaled_qk, dim=-1)
        out = einsum(softmax, v, "h s, s h o -> h o")
        out_flat = out.reshape(-1).to(globs.attn_out.dtype)
        if globs.attn_out.ndim == 1:
            globs.attn_out.copy_(out_flat)
        else:
            globs.attn_out[b].copy_(out_flat)


def fused_oproj_gemv_residual(globs: Globals, layer_idx: int) -> None:
    """Full layer: attn_out @ o_proj (GEMV) + residual -> hidden_states."""
    attn_b = _as_decode_BH(globs.attn_out)
    W = globs.o_proj_weights[layer_idx]
    out = einsum(W, attn_b, "o i, b i -> b o")
    if out.dim() == 2:
        for b in range(out.shape[0]):
            if not row_active(globs, b):
                out[b].zero_()
    elif not row_active(globs, 0):
        out.zero_()
    _decode_add_residual(globs.hidden_states, out)


def fused_mlp_upgate(globs: Globals, layer_idx: int) -> None:
    """MLP first half: RMSNorm(hidden_states) -> up/gate proj -> SiLU(gate)*up -> silu_out."""
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.mlp_norm_weights[layer_idx],
        eps=LlamaConfig.rms_norm_eps,
    )
    post_b = _as_decode_BH(post_ln)
    up = einsum(globs.up_proj_weights[layer_idx], post_b, "o i, b i -> b o")
    gate = einsum(globs.gate_proj_weights[layer_idx], post_b, "o i, b i -> b o")
    silu_out = F.silu(gate) * up
    if silu_out.dim() == 2:
        for b in range(silu_out.shape[0]):
            if not row_active(globs, b):
                silu_out[b].zero_()
    elif not row_active(globs, 0):
        silu_out.zero_()
    _decode_write_bh_flat(globs.silu_out, silu_out)


def fused_mlp_down_residual(globs: Globals, layer_idx: int) -> None:
    """MLP second half: down_proj(silu_out) + residual -> hidden_states."""
    silu_b = _as_decode_BH(globs.silu_out)
    down = einsum(globs.down_proj_weights[layer_idx], silu_b, "o i, b i -> b o")
    if down.dim() == 2:
        for b in range(down.shape[0]):
            if not row_active(globs, b):
                down[b].zero_()
    elif not row_active(globs, 0):
        down.zero_()
    _decode_add_residual(globs.hidden_states, down)


def fused_mlp(globs: Globals, layer_idx: int) -> None:
    fused_mlp_upgate(globs, layer_idx)
    fused_mlp_down_residual(globs, layer_idx)


def fused_rms_lmhead_gemv(globs: Globals) -> None:
    """Final: RMSNorm(hidden_states) -> lm_head GEMV -> logits."""
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.lm_head_norm_weights,
        eps=LlamaConfig.rms_norm_eps,
    )
    post_b = _as_decode_BH(post_ln)
    logits_bh = einsum(globs.lm_head_weights, post_b, "o i, b i -> b o").to(
        globs.logits.dtype
    )
    if logits_bh.dim() == 2:
        for b in range(logits_bh.shape[0]):
            if not row_active(globs, b):
                logits_bh[b] = -1e9
    elif not row_active(globs, 0):
        logits_bh.fill_(-1e9)
    if globs.logits.ndim == 1:
        globs.logits.copy_(logits_bh.squeeze(0))
    else:
        globs.logits.copy_(logits_bh)


# -------- Prefill: full-sequence ops (one forward per layer) --------


def fused_prefill_rms_qkv_rope_write_cache(
    globs: Globals, layer_idx: int, seq_len: int
) -> None:
    """Prefill: RMS(hidden_states_seq) -> QKV -> RoPE on Q,K -> write K,V to cache, Q to prefill_q_seq."""
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    eps = LlamaConfig.rms_norm_eps
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    hs_b = _as_prefill_BSH(globs, seq_len)
    B = hs_b.shape[0]
    W = globs.qkv_weights[layer_idx]
    post_ln = _rms_norm(hs_b, globs.attn_norm_weights[layer_idx], eps=eps)
    qkv = einsum(W, post_ln, "o i, b s i -> b s o")
    q = qkv[:, :, :k_start].reshape(B, seq_len, n_heads, head_dim)
    k = qkv[:, :, k_start:v_start].reshape(B, seq_len, n_kv_heads, head_dim)
    v = qkv[:, :, v_start:].reshape(B, seq_len, n_kv_heads, head_dim)
    cos, sin = _prefill_rope_cos_sin(globs, seq_len)
    q_rope = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_rope = (k * cos) + (_rotate_half_interleaved(k) * sin)
    _prefill_write_q_seq(globs, seq_len, q_rope)
    for b in range(B):
        globs.k_cache[layer_idx, b, :seq_len, :, :].copy_(
            k_rope[b].to(globs.k_cache.dtype)
        )
        globs.v_cache[layer_idx, b, :seq_len, :, :].copy_(v[b].to(globs.v_cache.dtype))


def fused_prefill_attention(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: causal SDPA over full Q, K, V; write attn_out_seq."""
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads

    B = _prefill_batch_n(globs)
    for b in range(B):
        q = _prefill_attn_q_slice(globs, seq_len, b).float()
        k = globs.k_cache[layer_idx, b, :seq_len, :, :].float()
        v = globs.v_cache[layer_idx, b, :seq_len, :, :].float()
        k = (
            k.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1)
            .reshape(seq_len, n_heads, head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1)
            .reshape(seq_len, n_heads, head_dim)
        )
        q_bhld = q.unsqueeze(0).transpose(1, 2)
        k_bhld = k.unsqueeze(0).transpose(1, 2)
        v_bhld = v.unsqueeze(0).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q_bhld, k_bhld, v_bhld, is_causal=True
        )
        flat = attn_out.transpose(1, 2).squeeze(0).reshape(seq_len, -1)
        _prefill_write_attn_out_seq(globs, seq_len, b, flat)


def fused_prefill_oproj_residual(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: attn_out_seq @ o_proj + residual -> hidden_states_seq."""
    a = globs.attn_out_seq
    attn_b = a[:seq_len].unsqueeze(0) if a.ndim == 2 else a[:, :seq_len, :]
    W = globs.o_proj_weights[layer_idx]
    out = einsum(W, attn_b, "o i, b s i -> b s o")
    _prefill_add_hs_residual(globs, seq_len, out)


def fused_prefill_mlp(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Prefill: RMS -> up/gate -> SiLU*gate*up -> down -> residual over full sequence."""
    hs_b = _as_prefill_BSH(globs, seq_len)
    post_ln = _rms_norm(
        hs_b, globs.mlp_norm_weights[layer_idx], eps=LlamaConfig.rms_norm_eps
    )
    up = einsum(globs.up_proj_weights[layer_idx], post_ln, "o i, b s i -> b s o")
    gate = einsum(globs.gate_proj_weights[layer_idx], post_ln, "o i, b s i -> b s o")
    silu_out = F.silu(gate) * up
    down = einsum(globs.down_proj_weights[layer_idx], silu_out, "o i, b s i -> b s o")
    _prefill_add_hs_residual(globs, seq_len, down)
