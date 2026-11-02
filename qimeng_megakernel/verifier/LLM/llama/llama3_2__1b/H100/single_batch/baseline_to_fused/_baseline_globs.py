"""
Shared baseline reference and globs building for Llama 3.2 1B baseline→fused verification.

Used by test_single_layer_all_outputs, test_multi_layer_all_outputs, and other single/full
tests. Provides:
- expected_*_from_hf_actual: HF as reference (HF rotary_emb + HF apply_rotary_pos_emb).
- expected_*_from_baseline_hf: same-formula reference (HF proj + our interleaved RoPE), for internal checks.
- make_globs_for_layer0_qkv, make_globs_from_prefill, make_globs_for_layer_N_qkv.
"""
from __future__ import annotations

from typing import Any

import os
import torch
import torch.nn.functional as F
from einops import einsum

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
except Exception:
    hf_apply_rotary_pos_emb = None

from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.model_config import (
    LlamaConfig,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.instructions import (
    scheduler as llama_scheduler,
)
from qimeng_megakernel.models.LLM.llama.preprocess import (
    _load_hf_llama_model,
    _one_head_interleave_indices,
    _stack_llama_weights_from_hf,
    preprocess_decode as llama_preprocess_decode,
)
from qimeng_megakernel.models.LLM.preprocess import prepare_kv_cache_for_decode

LLAMA_32_1B_DEFAULT_MODEL = os.environ.get(
    "MODEL_LLAMA3_2_1B",
    os.environ.get("LLAMA_32_1B_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
)


def _rms_norm(inp: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)
    return weight * inp.to(input_dtype)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    new_x1 = -x2
    new_x2 = x1
    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)


def _apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def deinterleave_slot_inplace(k_cache: torch.Tensor, layer_idx: int, batch_idx: int, pos: int, half_head_dim: int) -> None:
    """De-interleave the last dimension of k_cache at (layer_idx, batch_idx, pos) in-place."""
    slot = k_cache[layer_idx, batch_idx, pos]  # (n_kv_heads, head_dim)
    head_dim = slot.shape[-1]
    half = half_head_dim
    tmp = slot.clone()
    slot[:, :half] = tmp[:, ::2]
    slot[:, half:] = tmp[:, 1::2]


def deinterleave_qk_to_hf_layout(
    q_post_rope: torch.Tensor,
    k_cache: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert fused outputs from interleaved layout to HF layout for comparison with HF baseline.
    Fused uses interleaved (d0, d_half, d1, d_half+1, ...); HF uses (d0..d_half-1, d_half..d_d-1).
    """
    half = head_dim // 2
    # q_post_rope: (n_heads * head_dim,) -> (n_heads, head_dim) -> de-interleave last dim
    q = q_post_rope.view(n_heads, head_dim)
    q_hf = torch.empty_like(q)
    q_hf[:, :half] = q[:, ::2]
    q_hf[:, half:] = q[:, 1::2]
    q_post_rope_hf = q_hf.reshape(-1)
    # k_cache: (..., n_kv_heads, head_dim) -> de-interleave last dim
    k_hf = torch.empty_like(k_cache)
    k_hf[..., :half] = k_cache[..., ::2]
    k_hf[..., half:] = k_cache[..., 1::2]
    return q_post_rope_hf, k_hf


def expected_layer0_rms_qkv_rope_full(globs: Globals) -> dict[str, torch.Tensor]:
    """Fused/same-formula reference for layer 0 RMSNorm + QKV + RoPE (full, non-tiled)."""
    layer_idx = 0
    pos_id = globs.position_id
    eps = LlamaConfig.rms_norm_eps
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    k_start = n_heads * head_dim
    v_start = k_start + n_kv_heads * head_dim

    hidden = globs.hidden_states
    post_ln = _rms_norm(
        hidden,
        globs.attn_norm_weights[layer_idx],
        eps=eps,
    )
    qkv = einsum(
        globs.qkv_weights[layer_idx],
        post_ln,
        "o i, i -> o",
    )
    q = qkv[:k_start].view(1, n_heads, head_dim)
    k = qkv[k_start:v_start].view(1, n_kv_heads, head_dim)
    v = qkv[v_start:].view(1, n_kv_heads, head_dim)

    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
        q, k, cos, sin, unsqueeze_dim=0
    )
    q_flat = q_rope.view(-1)
    k_flat = k_rope.view(-1)
    v_flat = v.view(-1)

    expected_q = q_flat.to(globs.q_post_rope.dtype)
    expected_k_cache = globs.k_cache.clone()
    expected_v_cache = globs.v_cache.clone()
    expected_k_cache[layer_idx, :, pos_id] = k_flat.view(
        expected_k_cache[layer_idx, :, pos_id].shape
    ).to(expected_k_cache.dtype)
    expected_v_cache[layer_idx, :, pos_id] = v_flat.view(
        expected_v_cache[layer_idx, :, pos_id].shape
    ).to(expected_v_cache.dtype)

    return {
        "q_post_rope": expected_q,
        "k_cache": expected_k_cache,
        "v_cache": expected_v_cache,
    }


def expected_layer0_from_hf_actual(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 QKV + HF rotary_emb + HF apply_rotary_pos_emb (HF as reference)."""
    if hf_apply_rotary_pos_emb is None:
        raise RuntimeError("transformers Llama apply_rotary_pos_emb not available")
    layer_idx = 0
    pos_id = globs.position_id
    layer = model.model.layers[layer_idx]
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    device = globs.hidden_states.device
    dtype = globs.hidden_states.dtype

    x = globs.hidden_states.unsqueeze(0).unsqueeze(0)
    position_ids = torch.tensor([[pos_id]], device=device, dtype=torch.long)
    cos, sin = model.model.rotary_emb(x, position_ids)

    hidden_norm = layer.input_layernorm(x)
    hidden_shape = (*hidden_norm.shape[:-1], n_heads, head_dim)
    kv_shape = (*hidden_norm.shape[:-1], n_kv_heads, head_dim)
    q = layer.self_attn.q_proj(hidden_norm).view(hidden_shape).transpose(1, 2)
    k = layer.self_attn.k_proj(hidden_norm).view(kv_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(hidden_norm).view(kv_shape).transpose(1, 2)

    q, k = hf_apply_rotary_pos_emb(q, k, cos, sin)

    q_flat = q.reshape(-1).to(dtype)
    k_flat = k.reshape(-1).to(dtype)
    v_flat = v.reshape(-1).to(dtype)

    expected_k_cache = globs.k_cache.clone()
    expected_v_cache = globs.v_cache.clone()
    slot_k = expected_k_cache[layer_idx, :, pos_id].shape
    expected_k_cache[layer_idx, :, pos_id] = k_flat.view(slot_k)
    expected_v_cache[layer_idx, :, pos_id] = v_flat.view(expected_v_cache[layer_idx, :, pos_id].shape)

    return {
        "q_post_rope": q_flat,
        "k_cache": expected_k_cache,
        "v_cache": expected_v_cache,
    }


def expected_layer0_attention_from_hf_actual(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 attention (softmax(q@k^T/sqrt(d))@v) using HF q,k,v."""
    import math

    layer_idx = 0
    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer0_from_hf_actual(globs, model)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    q = q_flat.view(n_heads, head_dim).float()
    k = k_cache[layer_idx, 0, :seq_len, :, :].float()
    v = v_cache[layer_idx, 0, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    attn_out = out.reshape(-1).to(dtype)
    return {"attn_out": attn_out}


def expected_layer0_o_proj_from_hf_actual(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 attn out then o_proj + residual."""
    layer_idx = 0
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    baseline_attn = expected_layer0_attention_from_hf_actual(globs, model)
    attn_out = baseline_attn["attn_out"]
    residual = globs.hidden_states.clone()
    o_out = layer.self_attn.o_proj(attn_out.unsqueeze(0).unsqueeze(0))
    hidden_states = (residual + o_out[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer0_full_from_hf_actual(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 full forward (attn block + mlp block)."""
    layer_idx = 0
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    o_proj_out = expected_layer0_o_proj_from_hf_actual(globs, model)
    hidden_states = o_proj_out["hidden_states"].clone()

    normed = layer.post_attention_layernorm(hidden_states.unsqueeze(0).unsqueeze(0))
    gate = layer.mlp.gate_proj(normed)
    up = layer.mlp.up_proj(normed)
    x = F.silu(gate) * up
    down = layer.mlp.down_proj(x)
    hidden_states = (hidden_states + down[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer0_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 QKV + interleaved RoPE via globs.rope_cos/sin (the kernel's pairing convention).
    HF proj output is HF head_dim order; reorder Q/K last dim to interleaved before RoPE
    so that cos/sin (interleaved) pair correct dimensions (preprocess_decode_pitfalls §2)."""
    layer_idx = 0
    pos_id = globs.position_id
    layer = model.model.layers[layer_idx]
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    device = globs.hidden_states.device
    dtype = globs.hidden_states.dtype

    x = globs.hidden_states.unsqueeze(0).unsqueeze(0)
    x_normed = layer.input_layernorm(x)
    x = x_normed
    q = layer.self_attn.q_proj(x)
    k = layer.self_attn.k_proj(x)
    v = layer.self_attn.v_proj(x)
    b, s, _ = q.shape
    q = q.view(b, s, n_heads, head_dim)
    k = k.view(b, s, n_kv_heads, head_dim)
    v = v.view(b, s, n_kv_heads, head_dim)
    # Reorder Q/K last dim to interleaved so interleaved RoPE pairs correct dims (prompt §2).
    idx = _one_head_interleave_indices(head_dim, device)
    q = q[..., idx].contiguous()
    k = k[..., idx].contiguous()
    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    q_s = q.squeeze(1)
    k_s = k.squeeze(1)
    q, k = _apply_rotary_pos_emb_interleaved(q_s, k_s, cos, sin, unsqueeze_dim=0)
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)

    q_flat = q.reshape(-1).to(dtype)
    k_flat = k.reshape(-1).to(dtype)
    v_flat = v.reshape(-1).to(dtype)

    expected_k_cache = globs.k_cache.clone()
    expected_v_cache = globs.v_cache.clone()
    slot_shape = expected_k_cache[layer_idx, :, pos_id].shape
    expected_k_cache[layer_idx, :, pos_id] = k_flat.view(slot_shape)
    expected_v_cache[layer_idx, :, pos_id] = v_flat.view(expected_v_cache[layer_idx, :, pos_id].shape)

    return {
        "q_post_rope": q_flat,
        "k_cache": expected_k_cache,
        "v_cache": expected_v_cache,
    }


def expected_layer0_attention_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 attention (softmax(q@k^T/sqrt(d))@v) using baseline q,k,v."""
    import math

    layer_idx = 0
    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer0_from_baseline_hf(globs, model)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    q = q_flat.view(n_heads, head_dim).float()
    k = k_cache[layer_idx, 0, :seq_len, :, :].float()
    v = v_cache[layer_idx, 0, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    attn_out = out.reshape(-1).to(dtype)
    return {"attn_out": attn_out}


def expected_layer0_o_proj_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 attn out then o_proj + residual."""
    layer_idx = 0
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    baseline_attn = expected_layer0_attention_from_baseline_hf(globs, model)
    attn_out = baseline_attn["attn_out"]
    residual = globs.hidden_states.clone()
    o_out = layer.self_attn.o_proj(attn_out.unsqueeze(0).unsqueeze(0))
    hidden_states = (residual + o_out[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer0_full_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 full forward (attn block + mlp block)."""
    layer_idx = 0
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    o_proj_out = expected_layer0_o_proj_from_baseline_hf(globs, model)
    hidden_states = o_proj_out["hidden_states"].clone()

    normed = layer.post_attention_layernorm(hidden_states.unsqueeze(0).unsqueeze(0))
    gate = layer.mlp.gate_proj(normed)
    up = layer.mlp.up_proj(normed)
    x = F.silu(gate) * up
    down = layer.mlp.down_proj(x)
    hidden_states = (hidden_states + down[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer_N_from_hf_actual(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L QKV + HF rotary_emb + HF apply_rotary_pos_emb (HF as reference)."""
    if hf_apply_rotary_pos_emb is None:
        raise RuntimeError("transformers Llama apply_rotary_pos_emb not available")
    pos_id = globs.position_id
    layer = model.model.layers[layer_idx]
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    device = globs.hidden_states.device
    dtype = globs.hidden_states.dtype

    x = globs.hidden_states.unsqueeze(0).unsqueeze(0)
    position_ids = torch.tensor([[pos_id]], device=device, dtype=torch.long)
    cos, sin = model.model.rotary_emb(x, position_ids)

    hidden_norm = layer.input_layernorm(x)
    hidden_shape = (*hidden_norm.shape[:-1], n_heads, head_dim)
    kv_shape = (*hidden_norm.shape[:-1], n_kv_heads, head_dim)
    q = layer.self_attn.q_proj(hidden_norm).view(hidden_shape).transpose(1, 2)
    k = layer.self_attn.k_proj(hidden_norm).view(kv_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(hidden_norm).view(kv_shape).transpose(1, 2)

    q, k = hf_apply_rotary_pos_emb(q, k, cos, sin)

    q_flat = q.reshape(-1).to(dtype)
    k_flat = k.reshape(-1).to(dtype)
    v_flat = v.reshape(-1).to(dtype)

    expected_k_cache = globs.k_cache.clone()
    expected_v_cache = globs.v_cache.clone()
    expected_k_cache[layer_idx, :, pos_id] = k_flat.view(expected_k_cache[layer_idx, :, pos_id].shape)
    expected_v_cache[layer_idx, :, pos_id] = v_flat.view(expected_v_cache[layer_idx, :, pos_id].shape)

    return {"q_post_rope": q_flat, "k_cache": expected_k_cache, "v_cache": expected_v_cache}


def expected_layer_N_attention_from_hf_actual(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L attention output (before o_proj) using HF q,k,v."""
    import math

    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer_N_from_hf_actual(globs, model, layer_idx)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    q = q_flat.view(n_heads, head_dim).float()
    k = k_cache[layer_idx, 0, :seq_len, :, :].float()
    v = v_cache[layer_idx, 0, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    attn_out = out.reshape(-1).to(dtype)
    return {"attn_out": attn_out}


def expected_layer_N_o_proj_from_hf_actual(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L attn out then o_proj + residual."""
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    baseline_attn = expected_layer_N_attention_from_hf_actual(globs, model, layer_idx)
    attn_out = baseline_attn["attn_out"]
    residual = globs.hidden_states.clone()
    o_out = layer.self_attn.o_proj(attn_out.unsqueeze(0).unsqueeze(0))
    hidden_states = (residual + o_out[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer_N_full_from_hf_actual(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L full forward (attn block + mlp block)."""
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    o_proj_out = expected_layer_N_o_proj_from_hf_actual(globs, model, layer_idx)
    hidden_states = o_proj_out["hidden_states"].clone()

    normed = layer.post_attention_layernorm(hidden_states.unsqueeze(0).unsqueeze(0))
    gate = layer.mlp.gate_proj(normed)
    up = layer.mlp.up_proj(normed)
    x = F.silu(gate) * up
    down = layer.mlp.down_proj(x)
    hidden_states = (hidden_states + down[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def advance_globs_by_hf_layer_actual(globs: Globals, model: Any, layer_idx: int) -> None:
    """Advance globs by one HF layer (HF RoPE): write k,v at layer_idx and set hidden_states = layer output."""
    base = expected_layer_N_from_hf_actual(globs, model, layer_idx)
    globs.k_cache.copy_(base["k_cache"])
    globs.v_cache.copy_(base["v_cache"])
    full = expected_layer_N_full_from_hf_actual(globs, model, layer_idx)
    globs.hidden_states.copy_(full["hidden_states"])


def expected_layer_N_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L RMSNorm + QKV + interleaved RoPE (globs.rope_cos/sin).
    Reorder Q/K last dim to interleaved before RoPE (same as expected_layer0_from_baseline_hf)."""
    pos_id = globs.position_id
    layer = model.model.layers[layer_idx]
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    device = globs.hidden_states.device
    dtype = globs.hidden_states.dtype

    x = globs.hidden_states.unsqueeze(0).unsqueeze(0)
    x = layer.input_layernorm(x)
    q = layer.self_attn.q_proj(x)
    k = layer.self_attn.k_proj(x)
    v = layer.self_attn.v_proj(x)
    b, s, _ = q.shape
    q = q.view(b, s, n_heads, head_dim)
    k = k.view(b, s, n_kv_heads, head_dim)
    v = v.view(b, s, n_kv_heads, head_dim)
    idx = _one_head_interleave_indices(head_dim, device)
    q = q[..., idx].contiguous()
    k = k[..., idx].contiguous()
    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    q_s = q.squeeze(1)
    k_s = k.squeeze(1)
    q, k = _apply_rotary_pos_emb_interleaved(q_s, k_s, cos, sin, unsqueeze_dim=0)
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)

    q_flat = q.reshape(-1).to(dtype)
    k_flat = k.reshape(-1).to(dtype)
    v_flat = v.reshape(-1).to(dtype)

    expected_k_cache = globs.k_cache.clone()
    expected_v_cache = globs.v_cache.clone()
    expected_k_cache[layer_idx, :, pos_id] = k_flat.view(expected_k_cache[layer_idx, :, pos_id].shape)
    expected_v_cache[layer_idx, :, pos_id] = v_flat.view(expected_v_cache[layer_idx, :, pos_id].shape)

    return {"q_post_rope": q_flat, "k_cache": expected_k_cache, "v_cache": expected_v_cache}


def expected_layer_N_attention_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L attention output (before o_proj)."""
    import math

    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = LlamaConfig.num_attention_heads
    n_kv_heads = LlamaConfig.num_key_value_heads
    head_dim = LlamaConfig.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer_N_from_baseline_hf(globs, model, layer_idx)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    q = q_flat.view(n_heads, head_dim).float()
    k = k_cache[layer_idx, 0, :seq_len, :, :].float()
    v = v_cache[layer_idx, 0, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    attn_out = out.reshape(-1).to(dtype)
    return {"attn_out": attn_out}


def expected_layer_N_o_proj_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L attn out then o_proj + residual."""
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    baseline_attn = expected_layer_N_attention_from_baseline_hf(globs, model, layer_idx)
    attn_out = baseline_attn["attn_out"]
    residual = globs.hidden_states.clone()
    o_out = layer.self_attn.o_proj(attn_out.unsqueeze(0).unsqueeze(0))
    hidden_states = (residual + o_out[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer_N_full_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Baseline: HF layer L full forward (attn block + mlp block)."""
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    o_proj_out = expected_layer_N_o_proj_from_baseline_hf(globs, model, layer_idx)
    hidden_states = o_proj_out["hidden_states"].clone()

    normed = layer.post_attention_layernorm(hidden_states.unsqueeze(0).unsqueeze(0))
    gate = layer.mlp.gate_proj(normed)
    up = layer.mlp.up_proj(normed)
    x = F.silu(gate) * up
    down = layer.mlp.down_proj(x)
    hidden_states = (hidden_states + down[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def advance_globs_by_hf_layer(globs: Globals, model: Any, layer_idx: int) -> None:
    """Advance globs by one HF layer: write k,v at layer_idx and set hidden_states = layer output."""
    base = expected_layer_N_from_baseline_hf(globs, model, layer_idx)
    globs.k_cache.copy_(base["k_cache"])
    globs.v_cache.copy_(base["v_cache"])
    full = expected_layer_N_full_from_baseline_hf(globs, model, layer_idx)
    globs.hidden_states.copy_(full["hidden_states"])


def make_globs_for_layer_N_qkv(
    layer_idx: int,
    model_name: str = LLAMA_32_1B_DEFAULT_MODEL,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    position_id: int = 0,
    return_model: bool = False,
    use_hf_actual: bool = False,
) -> Globals | tuple[Globals, Any]:
    """Build globs for one decode step with hidden_states = input to layer_idx (layer 0 from embed; layer>0 from HF 0..layer_idx-1).
    If use_hf_actual=True, advance layers with HF RoPE (advance_globs_by_hf_layer_actual) for HF-as-reference tests."""
    globs, model = make_globs_for_layer0_qkv(
        model_name=model_name,
        device=device,
        dtype=dtype,
        position_id=position_id,
        return_model=True,
    )
    advance_fn = advance_globs_by_hf_layer_actual if use_hf_actual else advance_globs_by_hf_layer
    for i in range(layer_idx):
        advance_fn(globs, model, i)
    if return_model:
        return globs, model
    return globs


def make_globs_from_prefill(
    prefill: Any,
    model: Any,
    model_name: str,
) -> tuple[Globals, Any]:
    """Build globs for one decode step from prefill: hidden_states = embed(first_gen_token), position_id = prompt_len."""
    globs = llama_preprocess_decode(
        prefill=prefill,
        model=model,
        model_name=model_name,
        max_new_tokens=1,
    )
    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    globs.hidden_states.copy_(
        emb.squeeze(0).squeeze(0).to(globs.hidden_states.dtype)
    )
    prompt_len = int(prefill.input_ids.shape[1])
    assert globs.position_id == prompt_len, "preprocess should set position_id = prompt_len"
    return globs, model


def expected_logits_same_formula(globs: Globals) -> dict[str, torch.Tensor]:
    """Same-formula reference for lm_head: RMSNorm(hidden_states) @ lm_head_weights."""
    eps = LlamaConfig.rms_norm_eps
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.lm_head_norm_weights,
        eps=eps,
    )
    logits = einsum(globs.lm_head_weights, post_ln, "o i, i -> o")
    return {"logits": logits.to(globs.logits.dtype)}


def final_norm_hidden_states(globs: Globals) -> torch.Tensor:
    """Hidden states after final RMSNorm (input to lm_head). Same as fused layer_hidden[-1]."""
    return _rms_norm(
        globs.hidden_states,
        globs.lm_head_norm_weights,
        LlamaConfig.rms_norm_eps,
    )


def run_baseline_decode_step(globs: Globals, model: Any) -> torch.Tensor:
    """One decode step with same baseline as single/multi (HF proj + interleaved RoPE). Runs all layers then lm_head; returns logits."""
    n_layers = LlamaConfig.num_hidden_layers
    for layer_idx in range(n_layers):
        advance_globs_by_hf_layer(globs, model, layer_idx)
    return expected_logits_same_formula(globs)["logits"]


def select_batch_row(globs: Any, batch_idx: int) -> Globals:
    """Clone one logical row into the ordinary single-row baseline state."""
    position_ids = getattr(globs, "position_ids", None)
    if position_ids is None:
        raise ValueError("multi-batch baseline requires position_ids")
    return Globals(
        instructions=None,
        barriers=globs.barriers.clone(),
        position_id=int(position_ids[batch_idx].item()),
        attn_norm_weights=globs.attn_norm_weights,
        qkv_weights=globs.qkv_weights,
        rope_cos=globs.rope_cos,
        rope_sin=globs.rope_sin,
        o_proj_weights=globs.o_proj_weights,
        mlp_norm_weights=globs.mlp_norm_weights,
        up_proj_weights=globs.up_proj_weights,
        gate_proj_weights=globs.gate_proj_weights,
        down_proj_weights=globs.down_proj_weights,
        lm_head_norm_weights=globs.lm_head_norm_weights,
        lm_head_weights=globs.lm_head_weights,
        k_cache=globs.k_cache[:, batch_idx : batch_idx + 1].clone(),
        v_cache=globs.v_cache[:, batch_idx : batch_idx + 1].clone(),
        hidden_states=globs.hidden_states[batch_idx].clone(),
        q_post_rope=globs.q_post_rope[batch_idx].clone(),
        attn_out=globs.attn_out[batch_idx].clone(),
        attn_out_intermediates=None,
        attn_lse_intermediates=None,
        silu_out=globs.silu_out[batch_idx].clone(),
        logits=globs.logits[batch_idx].clone(),
    )


def make_globs_for_layer0_qkv(
    model_name: str = LLAMA_32_1B_DEFAULT_MODEL,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    position_id: int = 0,
    return_model: bool = False,
) -> Globals | tuple[Globals, Any]:
    """Build globs for one decode step at position_id (layer 0 input from embed(1)); cache empty [L,B,S,H,D]."""
    dev = torch.device(device)
    model = _load_hf_llama_model(model_name, device=dev, dtype=dtype)
    LlamaConfig.update_from_hf(model.config)

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    hidden_size = model.config.hidden_size
    num_attn_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size
    vocab_size = model.config.vocab_size

    weights = _stack_llama_weights_from_hf(model)
    seq_len = 1
    past_key_values = []
    for _ in range(n_layers):
        k = torch.zeros(1, n_kv_heads, seq_len, head_dim, device=dev, dtype=dtype)
        v = torch.zeros(1, n_kv_heads, seq_len, head_dim, device=dev, dtype=dtype)
        past_key_values.append((k, v))

    kv = prepare_kv_cache_for_decode(past_key_values, seq_len_hint=seq_len)
    # Use scheduler's capped SM count to match A100 configuration (108 SMs)
    # even when running on H100 (132 SMs)
    sm_count = 1 if dev.type != "cuda" else llama_scheduler._get_sm_count(dev)
    max_attn_partitions = sm_count

    def make_buf(shape, buf_dtype=dtype):
        return torch.zeros(shape, device=dev, dtype=buf_dtype)

    hidden_states = make_buf(hidden_size)
    hidden_states.copy_(
        model.model.embed_tokens(torch.tensor([[1]], device=dev, dtype=torch.long))[0, 0]
    )

    barriers = torch.zeros(
        (n_layers, 10, num_attn_heads + n_kv_heads * 2),
        dtype=torch.int32,
        device=dev,
    )

    globs = Globals(
        position_id=position_id,
        k_cache=kv.k_cache.contiguous(),
        v_cache=kv.v_cache.contiguous(),
        qkv_weights=weights["qkv_weights"].contiguous(),
        o_proj_weights=weights["o_proj_weights"].contiguous(),
        attn_norm_weights=weights["attn_norm_weights"].contiguous(),
        mlp_norm_weights=weights["mlp_norm_weights"].contiguous(),
        up_proj_weights=weights["up_proj_weights"].contiguous(),
        gate_proj_weights=weights["gate_proj_weights"].contiguous(),
        down_proj_weights=weights["down_proj_weights"].contiguous(),
        lm_head_norm_weights=weights["lm_head_norm_weights"].contiguous(),
        lm_head_weights=weights["lm_head_weights"].contiguous(),
        rope_cos=weights["rope_cos"].contiguous(),
        rope_sin=weights["rope_sin"].contiguous(),
        hidden_states=hidden_states,
        q_post_rope=make_buf(hidden_size),
        attn_out=make_buf(hidden_size),
        attn_out_intermediates=make_buf(
            [num_attn_heads, max_attn_partitions, head_dim],
            buf_dtype=torch.float32,
        ),
        attn_lse_intermediates=make_buf(
            [num_attn_heads, max_attn_partitions],
            buf_dtype=torch.float32,
        ),
        silu_out=make_buf(intermediate_size),
        logits=make_buf(vocab_size),
        barriers=barriers,
    )
    if return_model:
        return globs, model
    return globs
