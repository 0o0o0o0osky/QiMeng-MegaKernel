"""
Shared baseline reference and globs building for Qwen3-0.6B baseline→fused verification (A100).

Qwen3 architecture differences vs Qwen2.5:
- Q-norm and K-norm: per-head RMSNorm applied AFTER QKV projection and BEFORE RoPE
- No QKV bias (attention_bias=False)
- GQA ratio = 2 (16 Q heads, 8 KV heads)
- tie_word_embeddings=True

In the baseline functions, we use HF weights directly (natural order) and apply the
same interleaved RoPE convention as the fused ops. The sequence is:
  1. input_layernorm → x_normed
  2. q/k/v_proj(x_normed)  [no bias]
  3. Reshape to heads, apply interleave index (natural → interleaved layout)
  4. Apply HF q_norm/k_norm in natural order, then interleave  (equivalent to fused)
  5. Interleaved RoPE
  6. Write q_post_rope, k_cache, v_cache
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from einops import einsum

from qimeng_megakernel.models.LLM.llama.preprocess import _one_head_interleave_indices
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.model_config import Qwen3Config

QWEN3_0_6B_DEFAULT_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-0.6B")


def _rms_norm(inp: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)
    return weight * inp.to(input_dtype)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    stacked = torch.stack((-x2, x1), dim=-1)
    return stacked.view_as(x)


def _apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


def _baseline_qkv_qknorm_rope(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    """Shared implementation: QKV proj → Q/K-norm (HF) → interleave → RoPE."""
    pos_id = globs.position_id
    layer = model.model.layers[layer_idx]
    n_heads = Qwen3Config.num_attention_heads
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    device = globs.hidden_states.device
    dtype = globs.hidden_states.dtype
    eps = Qwen3Config.rms_norm_eps

    x = globs.hidden_states.unsqueeze(0).unsqueeze(0)
    x_normed = layer.input_layernorm(x)

    # No bias in Qwen3
    q = layer.self_attn.q_proj(x_normed)
    k = layer.self_attn.k_proj(x_normed)
    v = layer.self_attn.v_proj(x_normed)

    b, s, _ = q.shape
    q = q.view(b, s, n_heads, head_dim).squeeze(0).squeeze(0)    # [n_heads, head_dim]
    k = k.view(b, s, n_kv_heads, head_dim).squeeze(0).squeeze(0) # [n_kv_heads, head_dim]
    v = v.view(b, s, n_kv_heads, head_dim).squeeze(0).squeeze(0) # [n_kv_heads, head_dim]

    # Apply Q-norm and K-norm in natural order (HF formula: RMSNorm per head)
    q = _rms_norm(q, layer.self_attn.q_norm.weight.unsqueeze(0), eps=eps).squeeze()
    k = _rms_norm(k, layer.self_attn.k_norm.weight.unsqueeze(0), eps=eps).squeeze()

    # Re-add head dim if squeezed away (single head case)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if k.dim() == 1:
        k = k.unsqueeze(0)

    # Interleave Q and K only — V does not go through RoPE and is stored in natural order
    idx = _one_head_interleave_indices(head_dim, device)
    q = q[:, idx].contiguous()   # [n_heads, head_dim], interleaved
    k = k[:, idx].contiguous()   # [n_kv_heads, head_dim], interleaved
    # V stays in natural order (matches fused ops which don't permute v_proj weights)

    # RoPE in interleaved space
    cos = globs.rope_cos[pos_id]
    sin = globs.rope_sin[pos_id]
    q, k = _apply_rotary_pos_emb_interleaved(
        q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=0
    )

    q_flat = q.reshape(-1).to(dtype)
    k_flat = k.reshape(-1).to(dtype)
    v_flat = v.reshape(-1).to(dtype)

    expected_k_cache = globs.k_cache.clone()
    expected_v_cache = globs.v_cache.clone()
    k_slot = k_flat.view(n_kv_heads, head_dim).to(expected_k_cache.dtype)
    v_slot = v_flat.view(n_kv_heads, head_dim).to(expected_v_cache.dtype)
    if expected_k_cache.dim() == 5:
        # Cache may have more head slots than n_kv_heads (e.g. from HF Cache API); assign only our heads
        expected_k_cache[layer_idx, 0, pos_id, :n_kv_heads, :] = k_slot
        expected_v_cache[layer_idx, 0, pos_id, :n_kv_heads, :] = v_slot
    else:
        expected_k_cache[layer_idx, pos_id, :n_kv_heads, :] = k_slot
        expected_v_cache[layer_idx, pos_id, :n_kv_heads, :] = v_slot

    return {
        "q_post_rope": q_flat,
        "k_cache": expected_k_cache,
        "v_cache": expected_v_cache,
    }


# ---- Layer 0 helpers (convenience wrappers for test_1 / test_2 / test_3 / test_3b) ----

def expected_layer0_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: layer 0 QKV + Q/K-norm + interleaved RoPE."""
    return _baseline_qkv_qknorm_rope(globs, model, layer_idx=0)


def expected_layer0_attention_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = Qwen3Config.num_attention_heads
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer0_from_baseline_hf(globs, model)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    q = q_flat.view(n_heads, head_dim).float()
    if k_cache.dim() == 5:
        k = k_cache[0, 0, :seq_len, :, :].float()
        v = v_cache[0, 0, :seq_len, :, :].float()
    else:
        k = k_cache[0, :seq_len, :, :].float()
        v = v_cache[0, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    return {"attn_out": out.reshape(-1).to(dtype)}


def expected_layer0_o_proj_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    layer = model.model.layers[0]
    dtype = globs.hidden_states.dtype

    baseline_attn = expected_layer0_attention_from_baseline_hf(globs, model)
    attn_out = baseline_attn["attn_out"]
    residual = globs.hidden_states.clone()
    o_out = layer.self_attn.o_proj(attn_out.unsqueeze(0).unsqueeze(0))
    hidden_states = (residual + o_out[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer0_full_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    layer = model.model.layers[0]
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


# ---- Layer N helpers ----

def expected_layer_N_from_baseline_hf(
    globs: Globals, model: Any, layer_idx: int
) -> dict[str, torch.Tensor]:
    """Baseline: layer N QKV + Q/K-norm + interleaved RoPE."""
    return _baseline_qkv_qknorm_rope(globs, model, layer_idx=layer_idx)


def expected_layer_N_attention_from_baseline_hf(
    globs: Globals, model: Any, layer_idx: int
) -> dict[str, torch.Tensor]:
    pos_id = globs.position_id
    n_heads = Qwen3Config.num_attention_heads
    n_kv_heads = Qwen3Config.num_key_value_heads
    head_dim = Qwen3Config.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer_N_from_baseline_hf(globs, model, layer_idx)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    # Cap seq_len to actual cache length (long prompt may have position_id+1 > cache slots)
    want_len = (pos_id or 0) + 1
    if k_cache.dim() == 5:
        cache_len = k_cache.shape[2]
    else:
        cache_len = k_cache.shape[1]
    seq_len = min(want_len, cache_len)

    q = q_flat.view(n_heads, head_dim).float()
    if k_cache.dim() == 5:
        k = k_cache[layer_idx, 0, :seq_len, :, :].float()
        v = v_cache[layer_idx, 0, :seq_len, :, :].float()
    else:
        k = k_cache[layer_idx, :seq_len, :, :].float()
        v = v_cache[layer_idx, :seq_len, :, :].float()
    # Reshape using actual slice length (cache may have fewer slots than want_len)
    n_seq = k.shape[0]
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(n_seq, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(n_seq, n_heads, head_dim)
    seq_len = n_seq

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    return {"attn_out": out.reshape(-1).to(dtype)}


def expected_layer_N_o_proj_from_baseline_hf(
    globs: Globals, model: Any, layer_idx: int
) -> dict[str, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    baseline_attn = expected_layer_N_attention_from_baseline_hf(globs, model, layer_idx)
    attn_out = baseline_attn["attn_out"]
    residual = globs.hidden_states.clone()
    o_out = layer.self_attn.o_proj(attn_out.unsqueeze(0).unsqueeze(0))
    hidden_states = (residual + o_out[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer_N_full_from_baseline_hf(
    globs: Globals, model: Any, layer_idx: int
) -> dict[str, torch.Tensor]:
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
    """Advance globs forward by one HF layer (mutates in-place)."""
    base = expected_layer_N_from_baseline_hf(globs, model, layer_idx)
    globs.k_cache.copy_(base["k_cache"])
    globs.v_cache.copy_(base["v_cache"])
    full = expected_layer_N_full_from_baseline_hf(globs, model, layer_idx)
    globs.hidden_states.copy_(full["hidden_states"])


def expected_logits_same_formula(globs: Globals) -> dict[str, torch.Tensor]:
    """Compute logits using fused weights (no HF model); for lm_head verification."""
    eps = Qwen3Config.rms_norm_eps
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.lm_head_norm_weights,
        eps=eps,
    )
    logits = einsum(globs.lm_head_weights, post_ln, "o i, i -> o")
    return {"logits": logits.to(globs.logits.dtype)}


def run_baseline_decode_step(globs: Globals, model: Any) -> torch.Tensor:
    """Full decode step with baseline (HF proj + interleaved RoPE). Returns logits."""
    n_layers = Qwen3Config.num_hidden_layers
    for layer_idx in range(n_layers):
        advance_globs_by_hf_layer(globs, model, layer_idx)
    return expected_logits_same_formula(globs)["logits"]
