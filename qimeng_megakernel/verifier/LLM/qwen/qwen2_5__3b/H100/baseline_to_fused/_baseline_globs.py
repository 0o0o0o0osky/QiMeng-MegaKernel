"""
Shared baseline reference and globs building for Qwen2.5 3B baseline→fused verification (H100).

Uses Qwen2Config, qwen preprocess (interleaved K cache), and same baseline formulas as LLaMA.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import einsum

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
except Exception:
    hf_apply_rotary_pos_emb = None

from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.model_config import Qwen2Config
from qimeng_megakernel.models.LLM.llama.preprocess import (
    _one_head_interleave_indices,
    _interleave_k_cache,
)
from qimeng_megakernel.models.LLM.qwen.preprocess import (
    _load_hf_qwen2_model,
    _stack_qwen2_weights_from_hf,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.preprocess import (
    preprocess_decode as qwen2_preprocess_decode,
)
from qimeng_megakernel.models.LLM.preprocess import (
    prepare_kv_cache_for_decode,
    DecodePreprocessOutput,
)

QWEN2_5_3B_DEFAULT_MODEL = "Qwen/Qwen2.5-3B"


def _get_sm_count(dev: torch.device) -> int:
    if dev.type != "cuda":
        return 1
    return torch.cuda.get_device_properties(dev).multi_processor_count


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


def expected_layer0_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    """Baseline: HF layer0 QKV + interleaved RoPE (globs.rope_cos/sin)."""
    layer_idx = 0
    pos_id = globs.position_id
    layer = model.model.layers[layer_idx]
    n_heads = Qwen2Config.num_attention_heads
    n_kv_heads = Qwen2Config.num_key_value_heads
    head_dim = Qwen2Config.head_dim
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
    slot_shape = (n_kv_heads, head_dim)
    if expected_k_cache.dim() == 5:
        expected_k_cache[layer_idx, 0, pos_id, :, :] = k_flat.view(slot_shape).to(expected_k_cache.dtype)
        expected_v_cache[layer_idx, 0, pos_id, :, :] = v_flat.view(slot_shape).to(expected_v_cache.dtype)
    else:
        expected_k_cache[layer_idx, pos_id, :, :] = k_flat.view(slot_shape).to(expected_k_cache.dtype)
        expected_v_cache[layer_idx, pos_id, :, :] = v_flat.view(slot_shape).to(expected_v_cache.dtype)

    return {
        "q_post_rope": q_flat,
        "k_cache": expected_k_cache,
        "v_cache": expected_v_cache,
    }


def expected_layer0_attention_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    layer_idx = 0
    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = Qwen2Config.num_attention_heads
    n_kv_heads = Qwen2Config.num_key_value_heads
    head_dim = Qwen2Config.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer0_from_baseline_hf(globs, model)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    q = q_flat.view(n_heads, head_dim).float()
    if k_cache.dim() == 5:
        k = k_cache[layer_idx, 0, :seq_len, :, :].float()
        v = v_cache[layer_idx, 0, :seq_len, :, :].float()
    else:
        k = k_cache[layer_idx, :seq_len, :, :].float()
        v = v_cache[layer_idx, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    attn_out = out.reshape(-1).to(dtype)
    return {"attn_out": attn_out}


def expected_layer0_o_proj_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
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


def expected_layer_N_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    pos_id = globs.position_id
    layer = model.model.layers[layer_idx]
    n_heads = Qwen2Config.num_attention_heads
    n_kv_heads = Qwen2Config.num_key_value_heads
    head_dim = Qwen2Config.head_dim
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
    slot_shape = (n_kv_heads, head_dim)
    if expected_k_cache.dim() == 5:
        expected_k_cache[layer_idx, 0, pos_id, :, :] = k_flat.view(slot_shape).to(expected_k_cache.dtype)
        expected_v_cache[layer_idx, 0, pos_id, :, :] = v_flat.view(slot_shape).to(expected_v_cache.dtype)
    else:
        expected_k_cache[layer_idx, pos_id, :, :] = k_flat.view(slot_shape).to(expected_k_cache.dtype)
        expected_v_cache[layer_idx, pos_id, :, :] = v_flat.view(slot_shape).to(expected_v_cache.dtype)

    return {"q_post_rope": q_flat, "k_cache": expected_k_cache, "v_cache": expected_v_cache}


def expected_layer_N_attention_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    pos_id = globs.position_id
    seq_len = (pos_id or 0) + 1
    n_heads = Qwen2Config.num_attention_heads
    n_kv_heads = Qwen2Config.num_key_value_heads
    head_dim = Qwen2Config.head_dim
    gqa_ratio = n_heads // n_kv_heads
    dtype = globs.hidden_states.dtype

    baseline = expected_layer_N_from_baseline_hf(globs, model, layer_idx)
    q_flat = baseline["q_post_rope"]
    k_cache = baseline["k_cache"]
    v_cache = baseline["v_cache"]

    q = q_flat.view(n_heads, head_dim).float()
    if k_cache.dim() == 5:
        k = k_cache[layer_idx, 0, :seq_len, :, :].float()
        v = v_cache[layer_idx, 0, :seq_len, :, :].float()
    else:
        k = k_cache[layer_idx, :seq_len, :, :].float()
        v = v_cache[layer_idx, :seq_len, :, :].float()
    k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, n_heads, head_dim)

    qk = einsum(q, k, "h i, s h i -> h s")
    scaled_qk = qk * (1 / math.sqrt(head_dim))
    softmax = torch.softmax(scaled_qk, dim=-1)
    out = einsum(softmax, v, "h s, s h o -> h o")
    attn_out = out.reshape(-1).to(dtype)
    return {"attn_out": attn_out}


def expected_layer_N_o_proj_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    dtype = globs.hidden_states.dtype

    baseline_attn = expected_layer_N_attention_from_baseline_hf(globs, model, layer_idx)
    attn_out = baseline_attn["attn_out"]
    residual = globs.hidden_states.clone()
    o_out = layer.self_attn.o_proj(attn_out.unsqueeze(0).unsqueeze(0))
    hidden_states = (residual + o_out[0, 0]).to(dtype)
    return {"hidden_states": hidden_states}


def expected_layer_N_full_from_baseline_hf(globs: Globals, model: Any, layer_idx: int) -> dict[str, torch.Tensor]:
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
    base = expected_layer_N_from_baseline_hf(globs, model, layer_idx)
    globs.k_cache.copy_(base["k_cache"])
    globs.v_cache.copy_(base["v_cache"])
    full = expected_layer_N_full_from_baseline_hf(globs, model, layer_idx)
    globs.hidden_states.copy_(full["hidden_states"])


def make_globs_from_prefill(
    prefill: Any,
    model: Any,
    model_name: str,
) -> tuple[Globals, Any]:
    """Build globs for one decode step from prefill using Qwen2.5 3B H100 preprocess."""
    globs = qwen2_preprocess_decode(
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
    eps = Qwen2Config.rms_norm_eps
    post_ln = _rms_norm(
        globs.hidden_states,
        globs.lm_head_norm_weights,
        eps=eps,
    )
    logits = einsum(globs.lm_head_weights, post_ln, "o i, i -> o")
    return {"logits": logits.to(globs.logits.dtype)}


def run_baseline_decode_step(globs: Globals, model: Any) -> torch.Tensor:
    """One decode step with baseline (HF proj + interleaved RoPE). Runs all layers then lm_head; returns logits."""
    n_layers = Qwen2Config.num_hidden_layers
    for layer_idx in range(n_layers):
        advance_globs_by_hf_layer(globs, model, layer_idx)
    return expected_logits_same_formula(globs)["logits"]


def make_globs_for_layer0_qkv(
    model_name: str = QWEN2_5_3B_DEFAULT_MODEL,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    position_id: int = 0,
    return_model: bool = False,
) -> Globals | tuple[Globals, Any]:
    """Build globs for one decode step at position_id (layer 0 input from embed(1)); cache empty [L,B,S,H,D]."""
    dev = torch.device(device)
    model = _load_hf_qwen2_model(model_name, device=dev, dtype=dtype)
    Qwen2Config.update_from_hf(model.config)

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    hidden_size = model.config.hidden_size
    num_attn_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size
    vocab_size = model.config.vocab_size

    weights = _stack_qwen2_weights_from_hf(model)
    seq_len = 1
    past_key_values = []
    for _ in range(n_layers):
        k = torch.zeros(1, n_kv_heads, seq_len, head_dim, device=dev, dtype=dtype)
        v = torch.zeros(1, n_kv_heads, seq_len, head_dim, device=dev, dtype=dtype)
        past_key_values.append((k, v))

    kv = prepare_kv_cache_for_decode(past_key_values, seq_len_hint=seq_len)
    kv = DecodePreprocessOutput(
        k_cache=_interleave_k_cache(kv.k_cache, head_dim),
        v_cache=kv.v_cache,
    )
    max_attn_partitions = _get_sm_count(dev)

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
        qkv_bias=weights.get("qkv_bias"),
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
