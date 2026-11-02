"""HuggingFace references for Qwen1.5-MoE decode tests."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import einsum

from qimeng_megakernel.models.LLM.llama.preprocess import _one_head_interleave_indices
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.model_config import (
    QwenMoeConfig,
)


def _decode_length(globs: Globals) -> int:
    position = int(globs.position_id)
    if position < 0 or position >= globs.k_cache.shape[2]:
        raise ValueError(f"invalid decode position: {position}")
    return position + 1


def _rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    source_dtype = value.dtype
    value = value.float()
    variance = value.pow(2).mean(-1, keepdim=True)
    return weight * (value * torch.rsqrt(variance + QwenMoeConfig.rms_norm_eps)).to(source_dtype)


def _rotate_half_interleaved(value: torch.Tensor) -> torch.Tensor:
    even = value[..., ::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).view_as(value)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine.unsqueeze(0)
    sine = sine.unsqueeze(0)
    return (
        query * cosine + _rotate_half_interleaved(query) * sine,
        key * cosine + _rotate_half_interleaved(key) * sine,
    )


def expected_layer_from_baseline_hf(
    globs: Globals,
    model: Any,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Return QKV/RoPE outputs for one layer without mutating globs."""
    _decode_length(globs)
    position = int(globs.position_id)
    layer = model.model.layers[layer_idx]
    hidden = globs.hidden_states.view(1, 1, -1)
    normed = layer.input_layernorm(hidden)
    query = layer.self_attn.q_proj(normed)
    key = layer.self_attn.k_proj(normed)
    value = layer.self_attn.v_proj(normed)
    head_dim = QwenMoeConfig.head_dim
    query = query.view(1, QwenMoeConfig.num_attention_heads, head_dim)
    key = key.view(1, QwenMoeConfig.num_key_value_heads, head_dim)
    value = value.view(1, QwenMoeConfig.num_key_value_heads, head_dim)
    indices = _one_head_interleave_indices(head_dim, hidden.device)
    query = query[..., indices].contiguous()
    key = key[..., indices].contiguous()
    query, key = _apply_rope(
        query,
        key,
        globs.rope_cos[position],
        globs.rope_sin[position],
    )

    expected_k_cache = globs.k_cache.clone()
    expected_v_cache = globs.v_cache.clone()
    expected_k_cache[layer_idx, 0, position].copy_(
        key.reshape_as(expected_k_cache[layer_idx, 0, position])
    )
    expected_v_cache[layer_idx, 0, position].copy_(
        value.reshape_as(expected_v_cache[layer_idx, 0, position])
    )
    return {
        "q_post_rope": query.reshape(-1).to(globs.q_post_rope.dtype),
        "k_cache": expected_k_cache,
        "v_cache": expected_v_cache,
    }


def expected_layer_attention_from_baseline_hf(
    globs: Globals,
    model: Any,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Return decode attention output for one layer without mutating globs."""
    sequence_length = _decode_length(globs)
    reference = expected_layer_from_baseline_hf(globs, model, layer_idx)
    head_dim = QwenMoeConfig.head_dim
    query = reference["q_post_rope"].view(QwenMoeConfig.num_attention_heads, head_dim).float()
    key = reference["k_cache"][layer_idx, 0, :sequence_length].float()
    value = reference["v_cache"][layer_idx, 0, :sequence_length].float()
    group_size = QwenMoeConfig.num_attention_heads // QwenMoeConfig.num_key_value_heads
    key = key.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(
        sequence_length, QwenMoeConfig.num_attention_heads, head_dim
    )
    value = value.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(
        sequence_length, QwenMoeConfig.num_attention_heads, head_dim
    )
    scores = einsum(query, key, "h d, s h d -> h s") / math.sqrt(head_dim)
    output = einsum(torch.softmax(scores, dim=-1), value, "h s, s h d -> h d")
    return {"attn_out": output.reshape(-1).to(globs.attn_out.dtype)}


def expected_layer_o_proj_from_baseline_hf(
    globs: Globals,
    model: Any,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Return attention projection plus residual for one layer."""
    attention = expected_layer_attention_from_baseline_hf(globs, model, layer_idx)["attn_out"]
    projected = model.model.layers[layer_idx].self_attn.o_proj(attention.view(1, 1, -1))[0, 0]
    return {"hidden_states": globs.hidden_states + projected.to(globs.hidden_states.dtype)}


def expected_layer_full_from_baseline_hf(
    globs: Globals,
    model: Any,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Return the full attention and MoE layer output."""
    layer = model.model.layers[layer_idx]
    hidden = expected_layer_o_proj_from_baseline_hf(globs, model, layer_idx)["hidden_states"]
    normed = layer.post_attention_layernorm(hidden.view(1, 1, -1))
    moe_output = layer.mlp(normed)
    moe_hidden = moe_output[0] if isinstance(moe_output, tuple) else moe_output
    return {"hidden_states": hidden + moe_hidden[0, 0].to(hidden.dtype)}


def expected_layer0_from_baseline_hf(globs: Globals, model: Any) -> dict[str, torch.Tensor]:
    return expected_layer_from_baseline_hf(globs, model, 0)


def expected_layer0_attention_from_baseline_hf(
    globs: Globals,
    model: Any,
) -> dict[str, torch.Tensor]:
    return expected_layer_attention_from_baseline_hf(globs, model, 0)


def expected_layer0_shared_expert_from_baseline_hf(
    globs: Globals,
    model: Any,
) -> dict[str, torch.Tensor]:
    layer = model.model.layers[0]
    hidden = expected_layer_o_proj_from_baseline_hf(globs, model, 0)["hidden_states"]
    normed = layer.post_attention_layernorm(hidden.view(1, 1, -1))
    shared = layer.mlp.shared_expert(normed)[0, 0]
    return {"shared_out": shared.to(globs.shared_out.dtype)}


def expected_layer0_full_from_baseline_hf(
    globs: Globals,
    model: Any,
) -> dict[str, torch.Tensor]:
    return expected_layer_full_from_baseline_hf(globs, model, 0)


def advance_globs_by_hf_layer(globs: Globals, model: Any, layer_idx: int) -> None:
    qkv = expected_layer_from_baseline_hf(globs, model, layer_idx)
    hidden = expected_layer_full_from_baseline_hf(globs, model, layer_idx)["hidden_states"]
    globs.k_cache.copy_(qkv["k_cache"])
    globs.v_cache.copy_(qkv["v_cache"])
    globs.hidden_states.copy_(hidden)


def expected_logits_same_formula(globs: Globals) -> dict[str, torch.Tensor]:
    normed = _rms_norm(globs.hidden_states.reshape(-1), globs.lm_head_norm_weights)
    logits = F.linear(normed.unsqueeze(0), globs.lm_head_weights).squeeze(0)
    return {"logits": logits.to(globs.logits.dtype)}


def run_baseline_decode_step(globs: Globals, model: Any) -> torch.Tensor:
    for layer_idx in range(QwenMoeConfig.num_hidden_layers):
        advance_globs_by_hf_layer(globs, model, layer_idx)
    return expected_logits_same_formula(globs)["logits"]
