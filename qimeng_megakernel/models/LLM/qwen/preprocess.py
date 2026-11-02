"""
Qwen2 decode preprocess: shared helpers for stacking weights from HF and RoPE interleaving.
Qwen2 (Qwen2ForCausalLM) uses the same layer layout as LLaMA: model.model.layers[].self_attn
(q/k/v/o_proj), input_layernorm, post_attention_layernorm, mlp (up/gate/down_proj).
Reuses interleave helpers from llama.preprocess for KV cache and RoPE.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM

from qimeng_megakernel.models.LLM.preprocess import (
    DecodePreprocessOutput,
    prepare_kv_cache_for_decode,
)
from qimeng_megakernel.models.LLM.llama.preprocess import (
    _interleave_k_cache,
    _interleave_qk_weight_indices,
    _interleave_rope_cos_sin,
)


def _stack_qwen2_weights_from_hf(model: Any) -> dict[str, torch.Tensor]:
    """Stack per-layer weights from HuggingFace Qwen2ForCausalLM (same layout as LLaMA)."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("HF model does not expose expected Qwen2 model.model.layers.")

    config = model.config
    num_attention_heads = getattr(config, "num_attention_heads", config.num_attention_heads)
    num_key_value_heads = getattr(config, "num_key_value_heads", config.num_key_value_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_attention_heads)
    device = next(model.parameters()).device
    indices_for_q, indices_for_k = _interleave_qk_weight_indices(
        num_attention_heads, num_key_value_heads, head_dim, device
    )

    layers = model.model.layers
    qkv_weights = []
    qkv_biases = []  # Qwen2.5: q/k/v_proj have bias
    o_proj_weights = []
    attn_norm_weights = []
    mlp_norm_weights = []
    up_proj_weights = []
    gate_proj_weights = []
    down_proj_weights = []

    for layer in layers:
        q = layer.self_attn.q_proj.weight
        k = layer.self_attn.k_proj.weight
        v = layer.self_attn.v_proj.weight
        q = q[indices_for_q]
        k = k[indices_for_k]
        qkv_weights.append(torch.cat([q, k, v], dim=0))
        # Qwen2.5: q_proj, k_proj, v_proj have bias=True; o_proj has bias=False
        qb = getattr(layer.self_attn.q_proj, "bias", None)
        kb = getattr(layer.self_attn.k_proj, "bias", None)
        vb = getattr(layer.self_attn.v_proj, "bias", None)
        if qb is not None and kb is not None and vb is not None:
            qb = qb[indices_for_q]
            kb = kb[indices_for_k]
            qkv_biases.append(torch.cat([qb, kb, vb], dim=0))
        else:
            qkv_biases.append(None)
        o_proj_weights.append(layer.self_attn.o_proj.weight)
        attn_norm_weights.append(layer.input_layernorm.weight)
        mlp_norm_weights.append(layer.post_attention_layernorm.weight)
        up_proj_weights.append(layer.mlp.up_proj.weight)
        gate_proj_weights.append(layer.mlp.gate_proj.weight)
        down_proj_weights.append(layer.mlp.down_proj.weight)

    qkv_weights = torch.stack(qkv_weights, dim=0)
    if all(b is not None for b in qkv_biases):
        qkv_bias = torch.stack(qkv_biases, dim=0)
    else:
        qkv_bias = None
    o_proj_weights = torch.stack(o_proj_weights, dim=0)
    attn_norm_weights = torch.stack(attn_norm_weights, dim=0)
    mlp_norm_weights = torch.stack(mlp_norm_weights, dim=0)
    up_proj_weights = torch.stack(up_proj_weights, dim=0)
    gate_proj_weights = torch.stack(gate_proj_weights, dim=0)
    down_proj_weights = torch.stack(down_proj_weights, dim=0)

    lm_head_norm_weights = model.model.norm.weight
    lm_head_weights = model.lm_head.weight

    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is None and layers:
        rotary_emb = getattr(layers[0].self_attn, "rotary_emb", None)
    if rotary_emb is None:
        raise ValueError("Unable to locate rotary embedding on HF Qwen2 model.")

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    max_pos = config.max_position_embeddings
    position_ids = torch.arange(max_pos, device=lm_head_weights.device).unsqueeze(0)
    dummy = torch.empty(
        (1, max_pos, head_dim),
        device=lm_head_weights.device,
        dtype=torch.float32,
    )
    cos, sin = rotary_emb(dummy, position_ids)
    rope_cos = cos.squeeze(0).to(torch.float32)
    rope_sin = sin.squeeze(0).to(torch.float32)
    rope_cos, rope_sin = _interleave_rope_cos_sin(rope_cos, rope_sin, head_dim)

    return {
        "qkv_weights": qkv_weights,
        "qkv_bias": qkv_bias,
        "o_proj_weights": o_proj_weights,
        "attn_norm_weights": attn_norm_weights,
        "mlp_norm_weights": mlp_norm_weights,
        "up_proj_weights": up_proj_weights,
        "gate_proj_weights": gate_proj_weights,
        "down_proj_weights": down_proj_weights,
        "lm_head_norm_weights": lm_head_norm_weights,
        "lm_head_weights": lm_head_weights,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }


def _load_hf_qwen2_model(
    model_name: str, *, device: torch.device, dtype: torch.dtype
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model
