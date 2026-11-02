"""
Qwen3 decode preprocess: weight stacking helpers for Qwen3ForCausalLM.

Key differences from Qwen2 preprocess:
- No QKV bias (attention_bias=False in Qwen3)
- Added q_norm_weights and k_norm_weights (per-layer per-head RMSNorm, shape [L, head_dim])
  stored in interleaved format to match the interleaved QKV weight layout
- tie_word_embeddings=True: lm_head.weight shares embed_tokens.weight
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM

from qimeng_megakernel.models.LLM.llama.preprocess import (
    _interleave_k_cache,
    _interleave_qk_weight_indices,
    _interleave_rope_cos_sin,
    _one_head_interleave_indices,
)


def _stack_qwen3_weights_from_hf(model: Any) -> dict[str, torch.Tensor]:
    """Stack per-layer weights from HuggingFace Qwen3ForCausalLM.

    Qwen3-specific: q_norm/k_norm per head (interleaved), no qkv_bias.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("HF model does not expose expected Qwen3 model.model.layers.")

    config = model.config
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_attention_heads)
    device = next(model.parameters()).device

    indices_for_q, indices_for_k = _interleave_qk_weight_indices(
        num_attention_heads, num_key_value_heads, head_dim, device
    )
    # Per-element interleave index within one head: used to reorder q/k_norm weights
    one_head_idx = _one_head_interleave_indices(head_dim, device)

    layers = model.model.layers
    qkv_weights = []
    q_norm_weights = []
    k_norm_weights = []
    o_proj_weights = []
    attn_norm_weights = []
    mlp_norm_weights = []
    up_proj_weights = []
    gate_proj_weights = []
    down_proj_weights = []

    for layer in layers:
        q_w = layer.self_attn.q_proj.weight
        k_w = layer.self_attn.k_proj.weight
        v_w = layer.self_attn.v_proj.weight
        q_w = q_w[indices_for_q]
        k_w = k_w[indices_for_k]
        qkv_weights.append(torch.cat([q_w, k_w, v_w], dim=0))

        # q_norm / k_norm weights: stored in interleaved order so fused op applies them correctly
        q_norm_w = layer.self_attn.q_norm.weight[one_head_idx].clone()
        k_norm_w = layer.self_attn.k_norm.weight[one_head_idx].clone()
        q_norm_weights.append(q_norm_w)
        k_norm_weights.append(k_norm_w)

        o_proj_weights.append(layer.self_attn.o_proj.weight)
        attn_norm_weights.append(layer.input_layernorm.weight)
        mlp_norm_weights.append(layer.post_attention_layernorm.weight)
        up_proj_weights.append(layer.mlp.up_proj.weight)
        gate_proj_weights.append(layer.mlp.gate_proj.weight)
        down_proj_weights.append(layer.mlp.down_proj.weight)

    qkv_weights = torch.stack(qkv_weights, dim=0)
    q_norm_weights = torch.stack(q_norm_weights, dim=0)   # [L, head_dim]
    k_norm_weights = torch.stack(k_norm_weights, dim=0)   # [L, head_dim]
    o_proj_weights = torch.stack(o_proj_weights, dim=0)
    attn_norm_weights = torch.stack(attn_norm_weights, dim=0)
    mlp_norm_weights = torch.stack(mlp_norm_weights, dim=0)
    up_proj_weights = torch.stack(up_proj_weights, dim=0)
    gate_proj_weights = torch.stack(gate_proj_weights, dim=0)
    down_proj_weights = torch.stack(down_proj_weights, dim=0)

    lm_head_norm_weights = model.model.norm.weight
    # tie_word_embeddings=True: lm_head.weight == embed_tokens.weight (same storage)
    lm_head_weights = model.lm_head.weight

    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is None and layers:
        rotary_emb = getattr(layers[0].self_attn, "rotary_emb", None)
    if rotary_emb is None:
        raise ValueError("Unable to locate rotary embedding on HF Qwen3 model.")

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
        "q_norm_weights": q_norm_weights,
        "k_norm_weights": k_norm_weights,
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


def _load_hf_qwen3_model(
    model_name: str, *, device: torch.device, dtype: torch.dtype
) -> Any:
    import os
    kwargs = {"torch_dtype": dtype}
    if os.path.isdir(os.path.expanduser(model_name)):
        kwargs["local_files_only"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    model.eval()
    model.requires_grad_(False)
    return model
