"""Shared Llama decode preprocessing helpers and hardware-specific dispatch.

``preprocess_decode(..., hardware=...)`` selects the implementation for the
requested model and hardware. Shared helpers remain here for the A100 and H100
production routes and their verifiers.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM

from qimeng_megakernel.models.LLM.preprocess import (
    DecodePreprocessOutput,
    prepare_kv_cache_for_decode,
)

# Model- and hardware-specific implementations live in their local preprocess
# modules. This module provides shared helpers and interface dispatch.
def _one_head_interleave_indices(head_dim: int, device: torch.device) -> torch.Tensor:
    """Indices to reorder one head's head_dim to interleaved (same as _interleave_rope_cos_sin)."""
    half = head_dim // 2
    indices: list[int] = []
    for i in range(half):
        indices.append(i)
        indices.append(i + half)
    return torch.tensor(indices, device=device, dtype=torch.long)


def _interleave_rope_cos_sin(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reorder rope_cos/rope_sin to the interleaved layout the decode kernels expect.
    one_head_indices = [0, half, 1, half+1, ..., half-1, head_dim-1] so that
    apply_rotary_pos_emb_interleaved sees the same cos/sin per dimension pair.
    """
    one_head_indices = _one_head_interleave_indices(head_dim, rope_cos.device)
    rope_cos = rope_cos[..., one_head_indices].contiguous()
    rope_sin = rope_sin[..., one_head_indices].contiguous()
    return rope_cos, rope_sin


def _interleave_k_cache(k_cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Reorder key cache last dim to interleaved layout so fused attention (interleaved q) matches.
    k_cache: [L, B, S, H, D]; D uses the _interleave_rope_cos_sin permutation.
    """
    # k_cache is [L, B, S, H, D]; reorder last dim per head
    idx = _one_head_interleave_indices(head_dim, k_cache.device)
    return k_cache[..., idx].contiguous()


def _interleave_qk_weight_indices(
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Indices to reorder Q/K proj output dims to the interleaved layout.
    The interleaved tables and ``apply_rotary_pos_emb_interleaved`` match the
    official framework reference RoPE semantics.
    """
    half_head_dim = head_dim // 2
    indices_for_q_list: list[int] = []
    for n in range(num_attention_heads):
        offset = n * head_dim
        for i in range(half_head_dim):
            indices_for_q_list.append(offset + i)
            indices_for_q_list.append(offset + half_head_dim + i)
    indices_for_q = torch.tensor(indices_for_q_list, device=device, dtype=torch.long)
    indices_for_k_list: list[int] = []
    for n in range(num_key_value_heads):
        offset = n * head_dim
        for i in range(half_head_dim):
            indices_for_k_list.append(offset + i)
            indices_for_k_list.append(offset + half_head_dim + i)
    indices_for_k = torch.tensor(indices_for_k_list, device=device, dtype=torch.long)
    return indices_for_q, indices_for_k


def _stack_llama_weights_from_hf(model: Any) -> dict[str, torch.Tensor]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Hugging Face model does not expose the expected Llama layers.")

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
    o_proj_weights = []
    attn_norm_weights = []
    mlp_norm_weights = []
    up_proj_weights = []
    gate_proj_weights = []
    down_proj_weights = []

    for layer in layers:
        q = layer.self_attn.q_proj.weight  # (num_heads * head_dim, hidden)
        k = layer.self_attn.k_proj.weight  # (num_kv_heads * head_dim, hidden)
        v = layer.self_attn.v_proj.weight
        # Reorder Q/K output dims to interleaved (same permutation as the interleaved rope tables above)
        q = q[indices_for_q]
        k = k[indices_for_k]
        qkv_weights.append(torch.cat([q, k, v], dim=0))
        o_proj_weights.append(layer.self_attn.o_proj.weight)
        attn_norm_weights.append(layer.input_layernorm.weight)
        mlp_norm_weights.append(layer.post_attention_layernorm.weight)
        up_proj_weights.append(layer.mlp.up_proj.weight)
        gate_proj_weights.append(layer.mlp.gate_proj.weight)
        down_proj_weights.append(layer.mlp.down_proj.weight)

    qkv_weights = torch.stack(qkv_weights, dim=0)
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
        raise ValueError("Unable to locate rotary embedding on the Hugging Face model.")

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    max_pos = model.config.max_position_embeddings
    position_ids = torch.arange(max_pos, device=lm_head_weights.device).unsqueeze(0)
    dummy = torch.empty(
        (1, max_pos, head_dim),
        device=lm_head_weights.device,
        dtype=torch.float32,
    )
    cos, sin = rotary_emb(dummy, position_ids)
    rope_cos = cos.squeeze(0).to(torch.float32)
    rope_sin = sin.squeeze(0).to(torch.float32)

    # Reorder to interleaved layout so
    # apply_rotary_pos_emb_interleaved uses the same cos/sin per (q,k) dimension pair.
    rope_cos, rope_sin = _interleave_rope_cos_sin(rope_cos, rope_sin, head_dim)

    return {
        "qkv_weights": qkv_weights,
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


def _load_hf_llama_model(
    model_name: str, *, device: torch.device, dtype: torch.dtype
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _get_sm_count(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1


def _tensor_info(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} stride={t.stride()}"
    )


def preprocess_decode(
    *,
    prefill,
    model: Any | None = None,
    model_name: str | None = None,
    max_new_tokens: int | None = None,
    debug: bool = False,
    **_kwargs,
):
    """Dispatch to the 1B or 3B implementation for A100 or H100."""
    import importlib
    name = (model_name or getattr(model, "name_or_path", "") or "").lower()
    hw = _kwargs.get("hardware", "A100")
    if "3b" in name or "llama-3.2-3b" in name:
        # 3B: dispatch to the model-local hardware implementation.
        if hw == "H100":
            mod = importlib.import_module(
                "qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.preprocess"
            )
        else:
            mod = importlib.import_module(
                "qimeng_megakernel.models.LLM.llama.llama3_2__3b.A100.preprocess"
            )
    else:
        if hw == "H100":
            mod = importlib.import_module(
                "qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.preprocess"
            )
        else:
            mod = importlib.import_module(
                "qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.preprocess"
            )
    return mod.preprocess_decode(
        prefill=prefill,
        model=model,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        debug=debug,
        **_kwargs,
    )
