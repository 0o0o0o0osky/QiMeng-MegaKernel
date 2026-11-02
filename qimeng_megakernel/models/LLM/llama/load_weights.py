"""
Load Llama weights from disk (configuration and safetensors) in the reordered layout.
No Hugging Face model instantiation: one copy of weights, interleaved Q/K and RoPE at load time.

Output layout is the same as _stack_llama_weights_from_hf (llama/preprocess.py): Q/K weight
indices and RoPE cos/sin use the same interleave pattern. Shared by A100 and H100
decode/fused paths; H100 preprocess uses _stack_llama_weights_from_hf with this layout, so
weights from this module are compatible with H100 when building globs from disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from .preprocess import (
    _interleave_qk_weight_indices,
    _interleave_rope_cos_sin,
)


def load_llama_config(model_path: str) -> dict[str, Any]:
    """Load config.json from a local model directory."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    config_file = path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_file) as f:
        return json.load(f)


def _compute_rope_cos_sin(
    rope_theta: float,
    head_dim: int,
    max_position_embeddings: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return standard-layout RoPE tables matching the official framework reference."""
    dim = head_dim
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    position_ids = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(position_ids, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def compute_rope_cos_sin_interleaved(
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cosine and sine tensors in the shared interleaved layout."""
    head_dim = config.get("head_dim") or (config["hidden_size"] // config["num_attention_heads"])
    max_pos = config.get("max_position_embeddings", 2048)
    rope_theta = float(config.get("rope_theta", 10000.0))
    cos, sin = _compute_rope_cos_sin(
        rope_theta=rope_theta,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        device=device,
        dtype=dtype,
    )
    rope_cos, rope_sin = _interleave_rope_cos_sin(cos, sin, head_dim)
    return rope_cos.contiguous(), rope_sin.contiguous()


def _load_safetensors_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load full state dict from model_path: single model.safetensors or sharded via index."""
    from safetensors.torch import load_file

    path = Path(model_path)
    single = path / "model.safetensors"
    if single.exists():
        return load_file(str(single), device="cpu")
    index_file = path / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json in {model_path}")
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    state_dict = {}
    for shard in shards:
        shard_path = path / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        state_dict.update(load_file(str(shard_path), device="cpu"))
    return state_dict


def load_llama_weights_from_disk(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Load configuration and weights from disk in the reordered layout without
    instantiating the official framework model. Returns configuration and weight
    dictionaries with all weights on the target device for MegaKernel and fused decode.
    Q/K are reordered with _interleave_qk_weight_indices; rope_cos/sin are interleaved.
    """
    config = load_llama_config(model_path)
    state = _load_safetensors_state_dict(model_path)

    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim") or (config["hidden_size"] // num_attention_heads)
    num_hidden_layers = config["num_hidden_layers"]

    indices_for_q, indices_for_k = _interleave_qk_weight_indices(
        num_attention_heads, num_key_value_heads, head_dim, device
    )

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    qkv_dim = (
        num_attention_heads * head_dim
        + 2 * num_key_value_heads * head_dim
    )
    vocab_size = config["vocab_size"]

    # Allocate the largest blocks first to reduce fragmentation. The embedding
    # and language-model head each use approximately 768 MiB.
    embed_tokens_pre = torch.empty((vocab_size, hidden_size), device=device, dtype=dtype)
    embed_tokens_pre.copy_(state["model.embed_tokens.weight"].to(device=device, dtype=dtype))
    if config.get("tie_word_embeddings", False):
        lm_head_weights_pre = embed_tokens_pre
    else:
        lm_head_weights_pre = torch.empty((vocab_size, hidden_size), device=device, dtype=dtype)
        lm_head_weights_pre.copy_(state["model.lm_head.weight"].to(device=device, dtype=dtype))

    # Preallocate stacked tensors and populate them one layer at a time.
    stacked_qkv = torch.empty(
        (num_hidden_layers, qkv_dim, hidden_size), device=device, dtype=dtype
    )
    stacked_o = torch.empty(
        (num_hidden_layers, hidden_size, num_attention_heads * head_dim),
        device=device,
        dtype=dtype,
    )
    stacked_attn_ln = torch.empty(
        (num_hidden_layers, hidden_size), device=device, dtype=dtype
    )
    stacked_mlp_ln = torch.empty(
        (num_hidden_layers, hidden_size), device=device, dtype=dtype
    )
    stacked_up = torch.empty(
        (num_hidden_layers, intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    stacked_gate = torch.empty(
        (num_hidden_layers, intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    stacked_down = torch.empty(
        (num_hidden_layers, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
    )

    for i in range(num_hidden_layers):
        prefix = f"model.layers.{i}"
        q = state[f"{prefix}.self_attn.q_proj.weight"].to(device=device, dtype=dtype)[indices_for_q]
        k = state[f"{prefix}.self_attn.k_proj.weight"].to(device=device, dtype=dtype)[indices_for_k]
        v = state[f"{prefix}.self_attn.v_proj.weight"].to(device=device, dtype=dtype)
        stacked_qkv[i].copy_(torch.cat([q, k, v], dim=0))
        stacked_o[i].copy_(state[f"{prefix}.self_attn.o_proj.weight"].to(device=device, dtype=dtype))
        stacked_attn_ln[i].copy_(
            state[f"{prefix}.input_layernorm.weight"].to(device=device, dtype=dtype)
        )
        stacked_mlp_ln[i].copy_(
            state[f"{prefix}.post_attention_layernorm.weight"].to(device=device, dtype=dtype)
        )
        stacked_up[i].copy_(
            state[f"{prefix}.mlp.up_proj.weight"].to(device=device, dtype=dtype)
        )
        stacked_gate[i].copy_(
            state[f"{prefix}.mlp.gate_proj.weight"].to(device=device, dtype=dtype)
        )
        stacked_down[i].copy_(
            state[f"{prefix}.mlp.down_proj.weight"].to(device=device, dtype=dtype)
        )

    # Keep rope in float32: MK C++ expects .data_ptr<float>(); fused/Python can use float32 rope.
    rope_cos, rope_sin = compute_rope_cos_sin_interleaved(config, device, dtype=torch.float32)

    lm_head_weights = lm_head_weights_pre
    embed_tokens = embed_tokens_pre

    weights = {
        "qkv_weights": stacked_qkv.contiguous(),
        "o_proj_weights": stacked_o.contiguous(),
        "attn_norm_weights": stacked_attn_ln.contiguous(),
        "mlp_norm_weights": stacked_mlp_ln.contiguous(),
        "up_proj_weights": stacked_up.contiguous(),
        "gate_proj_weights": stacked_gate.contiguous(),
        "down_proj_weights": stacked_down.contiguous(),
        "lm_head_norm_weights": state["model.norm.weight"].to(device=device, dtype=dtype).contiguous(),
        "lm_head_weights": lm_head_weights,
        "rope_cos": rope_cos.contiguous(),
        "rope_sin": rope_sin.contiguous(),
        "embed_tokens": embed_tokens,
    }
    return config, weights
