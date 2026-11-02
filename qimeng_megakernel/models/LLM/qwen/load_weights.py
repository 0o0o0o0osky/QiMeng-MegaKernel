"""
Load Qwen2/Qwen2.5/Qwen3 weights from disk (config + safetensors) in the same layout as
_stack_qwen2_weights_from_hf / _stack_qwen3_weights_from_hf.
No HuggingFace model instantiation. Used by A100 fused prefill to avoid loading the full HF model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.llama.preprocess import (
    _interleave_qk_weight_indices,
    _interleave_rope_cos_sin,
    _one_head_interleave_indices,
)


def load_qwen2_config(model_path: str) -> dict[str, Any]:
    """Load config.json from model dir."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    config_file = path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_file) as f:
        return json.load(f)


def _load_safetensors_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load full state dict: single model.safetensors or sharded via index."""
    from safetensors.torch import load_file

    path = Path(model_path)
    single = path / "model.safetensors"
    if single.exists():
        return load_file(str(single), device="cpu")
    index_file = path / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {model_path}"
        )
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


def _compute_rope_cos_sin(
    rope_theta: float,
    head_dim: int,
    max_position_embeddings: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin [max_pos, head_dim]. Same math as HF."""
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


def _compute_rope_cos_sin_interleaved(
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rope_cos, rope_sin in interleaved layout (same as _qwen_moe_weights_from_hf / Qwen2)."""
    head_dim = config.get("head_dim") or (config["hidden_size"] // config["num_attention_heads"])
    max_pos = config.get("max_position_embeddings", 32768)
    rope_params = config.get("rope_parameters") or {}
    rope_theta = float(rope_params.get("rope_theta") or config.get("rope_theta", 1000000.0))
    cos, sin = _compute_rope_cos_sin(
        rope_theta=rope_theta,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        device=device,
        dtype=dtype,
    )
    rope_cos, rope_sin = _interleave_rope_cos_sin(cos, sin, head_dim)
    return rope_cos.contiguous(), rope_sin.contiguous()


def load_qwen2_weights_from_disk(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Load config and weights from disk in the same layout as _stack_qwen2_weights_from_hf (no HF model).
    Returns (config_dict, weights_dict). Includes qkv_bias when present (Qwen2.5).
    """
    config = load_qwen2_config(model_path)
    state = _load_safetensors_state_dict(model_path)

    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim") or (config["hidden_size"] // num_attention_heads)
    num_hidden_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    qkv_dim = (
        num_attention_heads * head_dim
        + 2 * num_key_value_heads * head_dim
    )
    vocab_size = config["vocab_size"]

    indices_for_q, indices_for_k = _interleave_qk_weight_indices(
        num_attention_heads, num_key_value_heads, head_dim, device
    )

    embed_tokens = state["model.embed_tokens.weight"].to(device=device, dtype=dtype).contiguous()
    if config.get("tie_word_embeddings", False):
        lm_head_weights = embed_tokens
    else:
        lm_head_key = (
            "lm_head.weight"
            if "lm_head.weight" in state
            else "model.lm_head.weight"
        )
        lm_head_weights = state[lm_head_key].to(
            device=device, dtype=dtype
        ).contiguous()

    stacked_qkv = torch.empty((num_hidden_layers, qkv_dim, hidden_size), device=device, dtype=dtype)
    stacked_qkv_bias = []
    stacked_o = torch.empty(
        (num_hidden_layers, hidden_size, num_attention_heads * head_dim),
        device=device,
        dtype=dtype,
    )
    stacked_attn_ln = torch.empty((num_hidden_layers, hidden_size), device=device, dtype=dtype)
    stacked_mlp_ln = torch.empty((num_hidden_layers, hidden_size), device=device, dtype=dtype)
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
        qb_key = f"{prefix}.self_attn.q_proj.bias"
        if qb_key in state:
            qb = state[qb_key].to(device=device, dtype=dtype)[indices_for_q]
            kb = state[f"{prefix}.self_attn.k_proj.bias"].to(device=device, dtype=dtype)[indices_for_k]
            vb = state[f"{prefix}.self_attn.v_proj.bias"].to(device=device, dtype=dtype)
            stacked_qkv_bias.append(torch.cat([qb, kb, vb], dim=0))
        else:
            stacked_qkv_bias.append(None)
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

    rope_cos, rope_sin = _compute_rope_cos_sin_interleaved(config, device, dtype=torch.float32)

    qkv_bias = None
    if all(b is not None for b in stacked_qkv_bias):
        qkv_bias = torch.stack(stacked_qkv_bias, dim=0).contiguous()

    weights = {
        "qkv_weights": stacked_qkv.contiguous(),
        "qkv_bias": qkv_bias,
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


def load_qwen3_weights_from_disk(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Load config and weights from disk in the same layout as _stack_qwen3_weights_from_hf (no HF model).
    Returns (config_dict, weights_dict). Includes q_norm_weights and k_norm_weights (Qwen3-specific).
    No qkv_bias (Qwen3 has attention_bias=False). tie_word_embeddings=True: lm_head shares embed_tokens.
    """
    config = load_qwen2_config(model_path)
    state = _load_safetensors_state_dict(model_path)

    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim") or (config["hidden_size"] // num_attention_heads)
    num_hidden_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    qkv_dim = (
        num_attention_heads * head_dim
        + 2 * num_key_value_heads * head_dim
    )

    indices_for_q, indices_for_k = _interleave_qk_weight_indices(
        num_attention_heads, num_key_value_heads, head_dim, device
    )
    one_head_idx = _one_head_interleave_indices(head_dim, device)

    embed_tokens = state["model.embed_tokens.weight"].to(device=device, dtype=dtype).contiguous()
    if config.get("tie_word_embeddings", False):
        lm_head_weights = embed_tokens
    else:
        lm_head_key = (
            "lm_head.weight"
            if "lm_head.weight" in state
            else "model.lm_head.weight"
        )
        lm_head_weights = state[lm_head_key].to(
            device=device, dtype=dtype
        ).contiguous()

    stacked_qkv = torch.empty((num_hidden_layers, qkv_dim, hidden_size), device=device, dtype=dtype)
    stacked_q_norm = torch.empty((num_hidden_layers, head_dim), device=device, dtype=dtype)
    stacked_k_norm = torch.empty((num_hidden_layers, head_dim), device=device, dtype=dtype)
    stacked_o = torch.empty(
        (num_hidden_layers, hidden_size, num_attention_heads * head_dim),
        device=device,
        dtype=dtype,
    )
    stacked_attn_ln = torch.empty((num_hidden_layers, hidden_size), device=device, dtype=dtype)
    stacked_mlp_ln = torch.empty((num_hidden_layers, hidden_size), device=device, dtype=dtype)
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

        q_norm_w = state[f"{prefix}.self_attn.q_norm.weight"].to(device=device, dtype=dtype)[one_head_idx]
        k_norm_w = state[f"{prefix}.self_attn.k_norm.weight"].to(device=device, dtype=dtype)[one_head_idx]
        stacked_q_norm[i].copy_(q_norm_w)
        stacked_k_norm[i].copy_(k_norm_w)

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

    rope_cos, rope_sin = _compute_rope_cos_sin_interleaved(config, device, dtype=torch.float32)

    weights = {
        "qkv_weights": stacked_qkv.contiguous(),
        "q_norm_weights": stacked_q_norm.contiguous(),
        "k_norm_weights": stacked_k_norm.contiguous(),
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


def load_qwen_moe_weights_from_disk(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Load Qwen2 MoE config and weights from disk in the same layout as
    _qwen_moe_weights_from_hf (no HF model). Returns (config_dict, weights_dict).
    """
    config = load_qwen2_config(model_path)
    state = _load_safetensors_state_dict(model_path)

    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim") or (
        config["hidden_size"] // config["num_attention_heads"]
    )
    num_hidden_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    num_experts = config.get("num_experts", 60)
    moe_intermediate_size = config.get("moe_intermediate_size", 1408)
    shared_intermediate_size = config.get(
        "shared_expert_intermediate_size",
        config.get("intermediate_size", 5632),
    )
    qkv_dim = (
        num_attention_heads * head_dim
        + 2 * num_key_value_heads * head_dim
    )

    indices_for_q, indices_for_k = _interleave_qk_weight_indices(
        num_attention_heads, num_key_value_heads, head_dim, device
    )

    embed_tokens = state["model.embed_tokens.weight"].to(
        device=device, dtype=dtype
    ).contiguous()
    if config.get("tie_word_embeddings", False):
        lm_head_weights = embed_tokens
    else:
        lm_head_key = (
            "lm_head.weight"
            if "lm_head.weight" in state
            else "model.lm_head.weight"
        )
        if lm_head_key not in state:
            raise KeyError("Neither lm_head.weight nor model.lm_head.weight found")
        lm_head_weights = state[lm_head_key].to(device=device, dtype=dtype).contiguous()

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
    stacked_router = torch.empty(
        (num_hidden_layers, num_experts, hidden_size),
        device=device,
        dtype=dtype,
    )
    stacked_shared_gate_proj = torch.empty(
        (num_hidden_layers, shared_intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    stacked_shared_up_proj = torch.empty(
        (num_hidden_layers, shared_intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    stacked_shared_down_proj = torch.empty(
        (num_hidden_layers, hidden_size, shared_intermediate_size),
        device=device,
        dtype=dtype,
    )
    stacked_shared_gate = torch.empty(
        (num_hidden_layers, 1, hidden_size), device=device, dtype=dtype
    )
    stacked_expert_gate_up = torch.empty(
        (
            num_hidden_layers,
            num_experts,
            2 * moe_intermediate_size,
            hidden_size,
        ),
        device=device,
        dtype=dtype,
    )
    stacked_expert_down = torch.empty(
        (num_hidden_layers, num_experts, hidden_size, moe_intermediate_size),
        device=device,
        dtype=dtype,
    )

    has_qkv_bias = f"model.layers.0.self_attn.q_proj.bias" in state
    stacked_qkv_bias = None
    if has_qkv_bias:
        stacked_qkv_bias = torch.empty(
            (num_hidden_layers, qkv_dim), device=device, dtype=dtype
        )

    for i in range(num_hidden_layers):
        prefix = f"model.layers.{i}"
        q = state[f"{prefix}.self_attn.q_proj.weight"].to(
            device=device, dtype=dtype
        )[indices_for_q]
        k = state[f"{prefix}.self_attn.k_proj.weight"].to(
            device=device, dtype=dtype
        )[indices_for_k]
        v = state[f"{prefix}.self_attn.v_proj.weight"].to(
            device=device, dtype=dtype
        )
        stacked_qkv[i].copy_(torch.cat([q, k, v], dim=0))
        if stacked_qkv_bias is not None:
            qb = state[f"{prefix}.self_attn.q_proj.bias"].to(
                device=device, dtype=dtype
            )[indices_for_q]
            kb = state[f"{prefix}.self_attn.k_proj.bias"].to(
                device=device, dtype=dtype
            )[indices_for_k]
            vb = state[f"{prefix}.self_attn.v_proj.bias"].to(
                device=device, dtype=dtype
            )
            stacked_qkv_bias[i].copy_(torch.cat([qb, kb, vb], dim=0))
        stacked_o[i].copy_(
            state[f"{prefix}.self_attn.o_proj.weight"].to(
                device=device, dtype=dtype
            )
        )
        stacked_attn_ln[i].copy_(
            state[f"{prefix}.input_layernorm.weight"].to(
                device=device, dtype=dtype
            )
        )
        stacked_mlp_ln[i].copy_(
            state[f"{prefix}.post_attention_layernorm.weight"].to(
                device=device, dtype=dtype
            )
        )
        stacked_router[i].copy_(
            state[f"{prefix}.mlp.gate.weight"].to(
                device=device, dtype=dtype
            )
        )
        stacked_shared_gate_proj[i].copy_(
            state[f"{prefix}.mlp.shared_expert.gate_proj.weight"].to(
                device=device, dtype=dtype
            )
        )
        stacked_shared_up_proj[i].copy_(
            state[f"{prefix}.mlp.shared_expert.up_proj.weight"].to(
                device=device, dtype=dtype
            )
        )
        stacked_shared_down_proj[i].copy_(
            state[f"{prefix}.mlp.shared_expert.down_proj.weight"].to(
                device=device, dtype=dtype
            )
        )
        stacked_shared_gate[i].copy_(
            state[f"{prefix}.mlp.shared_expert_gate.weight"].to(
                device=device, dtype=dtype
            )
        )
        # Support both stacked (gate_up_proj/down_proj) and per-expert (experts.{e}.gate_proj/up_proj/down_proj) layout
        if f"{prefix}.mlp.experts.gate_up_proj" in state:
            stacked_expert_gate_up[i].copy_(
                state[f"{prefix}.mlp.experts.gate_up_proj"].to(
                    device=device, dtype=dtype
                )
            )
            stacked_expert_down[i].copy_(
                state[f"{prefix}.mlp.experts.down_proj"].to(
                    device=device, dtype=dtype
                )
            )
        else:
            for e in range(num_experts):
                gate = state[f"{prefix}.mlp.experts.{e}.gate_proj.weight"].to(
                    device=device, dtype=dtype
                )
                up = state[f"{prefix}.mlp.experts.{e}.up_proj.weight"].to(
                    device=device, dtype=dtype
                )
                stacked_expert_gate_up[i, e].copy_(torch.cat([gate, up], dim=0))
                stacked_expert_down[i, e].copy_(
                    state[f"{prefix}.mlp.experts.{e}.down_proj.weight"].to(
                        device=device, dtype=dtype
                    )
                )

    rope_cos, rope_sin = _compute_rope_cos_sin_interleaved(
        config, device, dtype=torch.float32
    )

    weights = {
        "qkv_weights": stacked_qkv.contiguous(),
        "qkv_bias": stacked_qkv_bias.contiguous() if stacked_qkv_bias is not None else None,
        "o_proj_weights": stacked_o.contiguous(),
        "attn_norm_weights": stacked_attn_ln.contiguous(),
        "mlp_norm_weights": stacked_mlp_ln.contiguous(),
        "router_weights": stacked_router.contiguous(),
        "shared_expert_gate_proj_weights": stacked_shared_gate_proj.contiguous(),
        "shared_expert_up_proj_weights": stacked_shared_up_proj.contiguous(),
        "shared_expert_down_proj_weights": stacked_shared_down_proj.contiguous(),
        "shared_expert_gate_weights": stacked_shared_gate.contiguous(),
        "expert_gate_up_proj_weights": stacked_expert_gate_up.contiguous(),
        "expert_down_proj_weights": stacked_expert_down.contiguous(),
        "lm_head_norm_weights": state["model.norm.weight"]
        .to(device=device, dtype=dtype)
        .contiguous(),
        "lm_head_weights": lm_head_weights,
        "rope_cos": rope_cos.contiguous(),
        "rope_sin": rope_sin.contiguous(),
        "embed_tokens": embed_tokens,
    }
    return config, weights
