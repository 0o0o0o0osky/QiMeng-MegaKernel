"""Qwen2 MoE weights from HF model; same layout for prefill and decode (attention + MoE).

Memory discipline (L1, 2026-07-29)
---------------------------------
The routed-expert stacks are huge for Qwen1.5-MoE-A2.7B:
  expert_gate_up_proj_weights [24, 60, 2816, 2048] bf16 = 15.5 GiB
  expert_down_proj_weights    [24, 60, 2048, 1408] bf16 =  7.7 GiB
The previous implementation built them with a two-level ``torch.stack`` (per-layer
stack of 60 experts, then a final stack of 24 layers).  Both levels keep the whole
intermediate list alive while the destination is being allocated, so the final stack
momentarily needed 2x the routed-expert bytes *on top of* the still-live HF model
(~26.7 GiB) -> measured peak 68.36 GiB on an 80 GB H100, and OOM once anything else
(KV cache, a second test parametrisation) shared the device.

Every stacked tensor is now pre-allocated once with ``torch.empty`` and filled by
streaming ``copy_`` per layer / per expert, so the peak equals the steady-state
footprint.  Measured peak after the change: 52.24 GiB (-16.12 GiB).

The returned dict keys, shapes, dtypes and values are bit-identical to the previous
implementation -- this file is shared by the A100 and H100 ports.
"""

from __future__ import annotations

from typing import Any

import torch

from qimeng_megakernel.models.LLM.llama.preprocess import (
    _interleave_qk_weight_indices,
    _interleave_rope_cos_sin,
)


def _load_hf_qwen_moe_model(
    model_name: str, *, device: torch.device, dtype: torch.dtype
) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _empty_stack(sample: torch.Tensor, num_layers: int) -> torch.Tensor:
    """Pre-allocate the destination for ``torch.stack([sample-like] * num_layers, 0)``."""
    return torch.empty(
        (num_layers, *sample.shape), device=sample.device, dtype=sample.dtype
    )


def _is_module_list(experts_mod: Any) -> bool:
    """transformers < 4.56 exposes ``mlp.experts`` as an nn.ModuleList of experts."""
    try:
        iter(experts_mod)
    except TypeError:
        return False
    return True


def _qwen2moe_experts_fused_views(experts_mod: Any, config: Any):
    """Fused ``Qwen2MoeExperts`` (transformers >= 4.56) -> (gate_up, down) 3D views.

    Returns zero-copy views shaped (num_experts, out, in); raises if the layout is
    not one of the two documented forms.

    """
    num_experts = getattr(config, "num_experts", 60)
    hidden_size = config.hidden_size

    gate_up = getattr(experts_mod, "gate_up_proj", None)
    if gate_up is None or not hasattr(gate_up, "weight"):
        raise ValueError("Qwen2MoeExperts: gate_up_proj not found or has no weight")
    w_gu = gate_up.weight
    if w_gu.dim() == 3 and w_gu.shape[0] == num_experts:
        pass
    elif w_gu.dim() == 2 and w_gu.shape[1] == hidden_size:
        out_per_expert = w_gu.shape[0] // num_experts
        w_gu = w_gu.view(num_experts, out_per_expert, hidden_size)
    else:
        raise ValueError(
            f"Qwen2MoeExperts gate_up_proj weight shape not handled: {w_gu.shape}"
        )

    down_proj = getattr(experts_mod, "down_proj", None)
    if down_proj is None or not hasattr(down_proj, "weight"):
        raise ValueError("Qwen2MoeExperts: down_proj not found or has no weight")
    w_d = down_proj.weight
    if w_d.dim() == 3 and w_d.shape[0] == num_experts:
        pass
    elif w_d.dim() == 2:
        in_per_expert = w_d.shape[1]
        out_total = w_d.shape[0]
        w_d = w_d.view(num_experts, out_total // num_experts, in_per_expert)
    else:
        raise ValueError(
            f"Qwen2MoeExperts down_proj weight shape not handled: {w_d.shape}"
        )
    return w_gu, w_d


def _routed_expert_shapes(layer: Any, config: Any) -> tuple[int, tuple, tuple]:
    """(num_experts, gate_up_shape_per_expert, down_shape_per_expert) from one layer."""
    experts_mod = layer.mlp.experts
    if _is_module_list(experts_mod):
        experts = list(experts_mod)
        e0 = experts[0]
        gw = e0.gate_proj.weight
        uw = e0.up_proj.weight
        dw = e0.down_proj.weight
        return len(experts), (gw.shape[0] + uw.shape[0], gw.shape[1]), tuple(dw.shape)
    w_gu, w_d = _qwen2moe_experts_fused_views(experts_mod, config)
    return int(w_gu.shape[0]), tuple(w_gu.shape[1:]), tuple(w_d.shape[1:])


def _release_param(param: torch.Tensor) -> None:
    """Drop the storage behind an HF parameter while keeping the attribute alive."""
    data = param.data if isinstance(param, torch.nn.Parameter) else param
    data.set_(
        torch.empty(0, device=data.device, dtype=data.dtype).untyped_storage(),
        0,
        (0,),
        (1,),
    )


def _copy_layer_routed_experts(
    layer: Any,
    config: Any,
    dst_gate_up: torch.Tensor,
    dst_down: torch.Tensor,
    *,
    release_hf_experts: bool = False,
) -> None:
    """Stream one layer's routed-expert weights into pre-allocated destinations.

    dst_gate_up: (num_experts, gate_out + up_out, hidden)  -- gate rows first, then up
    dst_down:    (num_experts, hidden, moe_intermediate)
    """
    experts_mod = layer.mlp.experts
    if _is_module_list(experts_mod):
        for expert_idx, expert in enumerate(experts_mod):
            gw = expert.gate_proj.weight
            uw = expert.up_proj.weight
            dw = expert.down_proj.weight
            gate_rows = gw.shape[0]
            dst_gate_up[expert_idx, :gate_rows].copy_(gw)
            dst_gate_up[expert_idx, gate_rows:].copy_(uw)
            dst_down[expert_idx].copy_(dw)
            if release_hf_experts:
                _release_param(gw)
                _release_param(uw)
                _release_param(dw)
        return

    w_gu, w_d = _qwen2moe_experts_fused_views(experts_mod, config)
    dst_gate_up.copy_(w_gu)
    dst_down.copy_(w_d)
    if release_hf_experts:
        _release_param(experts_mod.gate_up_proj.weight)
        _release_param(experts_mod.down_proj.weight)


def _qwen_moe_weights_from_hf(
    model: Any, *, release_hf_experts: bool = False
) -> dict[str, torch.Tensor]:
    """Build weight dict from HF Qwen2MoeForCausalLM (attention + MoE).

    This dict is the single source of truth for both prefill and decode: same tensor
    shapes, dtypes, and key names. Prefill (fused_prefill._build_globs_for_prefill_*)
    and decode (Globals in decode_schedule) both consume this layout unchanged.
    Llama-style attention layout; MoE: shared expert + router + expert_gate_up/down.

    All per-layer stacks are pre-allocated and filled by streaming ``copy_`` so no
    temporary duplicate of the multi-GiB routed-expert tensors is ever materialised
    (see module docstring).

    release_hf_experts: opt-in. When True, each HF routed-expert parameter's storage is
        freed right after it has been copied out (~23 GiB reclaimed for this model).
        The HF model is then unusable for routed-expert forward passes, so callers that
        keep it as a numerical baseline (all current verifier tests) must leave it False.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("HF model does not expose expected Qwen2Moe layers.")

    config = model.config
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_attention_heads)
    device = next(model.parameters()).device
    indices_for_q, indices_for_k = _interleave_qk_weight_indices(
        num_attention_heads, num_key_value_heads, head_dim, device
    )

    layers = model.model.layers
    num_layers = len(layers)
    if num_layers == 0:
        raise ValueError("HF Qwen2Moe model has no layers.")
    layer0 = layers[0]

    # ---- pre-allocate every stacked destination from layer 0's shapes ----------
    q0 = layer0.self_attn.q_proj.weight
    k0 = layer0.self_attn.k_proj.weight
    v0 = layer0.self_attn.v_proj.weight
    q_rows, k_rows = q0.shape[0], k0.shape[0]
    qkv_rows = q_rows + k_rows + v0.shape[0]

    qkv_weights = torch.empty(
        (num_layers, qkv_rows, q0.shape[1]), device=q0.device, dtype=q0.dtype
    )
    has_qkv_bias = (
        hasattr(layer0.self_attn.q_proj, "bias")
        and layer0.self_attn.q_proj.bias is not None
    )
    qkv_bias = (
        torch.empty(
            (num_layers, qkv_rows),
            device=layer0.self_attn.q_proj.bias.device,
            dtype=layer0.self_attn.q_proj.bias.dtype,
        )
        if has_qkv_bias
        else None
    )

    o_proj_weights = _empty_stack(layer0.self_attn.o_proj.weight, num_layers)
    attn_norm_weights = _empty_stack(layer0.input_layernorm.weight, num_layers)
    mlp_norm_weights = _empty_stack(layer0.post_attention_layernorm.weight, num_layers)
    router_weights = _empty_stack(layer0.mlp.gate.weight, num_layers)
    shared_expert_gate_proj_weights = _empty_stack(
        layer0.mlp.shared_expert.gate_proj.weight, num_layers
    )
    shared_expert_up_proj_weights = _empty_stack(
        layer0.mlp.shared_expert.up_proj.weight, num_layers
    )
    shared_expert_down_proj_weights = _empty_stack(
        layer0.mlp.shared_expert.down_proj.weight, num_layers
    )
    shared_expert_gate_weights = _empty_stack(
        layer0.mlp.shared_expert_gate.weight, num_layers
    )

    num_experts, gate_up_shape, down_shape = _routed_expert_shapes(layer0, config)
    expert_gate_up_proj_weights = torch.empty(
        (num_layers, num_experts, *gate_up_shape), device=q0.device, dtype=q0.dtype
    )
    expert_down_proj_weights = torch.empty(
        (num_layers, num_experts, *down_shape), device=q0.device, dtype=q0.dtype
    )

    # ---- stream layer by layer -------------------------------------------------
    for layer_idx, layer in enumerate(layers):
        # Attention (same as Llama): Q/K output dims reordered to interleaved RoPE order
        q = layer.self_attn.q_proj.weight
        k = layer.self_attn.k_proj.weight
        v = layer.self_attn.v_proj.weight
        qkv_weights[layer_idx, :q_rows].copy_(q[indices_for_q])
        qkv_weights[layer_idx, q_rows : q_rows + k_rows].copy_(k[indices_for_k])
        qkv_weights[layer_idx, q_rows + k_rows :].copy_(v)
        if qkv_bias is not None:
            qb = layer.self_attn.q_proj.bias
            kb = layer.self_attn.k_proj.bias
            vb = layer.self_attn.v_proj.bias
            qkv_bias[layer_idx, :q_rows].copy_(qb[indices_for_q])
            qkv_bias[layer_idx, q_rows : q_rows + k_rows].copy_(kb[indices_for_k])
            qkv_bias[layer_idx, q_rows + k_rows :].copy_(vb)
        o_proj_weights[layer_idx].copy_(layer.self_attn.o_proj.weight)
        attn_norm_weights[layer_idx].copy_(layer.input_layernorm.weight)
        mlp_norm_weights[layer_idx].copy_(layer.post_attention_layernorm.weight)

        # MoE: router gate (num_experts, hidden)
        router_weights[layer_idx].copy_(layer.mlp.gate.weight)
        # Shared expert
        shared_expert_gate_proj_weights[layer_idx].copy_(
            layer.mlp.shared_expert.gate_proj.weight
        )
        shared_expert_up_proj_weights[layer_idx].copy_(
            layer.mlp.shared_expert.up_proj.weight
        )
        shared_expert_down_proj_weights[layer_idx].copy_(
            layer.mlp.shared_expert.down_proj.weight
        )
        shared_expert_gate_weights[layer_idx].copy_(layer.mlp.shared_expert_gate.weight)
        # Routed experts: HF has ModuleList of experts, or Qwen2MoeExperts (fused 3D weights)
        _copy_layer_routed_experts(
            layer,
            config,
            expert_gate_up_proj_weights[layer_idx],
            expert_down_proj_weights[layer_idx],
            release_hf_experts=release_hf_experts,
        )

    lm_head_norm_weights = model.model.norm.weight
    lm_head_weights = model.lm_head.weight

    # RoPE
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is None and layers:
        rotary_emb = getattr(layers[0].self_attn, "rotary_emb", None)
    if rotary_emb is None:
        raise ValueError("Unable to locate rotary embedding on HF Qwen2Moe model.")

    max_pos = getattr(config, "max_position_embeddings", 32768)
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
        "router_weights": router_weights,
        "shared_expert_gate_proj_weights": shared_expert_gate_proj_weights,
        "shared_expert_up_proj_weights": shared_expert_up_proj_weights,
        "shared_expert_down_proj_weights": shared_expert_down_proj_weights,
        "shared_expert_gate_weights": shared_expert_gate_weights,
        "expert_gate_up_proj_weights": expert_gate_up_proj_weights,
        "expert_down_proj_weights": expert_down_proj_weights,
        "lm_head_norm_weights": lm_head_norm_weights,
        "lm_head_weights": lm_head_weights,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }
