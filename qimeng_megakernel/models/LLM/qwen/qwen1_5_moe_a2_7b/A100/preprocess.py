# A100 decode preprocess — Qwen1.5-MoE-A2.7B. Builds Globals from HF prefill for fused decode.

from __future__ import annotations

from typing import Any

import torch

from qimeng_megakernel.models.LLM.preprocess import (
    DecodePreprocessOutput,
    prepare_kv_cache_for_decode,
)
from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.weights import (
    _load_hf_qwen_moe_model,
    _qwen_moe_weights_from_hf,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.globs import (
    DECODE_ALLOCATED_FIELDS,
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.model_config import (
    QwenMoeConfig,
    get_num_partials_for_device,
)


def _on_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device != device:
        t = t.to(device)
    return t.contiguous()


def _flat_expert_view(w: torch.Tensor, name: str) -> torch.Tensor:
    if w.dim() != 4 or not w.is_contiguous():
        raise ValueError(f"{name}: expected contiguous 4D expert weights")
    flat = w.view(w.shape[0] * w.shape[1], w.shape[2], w.shape[3])
    assert flat.data_ptr() == w.data_ptr()
    return flat


def preprocess_decode(
    *,
    prefill,
    model=None,
    model_name=None,
    max_new_tokens=None,
    debug=False,
    for_mk=False,
    **_kwargs,
) -> Globals:
    """Build Globals from HF prefill for Qwen1.5-MoE-A2.7B A100 fused decode."""
    seq_len_hint = int(prefill.kv_seq_len_before.max().item())
    target_seq_len = None
    kv = prepare_kv_cache_for_decode(prefill.past_key_values, seq_len_hint=seq_len_hint)

    if model is None and model_name:
        model = _load_hf_qwen_moe_model(
            model_name, device=prefill.logits.device, dtype=prefill.logits.dtype
        )
    if model is None:
        return Globals(
            k_cache=kv.k_cache.contiguous(), v_cache=kv.v_cache.contiguous()
        )

    QwenMoeConfig.update_from_hf(model.config)
    if max_new_tokens is not None:
        target_seq_len = seq_len_hint + max_new_tokens
    if target_seq_len is None and hasattr(model, "config"):
        mp = getattr(model.config, "max_position_embeddings", None)
        if isinstance(mp, int):
            target_seq_len = mp
    if target_seq_len is not None:
        kv = prepare_kv_cache_for_decode(
            prefill.past_key_values,
            seq_len_hint=seq_len_hint,
            target_seq_len=target_seq_len,
        )
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    kv = DecodePreprocessOutput(
        k_cache=_interleave_k_cache(kv.k_cache, head_dim), v_cache=kv.v_cache
    )

    k_cache = kv.k_cache.contiguous()
    v_cache = kv.v_cache.contiguous()
    weights = _qwen_moe_weights_from_hf(model)
    dev = torch.device(getattr(model, "device", kv.k_cache.device))
    dtype = getattr(model, "dtype", kv.k_cache.dtype)
    n_layers = QwenMoeConfig.num_hidden_layers
    n_heads = QwenMoeConfig.num_attention_heads
    n_kv = QwenMoeConfig.num_key_value_heads
    hidden_size = QwenMoeConfig.hidden_size
    vocab_size = QwenMoeConfig.vocab_size
    num_experts_per_tok = QwenMoeConfig.num_experts_per_tok
    num_experts = QwenMoeConfig.num_experts
    moe_intermediate = QwenMoeConfig.moe_intermediate_size

    def buf(shape, dt=dtype):
        return torch.zeros(shape, device=dev, dtype=dt)

    def buf32(shape):
        return torch.zeros(shape, device=dev, dtype=torch.float32)

    with torch.no_grad():
        first_gen = prefill.first_gen_token
        hidden_states_init = (
            model.model.embed_tokens(first_gen).squeeze(0).squeeze(0).to(dtype)
        )

    num_partials = get_num_partials_for_device(torch.device(dev))
    n_opcode_slots = QwenMoeConfig.num_opcode_rows
    barriers = torch.zeros(
        (n_layers, n_opcode_slots, QwenMoeConfig.barrier_slots),
        dtype=torch.int32,
        device=dev,
    )

    expert_gate_up = _on_device(weights["expert_gate_up_proj_weights"], dev)
    expert_down = _on_device(weights["expert_down_proj_weights"], dev)
    qkv_bias = weights.get("qkv_bias")
    if qkv_bias is None:
        qkv_bias = buf((n_layers, QwenMoeConfig.qkv_out_dim))
    else:
        qkv_bias = _on_device(qkv_bias, dev)

    globs = Globals(
        position_id=int(prefill.pos_id_token_1.max().item()),
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
        qkv_bias=qkv_bias,
        o_proj_weights=weights["o_proj_weights"].contiguous(),
        attn_norm_weights=weights["attn_norm_weights"].contiguous(),
        mlp_norm_weights=weights["mlp_norm_weights"].contiguous(),
        router_weights=weights["router_weights"].contiguous(),
        shared_expert_gate_proj_weights=weights[
            "shared_expert_gate_proj_weights"
        ].contiguous(),
        shared_expert_up_proj_weights=weights[
            "shared_expert_up_proj_weights"
        ].contiguous(),
        shared_expert_down_proj_weights=weights[
            "shared_expert_down_proj_weights"
        ].contiguous(),
        shared_expert_gate_weights=weights["shared_expert_gate_weights"].contiguous(),
        expert_gate_up_proj_weights=expert_gate_up,
        expert_down_proj_weights=expert_down,
        expert_gate_up_proj_weights_flat=_flat_expert_view(
            expert_gate_up, "expert_gate_up_proj_weights"
        ),
        expert_down_proj_weights_flat=_flat_expert_view(
            expert_down, "expert_down_proj_weights"
        ),
        lm_head_norm_weights=weights["lm_head_norm_weights"].contiguous(),
        lm_head_weights=weights["lm_head_weights"].contiguous(),
        rope_cos=weights["rope_cos"].contiguous(),
        rope_sin=weights["rope_sin"].contiguous(),
        hidden_states=hidden_states_init.contiguous(),
        q_post_rope=buf(hidden_size),
        attn_out=buf(hidden_size),
        attn_out_intermediates=buf32([n_heads, num_partials, head_dim]),
        attn_lse_intermediates=buf32([n_heads, num_partials]),
        normed_hidden_states=buf(hidden_size),
        shared_expert_silu_out=buf(QwenMoeConfig.intermediate_size),
        shared_out=buf(hidden_size),
        routed_out=buf(hidden_size),
        routed_out_f32=buf32(hidden_size),
        routed_silu_buffer=buf((num_experts_per_tok, moe_intermediate)),
        router_logits_buf=buf32(num_experts),
        expert_indices=torch.zeros(
            num_experts_per_tok, device=dev, dtype=torch.int64
        ),
        expert_weights=buf32(num_experts_per_tok),
        silu_out=buf(moe_intermediate),
        logits=buf(vocab_size),
        barriers=barriers,
    )
    for name in DECODE_ALLOCATED_FIELDS:
        value = getattr(globs, name)
        if value is None:
            raise RuntimeError(f"preprocess_decode left Globals.{name} unallocated")
        if isinstance(value, torch.Tensor):
            if value.device != dev:
                raise RuntimeError(f"Globals.{name} on {value.device}, expected {dev}")
            if not value.is_contiguous():
                raise RuntimeError(f"Globals.{name} is not contiguous")
    return globs
