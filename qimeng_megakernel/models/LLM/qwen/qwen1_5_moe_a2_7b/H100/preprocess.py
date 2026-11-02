"""Build the Qwen1.5-MoE-A2.7B H100 decode ``Globals`` from prefill state.

The configured device determines ``num_partials``. Decode buffers are allocated once,
barriers have shape ``[L, 15, 128]`` for opcodes 1 through 14 plus a spare row, and
routed-expert weights expose zero-copy three-dimensional views compatible with
ThunderKittens ``gl<>``. Every tensor is contiguous and resident on the target device.
This module allocates tensors; it does not schedule or execute instructions.
"""

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
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.H100.single_batch.decode_schedule.globs import (
    DECODE_ALLOCATED_FIELDS,
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.H100.single_batch.decode_schedule.model_config import (
    QwenMoeConfig,
    get_num_partials_for_device,
)


def _on_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Contiguous + on `device`, without gratuitous copies."""
    if t.device != device:
        t = t.to(device)
    return t.contiguous()


def _flat_expert_view(w: torch.Tensor, name: str) -> torch.Tensor:
    """Collapse (L, E, a, b) -> (L*E, a, b) with a guaranteed zero-copy `.view()`.

    `.view()` (not `.reshape()`) so that a layout that would force a copy raises instead
    of silently duplicating 15.5 GiB.  The data_ptr assert is the belt to that braces.
    """
    if w.dim() != 4:
        raise ValueError(f"{name}: expected 4D (L, E, a, b), got {tuple(w.shape)}")
    if not w.is_contiguous():
        raise ValueError(f"{name}: must be contiguous before flattening")
    flat = w.view(w.shape[0] * w.shape[1], w.shape[2], w.shape[3])
    assert flat.data_ptr() == w.data_ptr(), f"{name}: flat view copied storage"
    assert flat.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
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
    """Build a fully-allocated `Globals` for Qwen1.5-MoE-A2.7B H100 decode."""
    seq_len_hint = int(prefill.kv_seq_len_before.max().item())
    target_seq_len = None
    kv = prepare_kv_cache_for_decode(prefill.past_key_values, seq_len_hint=seq_len_hint)

    if model is None and model_name:
        model = _load_hf_qwen_moe_model(
            model_name, device=prefill.logits.device, dtype=prefill.logits.dtype
        )
    if model is None:
        # Degenerate path (KV-only probe); kept for parity with the A100 entry point.
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
    head_dim = QwenMoeConfig.head_dim
    # K cache is stored in the interleaved-RoPE order so that fused/mk attention can read
    # it with the same pair ordering the interleaved q uses.  [L, B, S, n_kv, head_dim].
    kv = DecodePreprocessOutput(
        k_cache=_interleave_k_cache(kv.k_cache, head_dim), v_cache=kv.v_cache
    )

    dev = torch.device(getattr(model, "device", kv.k_cache.device))
    dtype = getattr(model, "dtype", kv.k_cache.dtype)

    k_cache = _on_device(kv.k_cache, dev)
    v_cache = _on_device(kv.v_cache, dev)
    weights = _qwen_moe_weights_from_hf(model)

    n_layers = QwenMoeConfig.num_hidden_layers
    n_heads = QwenMoeConfig.num_attention_heads
    hidden_size = QwenMoeConfig.hidden_size
    vocab_size = QwenMoeConfig.vocab_size
    num_experts = QwenMoeConfig.num_experts
    num_experts_per_tok = QwenMoeConfig.num_experts_per_tok
    moe_intermediate = QwenMoeConfig.moe_intermediate_size
    shared_intermediate = QwenMoeConfig.intermediate_size

    def buf(shape, dt=dtype):
        return torch.zeros(shape, device=dev, dtype=dt)

    def buf32(shape):
        return torch.zeros(shape, device=dev, dtype=torch.float32)

    with torch.no_grad():
        first_gen = prefill.first_gen_token
        hidden_states_init = (
            model.model.embed_tokens(first_gen).squeeze(0).squeeze(0).to(dtype)
        )

    # D1: single source of truth, shared with the L3 scheduler.
    num_partials = get_num_partials_for_device(dev)

    # D3: opcodes 1..14 -> rows 0..13, +1 spare row.
    barriers = torch.zeros(
        (n_layers, QwenMoeConfig.num_opcode_rows, QwenMoeConfig.barrier_slots),
        dtype=torch.int32,
        device=dev,
    )

    # D4: 4D routed-expert stacks + their zero-copy 3D views.
    expert_gate_up = _on_device(weights["expert_gate_up_proj_weights"], dev)
    expert_down = _on_device(weights["expert_down_proj_weights"], dev)
    expert_gate_up_flat = _flat_expert_view(
        expert_gate_up, "expert_gate_up_proj_weights"
    )
    expert_down_flat = _flat_expert_view(expert_down, "expert_down_proj_weights")

    qkv_bias = weights.get("qkv_bias")
    if qkv_bias is None:
        # Qwen1.5-MoE always has QKV bias; keep an explicit zero tensor rather than None
        # so no downstream call site has to lazily materialise one.
        qkv_bias = buf((n_layers, QwenMoeConfig.qkv_out_dim))
    else:
        qkv_bias = _on_device(qkv_bias, dev)

    globs = Globals(
        # instructions stays None: owned by the L3 scheduler.
        barriers=barriers,
        position_id=int(prefill.pos_id_token_1.max().item()),
        # --- weights -----------------------------------------------------------
        attn_norm_weights=_on_device(weights["attn_norm_weights"], dev),
        qkv_weights=_on_device(weights["qkv_weights"], dev),
        qkv_bias=qkv_bias,
        rope_cos=_on_device(weights["rope_cos"], dev),
        rope_sin=_on_device(weights["rope_sin"], dev),
        o_proj_weights=_on_device(weights["o_proj_weights"], dev),
        mlp_norm_weights=_on_device(weights["mlp_norm_weights"], dev),
        shared_expert_gate_proj_weights=_on_device(
            weights["shared_expert_gate_proj_weights"], dev
        ),
        shared_expert_up_proj_weights=_on_device(
            weights["shared_expert_up_proj_weights"], dev
        ),
        shared_expert_down_proj_weights=_on_device(
            weights["shared_expert_down_proj_weights"], dev
        ),
        # The shared-expert gate layout is (L, 1, hidden).
        shared_expert_gate_weights=_on_device(
            weights["shared_expert_gate_weights"], dev
        ),
        router_weights=_on_device(weights["router_weights"], dev),
        router_logits_buf=buf32(num_experts),
        expert_gate_up_proj_weights=expert_gate_up,
        expert_down_proj_weights=expert_down,
        expert_gate_up_proj_weights_flat=expert_gate_up_flat,
        expert_down_proj_weights_flat=expert_down_flat,
        lm_head_norm_weights=_on_device(weights["lm_head_norm_weights"], dev),
        lm_head_weights=_on_device(weights["lm_head_weights"], dev),
        # --- buffers -----------------------------------------------------------
        k_cache=k_cache,
        v_cache=v_cache,
        hidden_states=hidden_states_init.contiguous(),
        q_post_rope=buf(hidden_size),
        attn_out=buf(hidden_size),
        attn_out_intermediates=buf32([n_heads, num_partials, head_dim]),
        attn_lse_intermediates=buf32([n_heads, num_partials]),
        # D2: all of the following were lazily allocated per decode step on A100.
        normed_hidden_states=buf(hidden_size),
        shared_expert_silu_out=buf(shared_intermediate),
        shared_out=buf(hidden_size),
        routed_out=buf(hidden_size),
        routed_out_f32=buf32(hidden_size),
        routed_silu_buffer=buf((num_experts_per_tok, moe_intermediate)),
        expert_indices=torch.zeros(
            num_experts_per_tok, device=dev, dtype=torch.int64
        ),
        expert_weights=buf32(num_experts_per_tok),
        silu_out=buf(moe_intermediate),
        logits=buf(vocab_size),
    )

    # D5 + D2 guard: fail loudly here rather than let a None reach the kernel.
    for name in DECODE_ALLOCATED_FIELDS:
        value = getattr(globs, name)
        if value is None:
            raise RuntimeError(f"preprocess_decode left Globals.{name} unallocated")
        if isinstance(value, torch.Tensor):
            if value.device.type != dev.type or value.device.index != dev.index:
                raise RuntimeError(
                    f"Globals.{name} on {value.device}, expected {dev}"
                )
            if not value.is_contiguous():
                raise RuntimeError(f"Globals.{name} is not contiguous")

    return globs
