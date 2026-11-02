"""Build canonical Qwen2.5 A100 decode globals from an HF prefill."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.preprocess import (
    DecodePreprocessOutput,
    prepare_kv_cache_for_decode,
)
from qimeng_megakernel.models.LLM.qwen.preprocess import (
    _load_hf_qwen2_model,
    _stack_qwen2_weights_from_hf,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.single_batch.decode_schedule.instructions.instruction_defs import (
    ATTENTION_REDUCTION_WORKSPACE_CAPACITY,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.single_batch.decode_schedule.model_config import (
    Qwen2Config,
)


def _get_sm_count(device) -> int:
    if getattr(device, "type", None) != "cuda":
        return 1
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1


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
    """Create the single canonical A100 layout consumed by PyVM and CUDA."""
    del for_mk
    seq_len_hint = int(prefill.kv_seq_len_before.max().item())
    target_seq_len = None
    kv = prepare_kv_cache_for_decode(
        prefill.past_key_values, seq_len_hint=seq_len_hint
    )
    if model is None and model_name:
        model = _load_hf_qwen2_model(
            model_name,
            device=prefill.logits.device,
            dtype=prefill.logits.dtype,
        )
    if model is None:
        return Globals(
            k_cache=kv.k_cache.contiguous(),
            v_cache=kv.v_cache.contiguous(),
        )

    Qwen2Config.update_from_hf(model.config)
    if debug:
        print("=== Qwen2.5 3B config (A100) ===")
    if max_new_tokens is not None:
        target_seq_len = seq_len_hint + max_new_tokens
    if target_seq_len is None:
        max_positions = getattr(model.config, "max_position_embeddings", None)
        if isinstance(max_positions, int):
            target_seq_len = max_positions
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
        k_cache=_interleave_k_cache(kv.k_cache, head_dim),
        v_cache=kv.v_cache,
    )

    weights = _stack_qwen2_weights_from_hf(model)
    device = getattr(model, "device", kv.k_cache.device)
    dtype = getattr(model, "dtype", kv.k_cache.dtype)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_partials = min(
        ATTENTION_REDUCTION_WORKSPACE_CAPACITY,
        max(1, min(_get_sm_count(device), 108) // num_kv_heads),
    )

    def buf(shape, tensor_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=tensor_dtype)

    def buf32(shape):
        return torch.zeros(shape, device=device, dtype=torch.float32)

    qkv_bias = weights.get("qkv_bias")
    if qkv_bias is not None:
        qkv_bias = qkv_bias.reshape(num_layers, -1, 16).contiguous()
    with torch.no_grad():
        hidden_states = (
            model.model.embed_tokens(prefill.first_gen_token)
            .squeeze(0)
            .squeeze(0)
            .to(dtype)
            .contiguous()
        )

    return Globals(
        position_id=int(prefill.pos_id_token_1.max().item()),
        k_cache=kv.k_cache.contiguous(),
        v_cache=kv.v_cache.contiguous(),
        qkv_weights=weights["qkv_weights"].contiguous(),
        qkv_bias=qkv_bias,
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
        q_post_rope=buf(num_heads * head_dim),
        attn_out=buf(num_heads * head_dim),
        attn_out_intermediates=buf32([num_heads, num_partials, head_dim]),
        attn_lse_intermediates=buf32([num_heads, num_partials]),
        silu_out=buf(model.config.intermediate_size),
        logits=buf(model.config.vocab_size),
        barriers=torch.zeros(
            [num_layers, 6, num_heads + num_kv_heads * 2],
            dtype=torch.int32,
            device=device,
        ),
    )
