# H100 decode preprocess — Qwen3-0.6B. Mirrors A100 decode globals, with an
# optional ThunderKittens-specific KV reshape when for_mk=True.
from __future__ import annotations

from typing import Any

import torch

from qimeng_megakernel.models.LLM.preprocess import DecodePreprocessOutput, prepare_kv_cache_for_decode
from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.preprocess import (
    _load_hf_qwen3_model,
    _stack_qwen3_weights_from_hf,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule.model_config import (
    QWEN3_H100_ATTENTION_INTERMEDIATE_COLS,
    Qwen3Config,
)


def _get_sm_count(device):
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
):
    seq_len_hint = int(prefill.kv_seq_len_before.max().item())
    target_seq_len = None
    kv = prepare_kv_cache_for_decode(prefill.past_key_values, seq_len_hint=seq_len_hint)

    if model is None and model_name:
        model = _load_hf_qwen3_model(
            model_name, device=prefill.logits.device, dtype=prefill.logits.dtype
        )

    if model is None:
        return Globals(k_cache=kv.k_cache.contiguous(), v_cache=kv.v_cache.contiguous())

    Qwen3Config.update_from_hf(model.config)
    if debug:
        print("=== Qwen3 0.6B config (H100) ===")
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
    hd = getattr(
        model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads
    )
    kv = DecodePreprocessOutput(
        k_cache=_interleave_k_cache(kv.k_cache, hd), v_cache=kv.v_cache
    )

    k_cache = kv.k_cache.contiguous()
    v_cache = kv.v_cache.contiguous()
    if for_mk and k_cache.dim() == 5:
        k_cache = k_cache.squeeze(1).contiguous()
        v_cache = v_cache.squeeze(1).contiguous()
        if k_cache.shape[1] < k_cache.shape[2]:
            k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
            v_cache = v_cache.permute(0, 2, 1, 3).contiguous()

    weights = _stack_qwen3_weights_from_hf(model)
    dev = getattr(model, "device", kv.k_cache.device)
    dtype = getattr(model, "dtype", kv.k_cache.dtype)
    nlayers = model.config.num_hidden_layers
    nheads = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    vocab_size = model.config.vocab_size
    head_dim = Qwen3Config.head_dim
    q_attn_dim = nheads * head_dim
    qkv_out_dim = (nheads + nkv * 2) * head_dim

    max_attn = min(_get_sm_count(dev), 132)
    num_partials = min(27, max(1, (max_attn + nkv - 1) // nkv))
    mk_intermediates_cols = (
        QWEN3_H100_ATTENTION_INTERMEDIATE_COLS if for_mk else num_partials
    )

    def buf(shape, dt=dtype):
        return torch.zeros(shape, device=dev, dtype=dt)

    def buf32(shape):
        return torch.zeros(shape, device=dev, dtype=torch.float32)

    with torch.no_grad():
        first_gen = prefill.first_gen_token
        hidden_states_init = model.model.embed_tokens(first_gen).squeeze(0).squeeze(0).to(dtype)

    return Globals(
        position_id=int(prefill.pos_id_token_1.max().item()),
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
        qkv_bias=None,
        o_proj_weights=weights["o_proj_weights"].contiguous(),
        attn_norm_weights=weights["attn_norm_weights"].contiguous(),
        q_norm_weights=weights["q_norm_weights"].contiguous(),
        k_norm_weights=weights["k_norm_weights"].contiguous(),
        mlp_norm_weights=weights["mlp_norm_weights"].contiguous(),
        up_proj_weights=weights["up_proj_weights"].contiguous(),
        gate_proj_weights=weights["gate_proj_weights"].contiguous(),
        down_proj_weights=weights["down_proj_weights"].contiguous(),
        lm_head_norm_weights=weights["lm_head_norm_weights"].contiguous(),
        lm_head_weights=weights["lm_head_weights"].contiguous(),
        rope_cos=weights["rope_cos"].contiguous(),
        rope_sin=weights["rope_sin"].contiguous(),
        hidden_states=hidden_states_init.contiguous(),
        qkv_raw=buf(qkv_out_dim),
        q_post_rope=buf(q_attn_dim),
        attn_out=buf(q_attn_dim),
        attn_out_intermediates=buf32([nheads, mk_intermediates_cols, head_dim]),
        attn_lse_intermediates=buf32([nheads, mk_intermediates_cols]),
        silu_out=buf(intermediate_size),
        logits=buf(vocab_size),
        barriers=torch.zeros(
            [nlayers, 10, nheads + nkv * 2], dtype=torch.int32, device=dev
        ),
    )
