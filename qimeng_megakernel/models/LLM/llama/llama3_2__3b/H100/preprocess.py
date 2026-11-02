# H100 (Hopper) decode preprocess — Llama 3.2 3B.
# ThunderKittens kernel uses 4D KV cache (L, S, H, D); Python Globals use 5D (L, B, S, H, D).
# When for_mk=True preprocess outputs 4D KV (L, S, H, D) so call_mk passes cache directly.
# Attention intermediates in the kernel are 4D (e.g. attn_out_intermediates 1×n_heads×partitions×head_dim);
# Python allocates [nheads, max_attn, head_dim] / [nheads, max_attn] for fused/pyvm compatibility.
from __future__ import annotations
from typing import Any
import torch
from qimeng_megakernel.models.LLM.preprocess import DecodePreprocessOutput, prepare_kv_cache_for_decode
from qimeng_megakernel.models.LLM.llama.preprocess import (
    _interleave_k_cache, _load_hf_llama_model, _stack_llama_weights_from_hf,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule.model_config import (
    ATTENTION_INTERMEDIATE_COLS,
    BARRIER_ROWS,
    LlamaConfig,
)


def preprocess_decode(*, prefill, model=None, model_name=None, max_new_tokens=None, debug=False, for_mk=False, **_kwargs):
    seq_len_hint = int(prefill.kv_seq_len_before.max().item())
    target_seq_len = None
    kv = prepare_kv_cache_for_decode(prefill.past_key_values, seq_len_hint=seq_len_hint)
    if model is None and model_name:
        model = _load_hf_llama_model(model_name, device=prefill.logits.device, dtype=prefill.logits.dtype)
    if model:
        LlamaConfig.update_from_hf(model.config)
        if debug:
            print("=== Llama 3.2 3B config (H100) ===")
        if max_new_tokens is not None:
            target_seq_len = seq_len_hint + max_new_tokens
        if target_seq_len is None and hasattr(model, "config"):
            mp = getattr(model.config, "max_position_embeddings", None)
            if isinstance(mp, int):
                target_seq_len = mp
        if target_seq_len is not None:
            kv = prepare_kv_cache_for_decode(
                prefill.past_key_values, seq_len_hint=seq_len_hint, target_seq_len=target_seq_len
            )
        hd = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
        kv = DecodePreprocessOutput(k_cache=_interleave_k_cache(kv.k_cache, hd), v_cache=kv.v_cache)
    k_cache = kv.k_cache.contiguous()
    v_cache = kv.v_cache.contiguous()
    if for_mk and k_cache.dim() == 5:
        k_cache = k_cache.squeeze(1).contiguous()
        v_cache = v_cache.squeeze(1).contiguous()
    if model is None:
        return Globals(k_cache=k_cache, v_cache=v_cache)
    weights = _stack_llama_weights_from_hf(model)
    dev = getattr(model, "device", kv.k_cache.device)
    dtype = getattr(model, "dtype", kv.k_cache.dtype)
    def buf(s, dt=dtype):
        return torch.zeros(s, device=dev, dtype=dt)
    def buf32(s):
        return torch.zeros(s, device=dev, dtype=torch.float32)
    nlayers = model.config.num_hidden_layers
    nheads = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    return Globals(
        position_id=int(prefill.pos_id_token_1.max().item()),
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
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
        hidden_states=buf(model.config.hidden_size),
        q_post_rope=buf(model.config.hidden_size),
        attn_out=buf(model.config.hidden_size),
        attn_out_intermediates=buf32(
            [nheads, ATTENTION_INTERMEDIATE_COLS, model.config.head_dim]
        ),
        attn_lse_intermediates=buf32([nheads, ATTENTION_INTERMEDIATE_COLS]),
        silu_out=buf(model.config.intermediate_size),
        logits=buf(model.config.vocab_size),
        barriers=torch.zeros(
            [nlayers, BARRIER_ROWS, nheads + nkv * 2],
            dtype=torch.int32,
            device=dev,
        ),
    )
