
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import importlib

import torch
from einops import rearrange

from qimeng_megakernel.models.LLM.registry import resolve_llm_impl


@dataclass
class DecodePreprocessOutput:
    k_cache: torch.Tensor
    v_cache: torch.Tensor


def _resolve_impl_module(
    model: str,
    *,
    preprocess_impl: str | None = None,
    arch: str | None = None,
    hardware: str | None = None,
):
    if preprocess_impl:
        return importlib.import_module(preprocess_impl)
    impl = resolve_llm_impl(model, arch=arch, hardware=hardware)
    return importlib.import_module(impl.preprocess)


def preprocess_decode(model: str, *args, **kwargs):
    preprocess_impl = kwargs.pop("preprocess_impl", None)
    arch = kwargs.get("arch")
    hardware = kwargs.get("hardware")
    module = _resolve_impl_module(
        model,
        preprocess_impl=preprocess_impl,
        arch=arch,
        hardware=hardware,
    )
    if not hasattr(module, "preprocess_decode"):
        raise NotImplementedError(
            f"decode preprocessing not implemented in: {module.__name__}"
        )
    kwargs.setdefault("model_name", model)
    # Map loaded_model -> model so impl receives model=loaded_model without conflicting with positional "model" (string)
    loaded = kwargs.pop("loaded_model", None)
    if loaded is not None:
        kwargs["model"] = loaded
    return module.preprocess_decode(*args, **kwargs)


def _normalize_kv_layout(
    kv: torch.Tensor,
    *,
    seq_len_hint: int | None = None,
) -> torch.Tensor:
    """
    Convert HF KV cache layout to ours: output (B, S, H, D) so stack gives [L, B, S, H, D].

    HF layout is documented and fixed per model:
    - transformers/cache_utils.py DynamicCache: "The expected shape for each tensor is
      [batch_size, num_heads, seq_len, head_dim]".
    - transformers/models/llama/modeling_llama.py LlamaAttention: key_states/value_states
      are .view(hidden_shape).transpose(1, 2) → (B, num_heads, seq_len, head_dim).
    - LlamaModel docstring (legacy cache): "shape (batch_size, num_heads, sequence_length,
      embed_size_per_head)".
    So for Llama + DynamicCache the per-layer layout is (B, H, S, D). We convert to (B, S, H, D).

    seq_len_hint: when known (e.g. prefill.kv_seq_len_before), use it to identify which dim
    is S (dim size == seq_len_hint). If omitted, use heuristic: if shape[1] > shape[2] treat
    as (B, H, S, D) and rearrange.
    """
    if kv.ndim != 4:
        raise ValueError(f"Expected kv cache with 4 dims, got {kv.ndim}")

    if seq_len_hint is not None:
        # HF layout (B, H, S, D): seq_len at dim 2
        if kv.shape[2] == seq_len_hint:
            return rearrange(kv, "b h s d -> b s h d")
        # Already (B, S, H, D): seq_len at dim 1
        if kv.shape[1] == seq_len_hint:
            return kv

    # Fallback: HF per-layer is (B, H, S, D); we want (B, S, H, D). Typically S <= H for short prompts.
    # If first middle dim > second, treat as (B, H, S, D) and convert.
    if kv.shape[1] > kv.shape[2]:
        return rearrange(kv, "b h s d -> b s h d")
    return kv


def stack_past_key_values(
    past_key_values: Any,
    *,
    seq_len_hint: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stack per-layer KV caches into [layers, batch, seq, head, dim].
    """
    if past_key_values is None:
        raise ValueError("past_key_values is required for decode preprocessing.")

    # New Cache API (transformers >= 4.45): Cache has .layers, each layer has .keys and .values
    if hasattr(past_key_values, "layers") and isinstance(getattr(past_key_values, "layers", None), (list, tuple)):
        k_layers = []
        v_layers = []
        for layer in past_key_values.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None or v is None or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise ValueError("Cache layer must have .keys and .values as torch.Tensors.")
            k_layers.append(_normalize_kv_layout(k, seq_len_hint=seq_len_hint))
            v_layers.append(_normalize_kv_layout(v, seq_len_hint=seq_len_hint))
        return torch.stack(k_layers, dim=0), torch.stack(v_layers, dim=0)

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        k_cache = past_key_values.key_cache
        v_cache = past_key_values.value_cache
        if isinstance(k_cache, (list, tuple)):
            k_layers = []
            v_layers = []
            for k, v in zip(k_cache, v_cache, strict=False):
                if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                    raise ValueError("KV cache entries must be torch.Tensors.")
                k_layers.append(_normalize_kv_layout(k, seq_len_hint=seq_len_hint))
                v_layers.append(_normalize_kv_layout(v, seq_len_hint=seq_len_hint))
            return torch.stack(k_layers, dim=0), torch.stack(v_layers, dim=0)
        if k_cache.ndim == 5:
            # Normalize to [layers, batch, seq, head, dim] (fused ops expect this). HF may give [L, B, H, S, D].
            if k_cache.shape[2] > k_cache.shape[3]:
                k_cache = rearrange(k_cache, "l b h s d -> l b s h d")
                v_cache = rearrange(v_cache, "l b h s d -> l b s h d")
            return k_cache, v_cache
        if k_cache.ndim == 4:
            k_cache = _normalize_kv_layout(k_cache, seq_len_hint=seq_len_hint)
            v_cache = _normalize_kv_layout(v_cache, seq_len_hint=seq_len_hint)
            return k_cache.unsqueeze(0), v_cache.unsqueeze(0)
        raise ValueError(
            "Unsupported cache shape; expected 4 or 5 dims for key_cache."
        )

    # Fused prefill returns (k_cache_5D, v_cache_5D) — same layout [L, B, S, H, D]; normalize like key_cache 5D.
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 2:
        a, b = past_key_values[0], past_key_values[1]
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.ndim == 5
            and b.ndim == 5
        ):
            k_cache, v_cache = a, b
            if k_cache.shape[2] > k_cache.shape[3]:
                k_cache = rearrange(k_cache, "l b h s d -> l b s h d")
                v_cache = rearrange(v_cache, "l b h s d -> l b s h d")
            return k_cache, v_cache

    if isinstance(past_key_values, (tuple, list)):
        k_layers = []
        v_layers = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise ValueError("Unrecognized past_key_values layer format.")
            k, v = layer[0], layer[1]
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise ValueError("KV cache entries must be torch.Tensors.")
            k = _normalize_kv_layout(k, seq_len_hint=seq_len_hint)
            v = _normalize_kv_layout(v, seq_len_hint=seq_len_hint)
            k_layers.append(k)
            v_layers.append(v)
        return torch.stack(k_layers, dim=0), torch.stack(v_layers, dim=0)

    raise ValueError("Unrecognized past_key_values format.")


def _pad_kv_cache_to_len(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    target_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_cache.ndim != 5 or v_cache.ndim != 5:
        raise ValueError("KV cache must be [layers, batch, seq, head, dim].")
    cur_len = k_cache.shape[2]
    if target_seq_len <= cur_len:
        return k_cache, v_cache
    pad_len = target_seq_len - cur_len
    pad_shape = (
        k_cache.shape[0],
        k_cache.shape[1],
        pad_len,
        k_cache.shape[3],
        k_cache.shape[4],
    )
    k_pad = torch.zeros(
        pad_shape, device=k_cache.device, dtype=k_cache.dtype
    )
    v_pad = torch.zeros(
        pad_shape, device=v_cache.device, dtype=v_cache.dtype
    )
    k_cache = torch.cat([k_cache, k_pad], dim=2)
    v_cache = torch.cat([v_cache, v_pad], dim=2)
    return k_cache, v_cache


def prepare_kv_cache_for_decode(
    past_key_values: Any,
    *,
    seq_len_hint: int | None = None,
    target_seq_len: int | None = None,
) -> DecodePreprocessOutput:
    k_cache, v_cache = stack_past_key_values(
        past_key_values, seq_len_hint=seq_len_hint
    )
    if target_seq_len is not None:
        k_cache, v_cache = _pad_kv_cache_to_len(
            k_cache, v_cache, target_seq_len=target_seq_len
        )
    return DecodePreprocessOutput(
        k_cache=k_cache.contiguous(),
        v_cache=v_cache.contiguous(),
    )
