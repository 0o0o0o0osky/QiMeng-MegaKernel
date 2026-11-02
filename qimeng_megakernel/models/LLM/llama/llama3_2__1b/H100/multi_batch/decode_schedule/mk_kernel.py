"""Load and call the H100 multi-batch decode megakernel."""

from __future__ import annotations

import importlib.util
import math
import sysconfig
from pathlib import Path
from typing import Any

import torch

from .globs import BATCH_CAPACITY, MAX_BATCH_SIZE, Globals
from .model_config import LlamaConfig

_MODULE_NAME = "qimeng_llama3_2_1b_h100_multi_batch_kernel"


def _default_mk_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "decode_kernel"


def get_mk(mk_dir: Path | None = None):
    """Return the prebuilt route-owned kernel."""
    root = (_default_mk_dir() if mk_dir is None else Path(mk_dir)).resolve()
    dso = root / f"{_MODULE_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    if not dso.is_file():
        raise ImportError(f"missing prebuilt kernel {dso}; build decode_kernel first")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, dso)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prebuilt kernel {dso}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "mk", None)):
        raise ImportError(f"{dso} does not export mk")
    return module.mk


def _plain_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.data if isinstance(tensor, torch.nn.Parameter) else tensor


def _require_tensor(name: str, value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise ValueError(f"multi-batch MK requires Globals.{name}")
    if not value.is_cuda or not value.is_contiguous():
        raise ValueError(f"Globals.{name} must be a contiguous CUDA tensor")
    return value


def _prepare_controls(globs: Globals, batch_size: int, device: torch.device) -> None:
    if globs.position_ids is None:
        pos = 0 if globs.position_id is None else int(globs.position_id)
        globs.position_ids = torch.full(
            (batch_size,), pos, dtype=torch.long, device=device
        )
    if globs.active_mask is None:
        globs.active_mask = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
    if globs.position_ids.dtype != torch.long or not globs.position_ids.is_contiguous():
        raise ValueError("Globals.position_ids must be contiguous torch.int64")
    if globs.active_mask is None or globs.active_mask.dtype not in (
        torch.bool,
        torch.uint8,
    ):
        raise ValueError("Globals.active_mask must be torch.bool or torch.uint8")
    if not globs.active_mask.is_contiguous():
        raise ValueError("Globals.active_mask must be contiguous")


def call_mk(globs: Globals, mk: Any) -> None:
    """Launch one instruction-queue persistent megakernel."""
    globs.validate_padded_storage()
    batch_size = int(globs.logical_batch_size)
    if not 2 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"H100 multi-batch MK supports B=2..{MAX_BATCH_SIZE}")
    hidden = _require_tensor("hidden_states_backing", globs.hidden_states_backing)
    device = hidden.device
    if torch.cuda.get_device_properties(device).multi_processor_count != 132:
        raise ValueError("H100 multi-batch MK requires exactly 132 SMs")
    _prepare_controls(globs, batch_size, device)

    instructions = _require_tensor("instructions", globs.instructions)
    if (
        instructions.dtype != torch.int32
        or instructions.ndim != 3
        or tuple(instructions.shape[::2]) != (132, 32)
    ):
        raise ValueError("Globals.instructions must be int32 [132, queue, 32]")
    mk_barriers = _require_tensor("mk_barriers", globs.mk_barriers)
    mk_barriers.zero_()

    k_cache = _require_tensor("k_cache", _plain_tensor(globs.k_cache))
    v_cache = _require_tensor("v_cache", _plain_tensor(globs.v_cache))
    if k_cache.ndim != 5 or tuple(k_cache.shape[:2]) != (
        LlamaConfig.num_hidden_layers,
        batch_size,
    ):
        raise ValueError("KV cache must be [layers, B, max_seq, kv_heads, head_dim]")
    max_seq_len = int(k_cache.shape[2])
    k_cache_4d = k_cache.view(
        LlamaConfig.num_hidden_layers,
        batch_size * max_seq_len,
        LlamaConfig.num_key_value_heads,
        LlamaConfig.head_dim,
    )
    v_cache_4d = v_cache.view_as(k_cache_4d)
    k_cache_decode = _require_tensor("k_cache_decode", globs.k_cache_decode)
    v_cache_decode = _require_tensor("v_cache_decode", globs.v_cache_decode)
    expected_decode_shape = (
        LlamaConfig.num_hidden_layers,
        BATCH_CAPACITY,
        LlamaConfig.num_key_value_heads,
        max_seq_len,
        LlamaConfig.head_dim,
    )
    if tuple(k_cache_decode.shape) != expected_decode_shape:
        raise ValueError(
            f"decode KV cache must be {expected_decode_shape}, got "
            f"{tuple(k_cache_decode.shape)}"
        )
    k_cache_decode_4d = k_cache_decode.view(
        LlamaConfig.num_hidden_layers,
        BATCH_CAPACITY * LlamaConfig.num_key_value_heads,
        max_seq_len,
        LlamaConfig.head_dim,
    )
    v_cache_decode_4d = v_cache_decode.view_as(k_cache_decode_4d)

    args = (
        instructions,
        hidden,
        _require_tensor("q_post_rope_backing", globs.q_post_rope_backing),
        k_cache_4d,
        v_cache_4d,
        k_cache_decode_4d,
        v_cache_decode_4d,
        _require_tensor("attn_out_backing", globs.attn_out_backing),
        _require_tensor("silu_out_backing", globs.silu_out_backing),
        _require_tensor("logits_backing", globs.logits_backing),
        _require_tensor("attn_rms_out_backing", globs.attn_rms_out_backing),
        _require_tensor("mlp_rms_out_backing", globs.mlp_rms_out_backing),
        _require_tensor("lm_head_rms_out_backing", globs.lm_head_rms_out_backing),
        globs.attn_rms_out_backing[:8],
        globs.attn_rms_out_backing[8:],
        globs.attn_out_backing[:8],
        globs.attn_out_backing[8:],
        _require_tensor("down_partials", globs.down_partials),
        _require_tensor("attn_norm_weights", _plain_tensor(globs.attn_norm_weights)),
        _require_tensor("qkv_weights", _plain_tensor(globs.qkv_weights)),
        _require_tensor("rope_cos", _plain_tensor(globs.rope_cos)),
        _require_tensor("rope_sin", _plain_tensor(globs.rope_sin)),
        _require_tensor("o_proj_weights", _plain_tensor(globs.o_proj_weights)),
        _require_tensor("mlp_norm_weights", _plain_tensor(globs.mlp_norm_weights)),
        _require_tensor("up_proj_weights", _plain_tensor(globs.up_proj_weights)),
        _require_tensor("gate_proj_weights", _plain_tensor(globs.gate_proj_weights)),
        _require_tensor("down_proj_weights", _plain_tensor(globs.down_proj_weights)),
        _require_tensor(
            "lm_head_norm_weights", _plain_tensor(globs.lm_head_norm_weights)
        ),
        _require_tensor("lm_head_weights", _plain_tensor(globs.lm_head_weights)),
        mk_barriers,
        globs.position_ids,
        globs.active_mask,
        batch_size,
        max_seq_len,
        1.0 / math.sqrt(LlamaConfig.head_dim),
        float(LlamaConfig.rms_norm_eps),
    )
    mk(*args, stream=torch.cuda.current_stream(device))


__all__ = ["call_mk", "get_mk"]
