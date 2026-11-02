"""
Load and call the compiled multi-batch decode megakernel from models .../multi_batch/decode_kernel/.
"""
from __future__ import annotations

import importlib.util
import math
import sysconfig
from pathlib import Path
from typing import Any

import torch

from .globs import Globals
from .model_config import LlamaConfig
from .mk_launch_tensors import (
    position_id_int_from_globs,
    prepare_mk_batch_control_tensors,
)

_MODULE_NAME = "qimeng_llama3_2_1b_a100_multi_batch_kernel"


def _tensor_for_mk(t: torch.Tensor | None) -> torch.Tensor | None:
    """Pass plain tensor to C++; nn.Parameter can make pybind throw."""
    if t is None:
        return t
    return t.data if isinstance(t, torch.nn.Parameter) else t


def _default_mk_dir() -> Path:
    """Same layout as single_batch: .../multi_batch/decode_kernel next to decode_schedule/."""
    this_dir = Path(__file__).resolve().parent
    return this_dir.parent / "decode_kernel"


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


def call_mk(globs: Globals, mk: Any) -> None:
    """
    Call the compiled mk kernel with current multi-batch Globals.
    """
    batch_size = 1 if globs.hidden_states.ndim == 1 else int(globs.hidden_states.shape[0])
    attn_scale = 1.0 / math.sqrt(LlamaConfig.head_dim)
    rms_norm_eps = LlamaConfig.rms_norm_eps
    pos_id = position_id_int_from_globs(globs)
    dev = globs.hidden_states.device

    position_ids_arg, active_mask_arg, attn_partial_valid_arg = prepare_mk_batch_control_tensors(
        dev,
        batch_size,
        globs,
    )

    k_cache = _tensor_for_mk(globs.k_cache)
    v_cache = _tensor_for_mk(globs.v_cache)

    base_args = (
        globs.instructions,
        globs.hidden_states,
        globs.q_post_rope,
        k_cache,
        v_cache,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        globs.attn_norm_weights,
        globs.qkv_weights,
        globs.rope_cos,
        globs.rope_sin,
        globs.o_proj_weights,
        globs.mlp_norm_weights,
        globs.up_proj_weights,
        globs.gate_proj_weights,
        globs.down_proj_weights,
        globs.lm_head_norm_weights,
        globs.lm_head_weights,
        globs.barriers,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids_arg,
        active_mask_arg,
        attn_partial_valid_arg,
    )
    ar = _tensor_for_mk(globs.attn_rms_out)
    mr = _tensor_for_mk(globs.mlp_rms_out)
    lr = _tensor_for_mk(globs.lm_head_rms_out)
    if ar is None or mr is None or lr is None:
        raise ValueError(
            "multi-batch mk requires Globals.attn_rms_out, mlp_rms_out, and "
            "lm_head_rms_out"
        )
    mk(*base_args, ar, mr, lr)
