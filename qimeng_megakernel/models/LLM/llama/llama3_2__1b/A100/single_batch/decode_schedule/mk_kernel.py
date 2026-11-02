"""
Load and call the compiled decode megakernel (mk) from decode_kernel/.
No registration: decode_route=\"mk\" uses this path directly.
C++ kernel expects 5D KV [L, 1, S, H, D] (see globals.cuh get_k_cache); pass through as-is.

The default decode_kernel issues first-tile weight cp.async before barrier_wait in GEMV
handlers (see decode_kernel/README.md).
"""
from __future__ import annotations

import importlib.util
import math
import sysconfig
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.instruction import Instruction
from .globs import Globals
from .model_config import LlamaConfig

_MODULE_NAME = "qimeng_llama3_2_1b_a100_single_batch_kernel"


def _tensor_for_mk(t: torch.Tensor | None) -> torch.Tensor | None:
    """Pass plain tensor to C++; nn.Parameter can make pybind throw."""
    if t is None:
        return t
    return t.data if isinstance(t, torch.nn.Parameter) else t


def _default_mk_dir() -> Path:
    """Path to the model-owned decode_kernel directory."""
    this_dir = Path(__file__).resolve().parent
    single_batch = this_dir.parent
    return single_batch / "decode_kernel"


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
    Call the compiled mk kernel with current Globals.
    C++ expects 5D KV [L, 1, S, H, D] (globals.cuh); fused prefill gives 5D, pass through.
    """
    attn_scale = 1.0 / math.sqrt(LlamaConfig.head_dim)
    rms_norm_eps = LlamaConfig.rms_norm_eps
    pos_id = globs.position_id if globs.position_id is not None else 0

    k_cache = _tensor_for_mk(globs.k_cache)
    v_cache = _tensor_for_mk(globs.v_cache)

    mk(
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
        attn_scale,
        rms_norm_eps,
    )


def run_instructions(
    globs: Globals,
    instructions: list[Instruction],
    *,
    mk: Any = None,
) -> None:
    """Execute one instruction list through the production A100 megakernel."""
    from .instructions import scheduler

    mk_call = get_mk() if mk is None else mk
    device = globs.hidden_states.device
    sm_count = scheduler._get_sm_count(device)
    queues = scheduler._round_robin_assign_to_sms(instructions, sm_count)
    scheduler.tensorize_instructions(globs, queues)
    call_mk(globs, mk_call)
