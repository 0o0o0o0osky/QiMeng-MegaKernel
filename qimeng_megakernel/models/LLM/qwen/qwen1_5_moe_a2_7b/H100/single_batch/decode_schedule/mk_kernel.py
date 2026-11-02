"""Load and call the Qwen1.5-MoE-A2.7B H100 single-batch decode megakernel.

The ``mk`` call, ``qwen_moe_globals`` fields, and pybind field pointers share one
ordered ABI. Preprocess owns mutable decode allocations; this module validates the
ABI and applies metadata-only shape adaptations.
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
from .model_config import QwenMoeConfig


_MODULE_NAME = "qimeng_qwen1_5_moe_a2_7b_h100_single_batch_kernel"

#: Fields that must be non-None before the kernel is called.  Mirrors
#: `globs.DECODE_ALLOCATED_FIELDS` minus the ones this kernel does not take, plus
#: `instructions` (owned by the scheduler).
_REQUIRED_FIELDS = (
    "instructions",
    "barriers",
    "hidden_states",
    "normed_hidden_states",
    "q_post_rope",
    "k_cache",
    "v_cache",
    "attn_out",
    "attn_lse_intermediates",
    "attn_out_intermediates",
    "shared_expert_silu_out",
    "shared_out",
    "routed_silu_buffer",
    "routed_out_f32",
    "router_logits_buf",
    "expert_indices",
    "expert_weights",
    "logits",
    "attn_norm_weights",
    "qkv_weights",
    "qkv_bias",
    "rope_cos",
    "rope_sin",
    "o_proj_weights",
    "mlp_norm_weights",
    "shared_expert_gate_proj_weights",
    "shared_expert_up_proj_weights",
    "shared_expert_down_proj_weights",
    "shared_expert_gate_weights",
    "router_weights",
    "expert_gate_up_proj_weights_flat",
    "expert_down_proj_weights_flat",
    "lm_head_norm_weights",
    "lm_head_weights",
)


def _default_mk_dir() -> Path:
    """Return the route-owned decode_kernel directory."""
    return Path(__file__).resolve().parent.parent / "decode_kernel"


def get_mk(mk_dir: Path | None = None):
    """Return the prebuilt route-owned native kernel."""
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


def _tensor_for_mk(t: torch.Tensor | None) -> torch.Tensor | None:
    """Hand pybind a plain tensor; an `nn.Parameter` makes it throw 'Expected a torch.Tensor'."""
    if t is None:
        return t
    return t.data if isinstance(t, torch.nn.Parameter) else t


def _assert_ready(globs: Globals) -> None:
    missing = [name for name in _REQUIRED_FIELDS if getattr(globs, name, None) is None]
    if missing:
        raise ValueError(
            "Globals is not decode-ready; these fields are None: "
            + ", ".join(missing)
            + ". H100 preprocess/fused-prefill must pre-allocate every decode buffer."
        )


def call_mk(globs: Globals, mk: Any) -> None:
    """Call the compiled H100 kernel in the locked ABI order."""
    kernel = mk

    _assert_ready(globs)

    attn_scale = 1.0 / math.sqrt(QwenMoeConfig.head_dim)
    rms_norm_eps = float(QwenMoeConfig.rms_norm_eps)
    pos_id = int(globs.position_id) if globs.position_id is not None else 0

    instructions = globs.instructions
    if not instructions.is_contiguous():
        instructions = instructions.contiguous()

    # ThunderKittens `gl<>` is at most 4D.  L1 lays the KV cache out as
    # [L, B=1, S, n_kv, head_dim]; dropping the batch axis is a pure view (dim size 1),
    # and `.contiguous()` on an already-contiguous view is a no-op -- no allocation.
    k_cache = _tensor_for_mk(globs.k_cache)
    v_cache = _tensor_for_mk(globs.v_cache)
    if k_cache.dim() == 5:
        k_cache = k_cache.squeeze(1)
        v_cache = v_cache.squeeze(1)

    # The native ABI consumes shared-expert gate weights as (L, hidden).
    gate_w = _tensor_for_mk(globs.shared_expert_gate_weights)
    if gate_w.dim() == 3:
        assert gate_w.shape[1] == 1, (
            f"shared_expert_gate_weights must be (L, 1, H); got {tuple(gate_w.shape)}"
        )
        gate_w = gate_w.squeeze(1)

    kernel(
        # --- instruction stream -------------------------------------------------
        instructions,
        # --- activation buffers -------------------------------------------------
        globs.hidden_states,
        globs.normed_hidden_states,
        globs.q_post_rope,
        k_cache,
        v_cache,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.shared_expert_silu_out,
        globs.shared_out,
        globs.routed_silu_buffer,
        globs.routed_out_f32,
        globs.router_logits_buf,
        globs.expert_indices,
        globs.expert_weights,
        globs.logits,
        # --- weights ------------------------------------------------------------
        globs.attn_norm_weights,
        globs.qkv_weights,
        globs.qkv_bias,
        globs.rope_cos,
        globs.rope_sin,
        globs.o_proj_weights,
        globs.mlp_norm_weights,
        globs.shared_expert_gate_proj_weights,
        globs.shared_expert_up_proj_weights,
        globs.shared_expert_down_proj_weights,
        gate_w,
        globs.router_weights,
        globs.expert_gate_up_proj_weights_flat,
        globs.expert_down_proj_weights_flat,
        _tensor_for_mk(globs.lm_head_norm_weights),
        _tensor_for_mk(globs.lm_head_weights),
        # --- sync + scalars -----------------------------------------------------
        globs.barriers,
        pos_id,
        attn_scale,
        rms_norm_eps,
        stream=torch.cuda.current_stream(globs.hidden_states.device),
    )


def run_instructions(
    globs: Globals,
    instructions: list[Instruction],
    *,
    mk: Any = None,
) -> None:
    """Run an explicit instruction sequence through the production megakernel."""
    from .instructions import scheduler

    if mk is None:
        mk = get_mk()
    sm_count = scheduler._get_sm_count(globs.hidden_states.device)
    queues = scheduler._round_robin_assign_to_sms(instructions, sm_count)
    scheduler.tensorize_instructions(globs, queues)
    call_mk(globs, mk)
