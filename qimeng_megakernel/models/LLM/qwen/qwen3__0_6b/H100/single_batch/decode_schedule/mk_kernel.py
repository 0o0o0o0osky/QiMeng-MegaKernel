"""Load and call the Qwen3 0.6B H100 decode megakernel."""
from __future__ import annotations

import importlib.util
import math
import sysconfig
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.instruction import Instruction
from .globs import Globals
from .instructions.scheduler import RUNTIME_MID_KV_MAX, RUNTIME_P1_KV_MAX
from .model_config import QWEN3_H100_ATTENTION_INTERMEDIATE_COLS, Qwen3Config

_MODULE_NAME = "qimeng_qwen3_0_6b_h100_single_batch_kernel"


def _default_mk_dir() -> Path:
    """Return the model-owned decode_kernel directory."""
    this_dir = Path(__file__).resolve().parent
    return (this_dir.parent / "decode_kernel").resolve()


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
    if t is None:
        return t
    return t.data if isinstance(t, torch.nn.Parameter) else t


def call_mk(globs: Globals, mk: Any) -> None:
    """
    Call the compiled Qwen3 H100 mk kernel. Adapts buffer shapes (4D, intermediates cols), then invokes mk.
    """

    pos_id = int(globs.position_id) if globs.position_id is not None else 0
    # The scheduler builds immutable short/mid/long fanout tensors once before the token
    # loop.  Select by the actual KV length so a long-capacity cache does not
    # force every early token to pay the long-sequence split/reduction overhead.
    instructions_tensor = globs.instructions
    instruction_buckets = getattr(
        globs, "_qwen3_instruction_tensors_by_kv_bucket", None
    )
    if instruction_buckets:
        # These are instantaneous crossover points.  They are intentionally
        # below the scheduler's whole-stream fixed-fanout boundaries.
        bucket = (
            0
            if pos_id < RUNTIME_P1_KV_MAX
            else (1 if pos_id < RUNTIME_MID_KV_MAX else 2)
        )
        instructions_tensor = instruction_buckets[bucket]
    if instructions_tensor is not None and not instructions_tensor.is_contiguous():
        instructions_tensor = instructions_tensor.contiguous()

    # Normalize preprocessing layouts to the kernel ABI.
    def _as_mk_4d(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        if t.dim() == 4:
            return t if t.is_contiguous() else t.contiguous()
        if t.dim() == 1:
            return t.view(1, 1, 1, -1).contiguous()
        if t.dim() == 2:
            return t.unsqueeze(0).unsqueeze(0).contiguous()
        if t.dim() == 3:
            return t.unsqueeze(0).contiguous()
        return t.contiguous()

    hidden_states_4d = _as_mk_4d(globs.hidden_states)
    qkv_raw_4d = _as_mk_4d(globs.qkv_raw)
    q_post_rope_4d = _as_mk_4d(globs.q_post_rope)
    attn_out_4d = _as_mk_4d(globs.attn_out)
    silu_out_4d = _as_mk_4d(globs.silu_out)
    logits_4d = _as_mk_4d(globs.logits)
    attn_lse_4d = _as_mk_4d(globs.attn_lse_intermediates)
    attn_out_inter_4d = _as_mk_4d(globs.attn_out_intermediates)
    if attn_lse_4d.shape[-1] != QWEN3_H100_ATTENTION_INTERMEDIATE_COLS:
        raise ValueError(
            "attn_lse_intermediates must be allocated with the H100 native column count"
        )
    if attn_out_inter_4d.shape[2] != QWEN3_H100_ATTENTION_INTERMEDIATE_COLS:
        raise ValueError(
            "attn_out_intermediates must be allocated with the H100 native column count"
        )

    k_cache_4d = globs.k_cache.squeeze(1) if globs.k_cache.dim() == 5 else globs.k_cache
    v_cache_4d = globs.v_cache.squeeze(1) if globs.v_cache.dim() == 5 else globs.v_cache
    if not k_cache_4d.is_contiguous():
        k_cache_4d = k_cache_4d.contiguous()
    if not v_cache_4d.is_contiguous():
        v_cache_4d = v_cache_4d.contiguous()

    rope_cos = globs.rope_cos
    rope_sin = globs.rope_sin
    if rope_cos is not None and rope_cos.dim() == 2:
        rope_cos = rope_cos.unsqueeze(0).unsqueeze(0).contiguous()
    if rope_sin is not None and rope_sin.dim() == 2:
        rope_sin = rope_sin.unsqueeze(0).unsqueeze(0).contiguous()

    lm_head_norm = _tensor_for_mk(globs.lm_head_norm_weights)
    lm_head_w = _tensor_for_mk(globs.lm_head_weights)
    attn_scale = 1.0 / math.sqrt(Qwen3Config.head_dim)
    scale = float(attn_scale)
    rms_eps = float(Qwen3Config.rms_norm_eps)

    mk(
        instructions_tensor,
        hidden_states_4d,
        qkv_raw_4d,
        q_post_rope_4d,
        k_cache_4d,
        v_cache_4d,
        attn_out_4d,
        attn_lse_4d,
        attn_out_inter_4d,
        silu_out_4d,
        logits_4d,
        globs.attn_norm_weights,
        globs.q_norm_weights,
        globs.k_norm_weights,
        globs.qkv_weights,
        rope_cos,
        rope_sin,
        globs.o_proj_weights,
        globs.mlp_norm_weights,
        globs.up_proj_weights,
        globs.gate_proj_weights,
        globs.down_proj_weights,
        lm_head_norm,
        lm_head_w,
        globs.barriers,
        pos_id,
        scale,
        rms_eps,
        stream=torch.cuda.current_stream(hidden_states_4d.device),
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
