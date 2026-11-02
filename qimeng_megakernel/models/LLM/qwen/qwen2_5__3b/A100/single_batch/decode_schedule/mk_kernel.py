"""
Load and call the compiled Qwen2.5 3B A100 decode megakernel (decode_kernel/).
KV cache layout: [L, 1, S, n_kv, hd] (5D); A100 kernel expects this shape directly.
Includes qkv_bias (Qwen2.5 has bias on q/k/v_proj).

Uses the model-owned single-batch decode_kernel/ built in place.
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
from .model_config import Qwen2Config

_MODULE_NAME = "qimeng_qwen2_5_3b_a100_single_batch_kernel"


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


def _tensor_for_mk(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None:
        return t
    return t.data if isinstance(t, torch.nn.Parameter) else t


def call_mk(globs: Globals, mk: Any) -> None:
    """
    Call the compiled Qwen A100 mk kernel.
    KV cache: A100 kernel expects [L, 1, S, n_kv, hd] (5D); pass globs.k_cache directly.
    """
    lm_head_norm = _tensor_for_mk(globs.lm_head_norm_weights)
    lm_head_w = _tensor_for_mk(globs.lm_head_weights)
    attn_scale = 1.0 / math.sqrt(Qwen2Config.head_dim)
    rms_norm_eps = float(Qwen2Config.rms_norm_eps)
    pos_id = int(globs.position_id) if globs.position_id is not None else 0

    instructions = globs.instructions
    if instructions is not None and not instructions.is_contiguous():
        instructions = instructions.contiguous()

    # A100 kernel expects 5D kv cache: [L, 1, S, n_kv, hd]. Add batch dim if 4D.
    k = globs.k_cache
    v = globs.v_cache
    if k.dim() == 4:
        k = k.unsqueeze(1)
    if v.dim() == 4:
        v = v.unsqueeze(1)
    k_cache_5d = k.contiguous()
    v_cache_5d = v.contiguous()

    qkv_bias = _tensor_for_mk(globs.qkv_bias)
    # Kernel expects (L, QKV_NUM_BLOCKS, 16); preprocess may give (L, qkv_out_dim). Reshape.
    if qkv_bias is not None and qkv_bias.dim() == 2:
        qkv_bias = qkv_bias.view(qkv_bias.size(0), -1, 16).contiguous()

    rope_cos = globs.rope_cos
    rope_sin = globs.rope_sin
    if rope_cos.dim() == 3:
        rope_cos = rope_cos[0].contiguous()
        rope_sin = rope_sin[0].contiguous()

    mk(
        instructions,
        globs.hidden_states,
        globs.q_post_rope,
        k_cache_5d,
        v_cache_5d,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        globs.attn_norm_weights,
        globs.qkv_weights,
        qkv_bias,
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
        attn_scale,
        rms_norm_eps,
    )
    torch.cuda.synchronize(globs.hidden_states.device)


def run_instructions(
    globs: Globals,
    instructions: list[Instruction],
    *,
    mk: Any = None,
) -> None:
    """Execute an instruction list through the selected A100 megakernel."""
    from .instructions import scheduler

    mk_call = get_mk() if mk is None else mk
    sm_count = scheduler._get_sm_count(globs.hidden_states.device)
    queues = scheduler._round_robin_assign_to_sms(instructions, sm_count)
    scheduler.tensorize_instructions(globs, queues)
    call_mk(globs, mk_call)
