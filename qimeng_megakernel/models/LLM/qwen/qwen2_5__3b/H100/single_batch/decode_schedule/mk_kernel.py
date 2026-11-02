"""Load and call the Qwen2.5 3B H100 decode megakernel."""
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


_MODULE_NAME = "qimeng_qwen2_5_3b_h100_single_batch_kernel"


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
    """Pass plain tensor to C++; nn.Parameter can make pybind throw 'Expected a torch.Tensor'."""
    if t is None:
        return t
    return t.data if isinstance(t, torch.nn.Parameter) else t


def call_mk(globs: Globals, mk: Any) -> None:
    """
    Call the compiled Qwen mk kernel. Argument order matches decode_kernel/mk.cu and globals.cuh.
    KV cache: Python globs use (L, kv_heads, seq, hd); kernel expects (L, seq, kv_heads, hd) -> permute.
    """
    lm_head_norm = _tensor_for_mk(globs.lm_head_norm_weights)
    lm_head_w = _tensor_for_mk(globs.lm_head_weights)
    attn_scale = 1.0 / math.sqrt(Qwen2Config.head_dim)
    rms_norm_eps = float(Qwen2Config.rms_norm_eps)
    pos_id = int(globs.position_id) if globs.position_id is not None else 0
    instructions = globs.instructions
    if instructions is not None and not instructions.is_contiguous():
        instructions = instructions.contiguous()

    k = globs.k_cache.squeeze(1) if globs.k_cache.dim() == 5 else globs.k_cache
    v = globs.v_cache.squeeze(1) if globs.v_cache.dim() == 5 else globs.v_cache
    # Kernel expects the same (L, seq, kv_heads, hd) layout produced by preprocessing.
    k_cache_4d = k.contiguous()
    v_cache_4d = v.contiguous()

    qkv_bias = _tensor_for_mk(globs.qkv_bias)
    # Kernel expects (L, QKV_NUM_BLOCKS, 16); preprocess may give (L, qkv_out_dim). Reshape for TMA.
    if qkv_bias is not None and qkv_bias.dim() == 2:
        qkv_bias = qkv_bias.view(qkv_bias.size(0), -1, 16).contiguous()

    # Normalize preprocessing variants to (1, 1, max_pos, head_dim).
    rope_cos = globs.rope_cos
    rope_sin = globs.rope_sin
    if rope_cos.dim() == 3:
        rope_cos = rope_cos[0]
        rope_sin = rope_sin[0]
    if rope_cos.dim() == 2:
        rope_cos = rope_cos.unsqueeze(0).unsqueeze(0)
        rope_sin = rope_sin.unsqueeze(0).unsqueeze(0)
    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()

    mk(
        instructions,
        globs.hidden_states,
        globs.q_post_rope,
        k_cache_4d,
        v_cache_4d,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        globs.attn_norm_weights,
        globs.qkv_weights,
        rope_cos,
        rope_sin,
        qkv_bias,
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
