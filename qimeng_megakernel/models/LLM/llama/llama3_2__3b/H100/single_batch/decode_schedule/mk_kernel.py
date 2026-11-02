"""
Load and call the compiled Llama 3.2 3B H100 single-batch decode megakernel.
Kernel expects 4D kv cache (layer, seq, head, dim). Preprocess outputs 4D when for_mk=True.
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
from .model_config import ATTENTION_INTERMEDIATE_COLS, BARRIER_ROWS, LlamaConfig


_MODULE_NAME = "qimeng_llama3_2_3b_h100_single_batch_kernel"


def _require_exact_tensor(
    globs: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = getattr(globs, name, None)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Llama-3.2-3B/H100 native ABI requires '{name}' to be a "
            f"torch.Tensor, got {type(tensor).__name__}"
        )
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"Llama-3.2-3B/H100 native ABI shape mismatch for '{name}': "
            f"expected {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise TypeError(
            f"Llama-3.2-3B/H100 native ABI dtype mismatch for '{name}': "
            f"expected {dtype}, got {tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"Llama-3.2-3B/H100 native ABI requires '{name}' to be contiguous"
        )
    return tensor


def _validate_native_scratch_shapes(globs: Globals) -> None:
    """Validate route-owned scratch tensors before entering the native ABI."""
    _require_exact_tensor(
        globs,
        "attn_lse_intermediates",
        shape=(LlamaConfig.num_attention_heads, ATTENTION_INTERMEDIATE_COLS),
        dtype=torch.float32,
    )
    _require_exact_tensor(
        globs,
        "attn_out_intermediates",
        shape=(
            LlamaConfig.num_attention_heads,
            ATTENTION_INTERMEDIATE_COLS,
            LlamaConfig.head_dim,
        ),
        dtype=torch.float32,
    )
    _require_exact_tensor(
        globs,
        "barriers",
        shape=(
            LlamaConfig.num_hidden_layers,
            BARRIER_ROWS,
            LlamaConfig.num_attention_heads + 2 * LlamaConfig.num_key_value_heads,
        ),
        dtype=torch.int32,
    )


def _default_mk_dir() -> Path:
    """Return the route-owned decode_kernel directory."""
    this_dir = Path(__file__).resolve().parent
    single_batch = this_dir.parent
    return single_batch / "decode_kernel"


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
    Call the compiled mk kernel. Argument order matches decode_kernel/mk.cu and globals.cuh.
    """
    _validate_native_scratch_shapes(globs)
    lm_head_norm = _tensor_for_mk(globs.lm_head_norm_weights)
    lm_head_w = _tensor_for_mk(globs.lm_head_weights)
    attn_scale = 1.0 / math.sqrt(LlamaConfig.head_dim)
    rms_norm_eps = float(LlamaConfig.rms_norm_eps)
    pos_id = int(globs.position_id) if globs.position_id is not None else 0
    instructions = globs.instructions
    if instructions is not None and not instructions.is_contiguous():
        instructions = instructions.contiguous()

    # MK (ThunderKittens/TMA) expects 4D KV (layer, seq, head, dim). Fused prefill gives 5D (L, B=1, S, H, D); squeeze(1).
    k_cache = _tensor_for_mk(globs.k_cache)
    v_cache = _tensor_for_mk(globs.v_cache)
    if k_cache is not None and k_cache.dim() == 5:
        k_cache = k_cache.squeeze(1).contiguous()
        v_cache = v_cache.squeeze(1).contiguous() if v_cache is not None else None

    mk(
        instructions,
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
    """Execute one instruction list through the model's compiled megakernel."""
    from .instructions import scheduler

    if mk is None:
        mk = get_mk()
    sm_count = scheduler._get_sm_count(globs.hidden_states.device)
    queues = scheduler._round_robin_assign_to_sms(instructions, sm_count)
    scheduler.tensorize_instructions(globs, queues)
    call_mk(globs, mk)
