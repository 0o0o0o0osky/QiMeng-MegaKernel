"""Instruction-based decode for Qwen1.5-MoE-A2.7B @ H100.

- `instruction_defs`: the 12 emitted opcodes (1-4, 7, 8-14).  Opcodes 5 / 6 are unused
  (this model has no dense MLP).
- `scheduler`: emits the per-layer instruction stream in topological order, assigns it
  round-robin to 132 SMs, and serializes it into `globs.instructions`.  Also carries the
  barrier model (`barrier_signals_of` / `barrier_waits_of`) that L4 implements against.
- `python_vm`: the PyVM handlers, imported lazily by callers that execute them.

The exported instruction definitions, scheduler and barrier model define the L4 interface.
"""

import importlib

from .instruction_defs import (
    INSTRUCTION_BY_OPCODE,
    MOE_INTERMEDIATE_SIZE,
    NUM_DOWNPROJ_SPLIT_K_SHARED,
    SPLIT_K_GRANULARITY_SHARED,
    TILE_M_GRANULARITY,
    TILE_N_GRANULARITY,
    DecodeAttentionReduction,
    DecodeAttentionSplitKV,
    MoE_Combine_Commit,
    OProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    RMS_QKVProj_GEMV_RoPE,
    RoutedExpert_DownProj,
    RoutedExpert_UpGate_SiLU,
    Router_GEMV,
    Router_SoftmaxTopK,
    SharedExpert_DownProj,
    SharedExpert_UpGate_SiLU,
)
from . import scheduler

__all__ = [
    "RMS_QKVProj_GEMV_RoPE",
    "DecodeAttentionSplitKV",
    "DecodeAttentionReduction",
    "OProj_GEMV_Residual",
    "RMS_LMHead_GEMV",
    "Router_GEMV",
    "Router_SoftmaxTopK",
    "SharedExpert_UpGate_SiLU",
    "SharedExpert_DownProj",
    "RoutedExpert_UpGate_SiLU",
    "RoutedExpert_DownProj",
    "MoE_Combine_Commit",
    "INSTRUCTION_BY_OPCODE",
    "TILE_M_GRANULARITY",
    "TILE_N_GRANULARITY",
    "NUM_DOWNPROJ_SPLIT_K_SHARED",
    "SPLIT_K_GRANULARITY_SHARED",
    "MOE_INTERMEDIATE_SIZE",
    "scheduler",
    "python_vm",
]


def __getattr__(name: str):
    # Keep schedule construction independent of the torch-heavy handler module.
    if name == "python_vm":
        return importlib.import_module(".python_vm", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
