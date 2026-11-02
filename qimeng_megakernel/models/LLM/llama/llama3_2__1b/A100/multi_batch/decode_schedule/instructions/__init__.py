from __future__ import annotations

import importlib

from .instruction_defs import (
    TILE_M_GRANULARITY,
    TILE_N_GRANULARITY,
    QKVProj_GEMM_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMM_Residual,
    UpGateProj_GEMM_SiLU,
    DownProj_GEMM_Residual,
    LMHeadProj_GEMM,
    AttnRMS,
    MlpRMS,
    LMHeadRMS,
)

__all__ = [
    "TILE_M_GRANULARITY",
    "TILE_N_GRANULARITY",
    "QKVProj_GEMM_RoPE",
    "DecodeAttentionSplitKV",
    "DecodeAttentionReduction",
    "OProj_GEMM_Residual",
    "UpGateProj_GEMM_SiLU",
    "DownProj_GEMM_Residual",
    "LMHeadProj_GEMM",
    "AttnRMS",
    "MlpRMS",
    "LMHeadRMS",
    "scheduler",
    "python_vm",
]


def __getattr__(name: str):
    if name == "scheduler":
        return importlib.import_module(".scheduler", __name__)
    if name == "python_vm":
        return importlib.import_module(".python_vm", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
