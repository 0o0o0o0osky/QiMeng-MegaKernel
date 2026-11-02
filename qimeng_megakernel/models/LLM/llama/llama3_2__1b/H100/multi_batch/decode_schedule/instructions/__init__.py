from __future__ import annotations

import importlib

from .instruction_defs import (
    TILE_M_GRANULARITY,
    TILE_N_GRANULARITY,
    DecodeAttentionReduction,
    DecodeAttentionSplitKV,
    DownProj_GEMM_Residual,
    OProj_GEMM_Residual,
    RMS_LMHead_GEMM,
    RMS_QKVProj_GEMM_RoPE,
    RMS_UpGateProj_GEMM_SiLU,
)

__all__ = [
    "TILE_M_GRANULARITY",
    "TILE_N_GRANULARITY",
    "DecodeAttentionReduction",
    "DecodeAttentionSplitKV",
    "DownProj_GEMM_Residual",
    "OProj_GEMM_Residual",
    "RMS_LMHead_GEMM",
    "RMS_QKVProj_GEMM_RoPE",
    "RMS_UpGateProj_GEMM_SiLU",
    "python_vm",
    "scheduler",
]


def __getattr__(name: str):
    if name == "scheduler":
        return importlib.import_module(".scheduler", __name__)
    if name == "python_vm":
        return importlib.import_module(".python_vm", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
