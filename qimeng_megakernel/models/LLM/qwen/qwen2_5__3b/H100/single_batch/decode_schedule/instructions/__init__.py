"""Qwen2.5 3B decode_schedule instructions (full decode)."""
from .instruction_defs import (
    BARRIER_ROW,
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
    TILE_N_GRANULARITY,
    NUM_DOWNPROJ_SPLIT_K,
    DOWNPROJ_BARRIER_COUNT,
)
from . import scheduler
from . import instruction_defs
from . import python_vm

__all__ = [
    "BARRIER_ROW",
    "RMS_QKVProj_GEMV_RoPE",
    "DecodeAttentionSplitKV",
    "DecodeAttentionReduction",
    "OProj_GEMV_Residual",
    "RMS_UpGateProj_GEMV_SiLU",
    "DownProj_GEMV_Residual",
    "RMS_LMHead_GEMV",
    "TILE_N_GRANULARITY",
    "NUM_DOWNPROJ_SPLIT_K",
    "DOWNPROJ_BARRIER_COUNT",
    "scheduler",
    "instruction_defs",
    "python_vm",
]
