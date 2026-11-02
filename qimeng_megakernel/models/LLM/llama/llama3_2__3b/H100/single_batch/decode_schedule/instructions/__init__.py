"""
3B H100 instructions: instruction_defs, scheduler, python_vm (same instruction set as 1B; uses 3B LlamaConfig).
"""
from .instruction_defs import (
    BARRIER_ROW,
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    RMS_UpGateProj_GEMV_SiLU,
    DownProj_GEMV_Residual,
    RMS_LMHead_GEMV,
)
from . import instruction_defs
from . import scheduler
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
    "instruction_defs",
    "scheduler",
    "python_vm",
]
