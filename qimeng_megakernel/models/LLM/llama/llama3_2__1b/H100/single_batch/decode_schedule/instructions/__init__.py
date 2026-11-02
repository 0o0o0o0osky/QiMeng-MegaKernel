"""
Instruction-based decode: instruction defs, scheduler (emit instructions), python_vm (handlers).

- instruction_defs: RMS_QKVProj_GEMV_RoPE, DecodeAttentionSplitKV, DecodeAttentionReduction,
  OProj_GEMV_Residual, RMS_UpGateProj_GEMV_SiLU, DownProj_GEMV_Residual,
  RMS_LMHead_GEMV.
- scheduler: build, schedule_qkvproj_gemv, schedule_oproj_gemv, schedule_lmhead_gemv, etc.
- python_vm: rms_qkvproj_gemv_rope, oproj_gemv_residual, rms_lmhead_gemv, etc.
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
from . import scheduler, python_vm

__all__ = [
    "BARRIER_ROW",
    "RMS_QKVProj_GEMV_RoPE",
    "DecodeAttentionSplitKV",
    "DecodeAttentionReduction",
    "OProj_GEMV_Residual",
    "RMS_UpGateProj_GEMV_SiLU",
    "DownProj_GEMV_Residual",
    "RMS_LMHead_GEMV",
    "scheduler",
    "python_vm",
]
