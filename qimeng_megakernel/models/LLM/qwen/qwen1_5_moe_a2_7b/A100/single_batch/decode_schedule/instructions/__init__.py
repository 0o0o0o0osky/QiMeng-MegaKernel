"""Instructions (opcode defs, scheduler, python_vm) for Qwen1.5-MoE-A2.7B A100 decode."""

from . import instruction_defs
from . import scheduler
from . import python_vm
from .instruction_defs import (
    RMS_QKVProj_GEMV_RoPE,
    DecodeAttentionSplitKV,
    DecodeAttentionReduction,
    OProj_GEMV_Residual,
    Router_GEMV,
    Router_SoftmaxTopK,
    SharedExpert_UpGate_SiLU,
    SharedExpert_DownProj,
    RoutedExpert_UpGate_SiLU,
    RoutedExpert_DownProj,
    MoE_Combine_Commit,
    RMS_LMHead_GEMV,
)

__all__ = [
    "instruction_defs",
    "scheduler",
    "python_vm",
    "RMS_QKVProj_GEMV_RoPE",
    "DecodeAttentionSplitKV",
    "DecodeAttentionReduction",
    "OProj_GEMV_Residual",
    "Router_GEMV",
    "Router_SoftmaxTopK",
    "SharedExpert_UpGate_SiLU",
    "SharedExpert_DownProj",
    "RoutedExpert_UpGate_SiLU",
    "RoutedExpert_DownProj",
    "MoE_Combine_Commit",
    "RMS_LMHead_GEMV",
]
