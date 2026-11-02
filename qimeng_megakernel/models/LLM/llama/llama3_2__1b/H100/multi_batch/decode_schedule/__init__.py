"""Standard H100 multi-batch decode-schedule interface."""

from __future__ import annotations

import importlib

from .globs import BATCH_CAPACITY, MAX_BATCH_SIZE, Globals
from .model_config import LlamaConfig, ModelConfig

__all__ = [
    "MAX_BATCH_SIZE",
    "BATCH_CAPACITY",
    "Globals",
    "LlamaConfig",
    "ModelConfig",
    "fused_operations",
    "instruction_defs",
    "instructions",
    "layer_scheduler",
    "mk_kernel",
    "python_vm",
    "scheduler",
]


def __getattr__(name: str):
    if name == "fused_operations":
        return importlib.import_module(".fused", __name__).operations
    if name == "layer_scheduler":
        return importlib.import_module(".fused", __name__).layer_scheduler
    if name == "instructions":
        return importlib.import_module(".instructions", __name__)
    if name == "instruction_defs":
        return importlib.import_module(".instructions.instruction_defs", __name__)
    if name == "scheduler":
        return importlib.import_module(".instructions.scheduler", __name__)
    if name == "python_vm":
        return importlib.import_module(".instructions.python_vm", __name__)
    if name == "mk_kernel":
        return importlib.import_module(".mk_kernel", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
