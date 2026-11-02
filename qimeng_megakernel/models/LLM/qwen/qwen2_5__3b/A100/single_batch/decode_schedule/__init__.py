"""Qwen2.5 3B A100 single-batch decode schedule."""
from __future__ import annotations

import importlib

from .globs import Globals
from .model_config import Qwen2Config

ModelConfig = Qwen2Config

__all__ = [
    "Globals",
    "ModelConfig",
    "Qwen2Config",
    "fused_operations",
    "layer_scheduler",
    "instructions",
    "instruction_defs",
    "scheduler",
    "python_vm",
    "mk_kernel",
]


def __getattr__(name: str):
    if name == "fused_operations":
        return importlib.import_module(".fused.operations", __name__)
    if name == "layer_scheduler":
        return importlib.import_module(".fused.layer_scheduler", __name__)
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
