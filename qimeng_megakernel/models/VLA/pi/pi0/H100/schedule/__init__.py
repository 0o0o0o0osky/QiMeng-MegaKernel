"""pi0 / H100 schedule — LLM-style fused + pyvm + instruction defs.

Named ``schedule`` (not ``decode_schedule``): pi0 is a VLA flow policy, not an
autoregressive decode model. Layout still mirrors
``models/LLM/llama/.../decode_schedule/``:

  fused/           torch full-op references (check fusion correctness)
  instructions/    instruction_defs + scheduler + python_vm (tiled handlers)

Kernels remain in sibling ``../kernel/`` (mk_siglip_view1/2/3, mk_gemma, mk_ae).
"""
from __future__ import annotations

import importlib

from .globs import AeGlobals, GemmaGlobals, SiglipGlobals
from .pi0_config import Pi0H100Config

__all__ = [
    "Pi0H100Config",
    "SiglipGlobals",
    "GemmaGlobals",
    "AeGlobals",
    "fused_operations",
    "python_vm",
    "scheduler",
]


def __getattr__(name: str):
    if name == "fused_operations":
        return importlib.import_module(".fused.operations", __name__)
    if name == "python_vm":
        return importlib.import_module(".instructions.python_vm", __name__)
    if name == "scheduler":
        return importlib.import_module(".instructions.scheduler", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
