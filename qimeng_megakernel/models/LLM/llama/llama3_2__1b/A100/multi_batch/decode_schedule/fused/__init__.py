# Re-export single_batch fused ops (already support B>1).
from __future__ import annotations

import importlib

__all__ = ["layer_scheduler", "operations"]


def __getattr__(name: str):
    if name == "operations":
        return importlib.import_module(
            "qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.single_batch.decode_schedule.fused.operations"
        )
    if name == "layer_scheduler":
        return importlib.import_module(
            "qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.single_batch.decode_schedule.fused.layer_scheduler"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
