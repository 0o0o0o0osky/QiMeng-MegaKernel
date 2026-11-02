"""Shared verifier skeleton -- the one package every model suite may import from.


===================  ========================================================
`compare`            numeric criteria: elementwise (max_rel + abs_floor),
                     S0/module-depth cosine (+ max_abs report + norm-ratio guard),
                     and the `InstructionTestCase` sequence runner.
`verification_schema` shared depth/role names and fixed numeric metric schema.
`tolerance`          canonical rtol/atol decision table and the L5 gate
                     constants.  Source of truth for
                     the shared verifier configuration.
`mk_test`            `MKTestConfig` / `run_test`: PyVM vs megakernel harness.
`generation`         lightweight helpers used by LLM greedy-generation tests.
`barrier_fill`       parameterised barrier pre-fill for isolated-op tests.
`real_data`          real-prefill `Globals` construction + backup/restore.
`copy_sync`          read-only verifier/production kernel source identity.
`verifier_kernel`    build/load the exact verifier-owned DSO for S3.
`stage_config`       `ModelVerifierConfig` and the gate dataclasses that each
                     suite's `verifier_config.py` instantiates.
===================  ========================================================

Submodules are exposed lazily: `generation` and `real_data` pull in
`qimeng_megakernel.models.*` (and therefore a full torch + transformers import),
which must not happen just because a suite wanted `compare`.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBMODULES = (
    "barrier_fill",
    "compare",
    "verification_schema",
    "copy_sync",
    "verifier_kernel",
    "generation",
    "mk_test",
    "real_data",
    "reduction_order",
    "stage_config",
    "tolerance",
)

__all__ = list(_SUBMODULES)

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from . import (  # noqa: F401
        barrier_fill,
        compare,
        verification_schema,
        copy_sync,
        verifier_kernel,
        generation,
        mk_test,
        real_data,
        reduction_order,
        stage_config,
        tolerance,
    )


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
