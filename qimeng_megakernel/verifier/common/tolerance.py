"""bf16 / fp16 / int8 rtol/atol decision table -- the canonical copy.

This module is the **source of truth**. New model suites import these values rather
than carrying a local copy, so they cannot start life with a silently forked table.

Policy:

* every threshold asserted anywhere in `verifier/` must be traceable either to this
  module or to that suite's `verifier_config.py` -- no bare literals in `test_*.py`;
* op-level comparisons are elementwise (`max_rel` + `abs_floor` guard). Cosine is
  used for S0 and S1-S3 module depth; see `reference/numerical-verification.md`;
* relaxing any gate requires calibration evidence in the config that relaxes it.
"""
from __future__ import annotations

import torch


# Tier rules (loose -> strict):
#
# unit_synthetic / unit_real (single op, 1 layer): strictest; catches most operator bugs
# single_layer  (one transformer block)          : medium; one RMS+QKV+attn+OProj+RMS+MLP accumulation
# multi_layer   (all N layers)                   : loose; N-layer accumulation
# e2e_generate  (end to end + greedy)            : top-1 agreement rather than a numeric delta
_TOLERANCE_TABLE = {
    # (dtype_str, scope) -> (rtol, atol, accumulate_factor)
    ("bf16", "unit"):         (5e-2, 5e-3, 1.0),
    ("bf16", "single_layer"): (8e-2, 1e-2, 1.0),
    ("bf16", "multi_layer"):  (1e-1, 5e-2, 0.05),
    ("fp16", "unit"):         (3e-2, 3e-3, 1.0),
    ("fp16", "single_layer"): (5e-2, 8e-3, 1.0),
    ("fp16", "multi_layer"):  (8e-2, 3e-2, 0.04),
    ("fp32", "unit"):         (1e-4, 1e-5, 1.0),
    ("fp32", "single_layer"): (1e-3, 1e-4, 1.0),
    ("fp32", "multi_layer"):  (5e-3, 1e-3, 0.01),
    ("int8", "unit"):         (1e-1, 5e-2, 1.0),
    ("int8", "single_layer"): (2e-1, 1e-1, 1.0),
    ("int8", "multi_layer"):  (3e-1, 2e-1, 0.05),
}


def _dtype_to_str(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.int8:    "int8",
        torch.float8_e4m3fn: "int8",
    }.get(dtype, "fp32")


def pick_tolerance(
    dtype: torch.dtype,
    scope: str,
    accumulate_layers: int = 1,
) -> tuple[float, float]:
    """Return (rtol, atol) for a precision tier / verification scope / layer count.

    Decision tree:
      1. dtype picks the base tier (bf16 is ~1.5x looser than fp16; fp32 strict; int8/fp8 loosest)
      2. scope weights by accumulation depth: unit / single block / multi_layer
      3. accumulate_layers > 1 (multi_layer only) widens atol by (1 + factor) ** (layers - 1)
    """
    if scope not in {"unit", "single_layer", "multi_layer"}:
        raise ValueError(f"unknown scope={scope}; choose unit / single_layer / multi_layer")
    key = (_dtype_to_str(dtype), scope)
    if key not in _TOLERANCE_TABLE:
        raise KeyError(f"no tolerance for {key}")
    rtol, atol, accum = _TOLERANCE_TABLE[key]
    if scope == "multi_layer" and accumulate_layers > 1:
        atol = atol * ((1.0 + accum) ** (accumulate_layers - 1))
    return rtol, atol


def report_max_rel(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs, max_rel) so a failing verifier can print a reproduction command."""
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-9)
    return float(diff.max()), float(rel.max())


# E2E L5 gate thresholds (deliberately outside the main table; verifiers must not
# invent substitutes).
E2E_TOP1_MIN_AGREE = 0.95
E2E_FORK_TOPK_K = 5
E2E_FORK_TOPK_MIN_OVERLAP = 4
E2E_FORK_TOPK_MIN_PROBABILITY_OVERLAP = 0.70

# Norm-ratio guard that accompanies every stage-level cosine gate (cosine is blind to
# a uniform rescale; see `compare.norm_ratio`). Kept report-only until a full
# production route green run has produced a distribution; see `compare.check_norm_ratio`.
NORM_RATIO_MIN = 0.98
NORM_RATIO_MAX = 1.02
