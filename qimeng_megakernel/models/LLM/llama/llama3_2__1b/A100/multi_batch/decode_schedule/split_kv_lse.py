"""Log2-domain LSE helpers for multi-batch SplitKV (legacy / CUDA alignment).

When ``Globals.attn_partial_valid`` is set (multi-batch PyVM), reduction uses **boolean**
per-partial validity instead of these thresholds; see ``instructions/python_vm.py``.

This module remains for: (1) **inactive-slot LSE filler** ``INACTIVE_PARTIAL_LSE``; (2)
**legacy** reduction when ``attn_partial_valid is None`` (must stay in sync with
``decode_attention_reduction.cu``: ``kSplitKvLseNonContributingLog2Leq``,
``kSplitKvLseMergedEmptyLog2``).
"""

from __future__ import annotations

# Stored in attn_lse_intermediates for empty / inactive partials (matches CUDA sentinel path).
INACTIVE_PARTIAL_LSE = float("-inf")

# Per-partial log2(sum exp): at or below this → non-contributing (CUDA clamps to -inf pre-exp2).
NONCONTRIBUTING_LSE_LOG2_LEQ = -79.0

# Non-terminal merge when zero partials contributed: strictly below NONCONTRIBUTING_LSE_LOG2_LEQ.
MERGED_EMPTY_LSE_LOG2 = NONCONTRIBUTING_LSE_LOG2_LEQ - 1.0
