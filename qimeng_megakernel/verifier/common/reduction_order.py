"""Exact BF16 reduction-order envelopes for split-K diagnostics.

This module is deliberately a diagnostic primitive, not a verifier gate.  It
answers one narrow question: if a BF16 destination starts at ``base`` and four
BF16 partials arrive one at a time, can ``actual`` be produced by any legal
arrival order when every store rounds the destination back to BF16?

The missing-one envelope is reported independently.  A value can belong to
both envelopes (for example when an omitted partial is zero), so callers must
not turn missing-one membership into a missing-write claim unless the value is
also outside the complete four-partial envelope.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import Sequence

import torch


BF16_FOUR_PARTIAL_ORDERS: tuple[tuple[int, ...], ...] = tuple(
    permutations(range(4))
)

MAX_EXACT_BF16_PARTIALS = 7


@dataclass(frozen=True)
class BF16ExactOrderClassification:
    """Per-element membership in an exact sequential-BF16 order envelope."""

    partial_count: int
    legal_order_count: int
    legal_order_mask: torch.Tensor

    @property
    def outside_legal_order_mask(self) -> torch.Tensor:
        return ~self.legal_order_mask


@dataclass(frozen=True)
class BF16ReductionOrderClassification:
    """Per-element membership in complete and missing-one BF16 envelopes."""

    legal_order_mask: torch.Tensor
    missing_one_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def outside_legal_order_mask(self) -> torch.Tensor:
        return ~self.legal_order_mask

    @property
    def missing_one_any_mask(self) -> torch.Tensor:
        result = self.missing_one_masks[0].clone()
        for mask in self.missing_one_masks[1:]:
            result |= mask
        return result

    @property
    def outside_but_missing_one_mask(self) -> torch.Tensor:
        return self.outside_legal_order_mask & self.missing_one_any_mask


def _validate_bf16_inputs(
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
    actual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(base, torch.Tensor):
        raise TypeError(f"base must be a Tensor, got {type(base).__name__}")
    if len(partials) != 4:
        raise ValueError(f"exactly four partials are required, got {len(partials)}")

    named_tensors: list[tuple[str, torch.Tensor]] = [("base", base)]
    for index, partial in enumerate(partials):
        if not isinstance(partial, torch.Tensor):
            raise TypeError(
                f"partial[{index}] must be a Tensor, got {type(partial).__name__}"
            )
        named_tensors.append((f"partial[{index}]", partial))
    if actual is not None:
        if not isinstance(actual, torch.Tensor):
            raise TypeError(f"actual must be a Tensor, got {type(actual).__name__}")
        named_tensors.append(("actual", actual))

    if base.numel() == 0:
        raise ValueError("BF16 reduction-order inputs must be non-empty")
    for name, tensor in named_tensors:
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")
        if tensor.shape != base.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match base "
                f"shape {tuple(base.shape)}"
            )
        if tensor.device != base.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match base device {base.device}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains non-finite BF16 values")

    return (partials[0], partials[1], partials[2], partials[3])


def _validate_exact_bf16_inputs(
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
    actual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Validate the bounded generic exact-order primitive.

    Exact enumeration grows factorially.  Seven partials (5040 orders) is a hard
    resource bound, not a numerical tolerance.  Larger reductions need a different
    proof representation instead of silently truncating the lawful set.
    """
    if not isinstance(base, torch.Tensor):
        raise TypeError(f"base must be a Tensor, got {type(base).__name__}")
    if isinstance(partials, (str, bytes)) or not isinstance(partials, Sequence):
        raise TypeError("partials must be a sequence of Tensor values")
    partial_values = tuple(partials)
    if not 2 <= len(partial_values) <= MAX_EXACT_BF16_PARTIALS:
        raise ValueError(
            "exact BF16 order enumeration requires between 2 and "
            f"{MAX_EXACT_BF16_PARTIALS} partials, got {len(partial_values)}"
        )

    named_tensors: list[tuple[str, torch.Tensor]] = [("base", base)]
    for index, partial in enumerate(partial_values):
        if not isinstance(partial, torch.Tensor):
            raise TypeError(
                f"partial[{index}] must be a Tensor, got {type(partial).__name__}"
            )
        named_tensors.append((f"partial[{index}]", partial))
    if actual is not None:
        if not isinstance(actual, torch.Tensor):
            raise TypeError(f"actual must be a Tensor, got {type(actual).__name__}")
        named_tensors.append(("actual", actual))

    if base.numel() == 0:
        raise ValueError("BF16 reduction-order inputs must be non-empty")
    for name, tensor in named_tensors:
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")
        if tensor.shape != base.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match base "
                f"shape {tuple(base.shape)}"
            )
        if tensor.device != base.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match base device {base.device}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains non-finite BF16 values")
    return partial_values


def sequential_bf16_accumulate_exact(
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
    order: object,
) -> torch.Tensor:
    """Accumulate every supplied partial exactly once, rounding after each add."""
    validated = _validate_exact_bf16_inputs(base, partials)
    order_tuple = _validate_exact_order(order, len(validated))
    return _sequential_bf16_accumulate_validated(base, validated, order_tuple)


def _validate_exact_order(order: object, partial_count: int) -> tuple[int, ...]:
    """Require one typed tuple containing the exact integer permutation."""
    if not isinstance(order, tuple):
        raise TypeError("order must be an exact tuple of non-bool int indices")
    if len(order) != partial_count:
        raise ValueError(
            f"order length must equal partial count {partial_count}, got {len(order)}"
        )
    typed_order: list[int] = []
    for index, value in enumerate(order):
        if type(value) is not int:
            raise TypeError(
                f"order[{index}] must be an exact non-bool int, got "
                f"{type(value).__name__}"
            )
        typed_order.append(value)
    expected = tuple(range(partial_count))
    order_tuple = tuple(typed_order)
    if tuple(sorted(order_tuple)) != expected:
        raise ValueError(
            "order must contain every partial index exactly once; expected "
            f"{expected}, got {order_tuple}"
        )
    return order_tuple


def _sequential_bf16_accumulate_validated(
    base: torch.Tensor,
    partials: tuple[torch.Tensor, ...],
    order: tuple[int, ...],
) -> torch.Tensor:
    """Hot inner loop after one caller-level structure/finite validation."""
    accumulator = base.clone()
    for index in order:
        accumulator = (accumulator.float() + partials[index].float()).to(
            torch.bfloat16
        )
    return accumulator


def enumerate_bf16_partial_results(
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Enumerate the complete bounded sequential-BF16 accumulation set."""
    validated = _validate_exact_bf16_inputs(base, partials)
    orders = tuple(permutations(range(len(validated))))
    order_indices = torch.tensor(orders, dtype=torch.int64, device=base.device)
    partial_stack = torch.stack(validated, dim=0)
    accumulator = base.unsqueeze(0).expand(len(orders), *base.shape).clone()
    for step in range(len(validated)):
        selected = partial_stack.index_select(0, order_indices[:, step])
        accumulator = (accumulator.float() + selected.float()).to(torch.bfloat16)
    return accumulator


def classify_bf16_partial_reduction(
    actual: torch.Tensor,
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
) -> BF16ExactOrderClassification:
    """Classify an output against every lawful order for two through seven partials."""
    validated = _validate_exact_bf16_inputs(base, partials, actual)
    candidates = enumerate_bf16_partial_results(base, validated)
    return BF16ExactOrderClassification(
        partial_count=len(validated),
        legal_order_count=factorial(len(validated)),
        legal_order_mask=_bitwise_membership(actual, candidates),
    )


def sequential_bf16_accumulate(
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
    order: Sequence[int],
) -> torch.Tensor:
    """Accumulate selected partials, rounding the destination after every add."""
    validated = _validate_bf16_inputs(base, partials)
    order_tuple = tuple(order)
    if len(set(order_tuple)) != len(order_tuple):
        raise ValueError(f"order contains duplicate partial indices: {order_tuple}")
    if any(index < 0 or index >= 4 for index in order_tuple):
        raise ValueError(f"order indices must be in [0, 4), got {order_tuple}")

    accumulator = base.clone()
    for index in order_tuple:
        accumulator = (accumulator.float() + validated[index].float()).to(
            torch.bfloat16
        )
    return accumulator


def enumerate_bf16_four_partial_results(
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return all 4! sequential-BF16 results in the public order table."""
    _validate_bf16_inputs(base, partials)
    return torch.stack(
        [
            sequential_bf16_accumulate(base, partials, order)
            for order in BF16_FOUR_PARTIAL_ORDERS
        ],
        dim=0,
    )


def _bitwise_membership(actual: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    # bfloat16 and int16 have the same item size.  Comparing the integer view
    # keeps this an exact diagnostic, including the sign of zero.
    actual_bits = actual.contiguous().view(torch.int16)
    candidate_bits = candidates.contiguous().view(torch.int16)
    return (candidate_bits == actual_bits.unsqueeze(0)).any(dim=0)


def classify_bf16_four_partial_reduction(
    actual: torch.Tensor,
    base: torch.Tensor,
    partials: Sequence[torch.Tensor],
) -> BF16ReductionOrderClassification:
    """Classify ``actual`` against complete and 3-of-4 BF16 order sets."""
    validated = _validate_bf16_inputs(base, partials, actual)
    legal_candidates = enumerate_bf16_four_partial_results(base, validated)
    legal_order_mask = _bitwise_membership(actual, legal_candidates)

    missing_one_masks: list[torch.Tensor] = []
    for omitted in range(4):
        present = tuple(index for index in range(4) if index != omitted)
        candidates = torch.stack(
            [
                sequential_bf16_accumulate(base, validated, order)
                for order in permutations(present)
            ],
            dim=0,
        )
        missing_one_masks.append(_bitwise_membership(actual, candidates))

    return BF16ReductionOrderClassification(
        legal_order_mask=legal_order_mask,
        missing_one_masks=(
            missing_one_masks[0],
            missing_one_masks[1],
            missing_one_masks[2],
            missing_one_masks[3],
        ),
    )


__all__ = [
    "BF16_FOUR_PARTIAL_ORDERS",
    "MAX_EXACT_BF16_PARTIALS",
    "BF16ExactOrderClassification",
    "BF16ReductionOrderClassification",
    "classify_bf16_partial_reduction",
    "classify_bf16_four_partial_reduction",
    "enumerate_bf16_partial_results",
    "enumerate_bf16_four_partial_results",
    "sequential_bf16_accumulate",
    "sequential_bf16_accumulate_exact",
]
