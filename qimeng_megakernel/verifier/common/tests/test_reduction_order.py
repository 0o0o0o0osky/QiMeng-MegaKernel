"""CPU tests for exact BF16 split-K reduction-order diagnostics."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.verifier.common.reduction_order import (
    BF16_FOUR_PARTIAL_ORDERS,
    MAX_EXACT_BF16_PARTIALS,
    classify_bf16_partial_reduction,
    classify_bf16_four_partial_reduction,
    enumerate_bf16_partial_results,
    enumerate_bf16_four_partial_results,
    sequential_bf16_accumulate,
    sequential_bf16_accumulate_exact,
)


def _fixture() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    base = torch.tensor([0.0, 1.0, -2.0], dtype=torch.bfloat16)
    partials = tuple(
        torch.tensor(values, dtype=torch.bfloat16)
        for values in (
            (1.0, 0.125, 8.0),
            (2.0, -0.25, -4.0),
            (4.0, 0.5, 2.0),
            (8.0, -1.0, -1.0),
        )
    )
    return base, partials


def test_every_enumerated_four_partial_order_is_legal() -> None:
    base, partials = _fixture()
    candidates = enumerate_bf16_four_partial_results(base, partials)

    assert len(BF16_FOUR_PARTIAL_ORDERS) == 24
    assert len(set(BF16_FOUR_PARTIAL_ORDERS)) == 24
    assert candidates.shape == (24, *base.shape)
    for order in BF16_FOUR_PARTIAL_ORDERS:
        actual = sequential_bf16_accumulate(base, partials, order)
        classification = classify_bf16_four_partial_reduction(
            actual,
            base,
            partials,
        )
        assert bool(classification.legal_order_mask.all())
        assert not bool(classification.outside_legal_order_mask.any())


def test_envelope_preserves_rounding_after_each_arrival() -> None:
    base = torch.zeros(1, dtype=torch.bfloat16)
    partials = tuple(
        torch.tensor([value], dtype=torch.bfloat16)
        for value in (10_000.0, -10_000.0, 1.0, 1.0)
    )

    candidates = enumerate_bf16_four_partial_results(base, partials)

    # A float32 sum followed by one final cast is always 2.  Sequential BF16
    # destination rounding makes 0, 1, and 2 all reachable depending on order.
    assert sorted(set(candidates.float().flatten().tolist())) == [0.0, 1.0, 2.0]
    for value in (0.0, 1.0, 2.0):
        classification = classify_bf16_four_partial_reduction(
            torch.tensor([value], dtype=torch.bfloat16),
            base,
            partials,
        )
        assert classification.legal_order_mask.tolist() == [True]


def test_value_outside_all_complete_and_missing_one_envelopes_is_rejected() -> None:
    base = torch.zeros(1, dtype=torch.bfloat16)
    partials = tuple(
        torch.tensor([value], dtype=torch.bfloat16) for value in (1.0, 2.0, 4.0, 8.0)
    )
    actual = torch.tensor([100.0], dtype=torch.bfloat16)

    classification = classify_bf16_four_partial_reduction(actual, base, partials)

    assert classification.outside_legal_order_mask.tolist() == [True]
    assert classification.missing_one_any_mask.tolist() == [False]
    assert classification.outside_but_missing_one_mask.tolist() == [False]


def test_missing_one_membership_identifies_the_omitted_partial() -> None:
    base = torch.zeros(1, dtype=torch.bfloat16)
    partials = tuple(
        torch.tensor([value], dtype=torch.bfloat16) for value in (1.0, 2.0, 4.0, 8.0)
    )
    actual = sequential_bf16_accumulate(base, partials, (0, 1, 3))

    classification = classify_bf16_four_partial_reduction(actual, base, partials)

    assert classification.legal_order_mask.tolist() == [False]
    assert [mask.tolist() for mask in classification.missing_one_masks] == [
        [False],
        [False],
        [True],
        [False],
    ]
    assert classification.outside_but_missing_one_mask.tolist() == [True]


@pytest.mark.parametrize(
    ("actual", "base", "partials", "error", "message"),
    [
        (
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.bfloat16),
            tuple(torch.zeros(2, dtype=torch.bfloat16) for _ in range(4)),
            TypeError,
            "actual must have dtype torch.bfloat16",
        ),
        (
            torch.zeros(2, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.bfloat16),
            (
                torch.zeros(2, dtype=torch.bfloat16),
                torch.zeros(2, dtype=torch.float32),
                torch.zeros(2, dtype=torch.bfloat16),
                torch.zeros(2, dtype=torch.bfloat16),
            ),
            TypeError,
            r"partial\[1\] must have dtype torch.bfloat16",
        ),
        (
            torch.zeros(3, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.bfloat16),
            tuple(torch.zeros(2, dtype=torch.bfloat16) for _ in range(4)),
            ValueError,
            "actual shape",
        ),
        (
            torch.zeros(2, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.bfloat16),
            (
                torch.zeros(2, dtype=torch.bfloat16),
                torch.zeros(2, dtype=torch.bfloat16),
                torch.zeros(3, dtype=torch.bfloat16),
                torch.zeros(2, dtype=torch.bfloat16),
            ),
            ValueError,
            r"partial\[2\] shape",
        ),
        (
            torch.zeros(2, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.bfloat16),
            tuple(torch.zeros(2, dtype=torch.bfloat16) for _ in range(3)),
            ValueError,
            "exactly four partials",
        ),
    ],
    ids=("actual-dtype", "partial-dtype", "actual-shape", "partial-shape", "arity"),
)
def test_shape_dtype_and_arity_validation_fail_closed(
    actual: torch.Tensor,
    base: torch.Tensor,
    partials: tuple[torch.Tensor, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        classify_bf16_four_partial_reduction(actual, base, partials)


def test_empty_and_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        classify_bf16_four_partial_reduction(
            torch.empty(0, dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.bfloat16),
            tuple(torch.empty(0, dtype=torch.bfloat16) for _ in range(4)),
        )

    base, partials = _fixture()
    bad_partials = list(partials)
    bad_partials[3] = bad_partials[3].clone()
    bad_partials[3][0] = float("nan")
    with pytest.raises(ValueError, match=r"partial\[3\] contains non-finite"):
        classify_bf16_four_partial_reduction(base, base, tuple(bad_partials))


def test_generic_six_partial_envelope_enumerates_every_order() -> None:
    base = torch.tensor([1.0, -2.0, 0.25], dtype=torch.bfloat16)
    partials = tuple(
        torch.tensor(
            [2.0 ** (-index), (-1.0) ** index * (index + 1), index / 16],
            dtype=torch.bfloat16,
        )
        for index in range(6)
    )
    actual = sequential_bf16_accumulate_exact(base, partials, (5, 3, 1, 0, 2, 4))
    candidates = enumerate_bf16_partial_results(base, partials)
    classification = classify_bf16_partial_reduction(actual, base, partials)

    assert candidates.shape == (720, 3)
    assert classification.partial_count == 6
    assert classification.legal_order_count == 720
    assert bool(classification.legal_order_mask.all().item())


@pytest.mark.parametrize(
    ("partials", "message"),
    (
        ((torch.ones(1, dtype=torch.bfloat16),), "between 2 and"),
        (
            tuple(
                torch.ones(1, dtype=torch.bfloat16)
                for _ in range(MAX_EXACT_BF16_PARTIALS + 1)
            ),
            "between 2 and",
        ),
    ),
)
def test_generic_exact_envelope_fails_closed_outside_resource_bound(
    partials: tuple[torch.Tensor, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        enumerate_bf16_partial_results(
            torch.zeros(1, dtype=torch.bfloat16), partials
        )


def test_generic_exact_accumulation_requires_a_full_permutation() -> None:
    base = torch.zeros(2, dtype=torch.bfloat16)
    partials = tuple(
        torch.full((2,), float(index + 1), dtype=torch.bfloat16)
        for index in range(6)
    )
    with pytest.raises(ValueError, match="order length must equal partial count"):
        sequential_bf16_accumulate_exact(base, partials, (0, 1, 2, 3, 4))
    for bad_order in ((0, 1, 2, 3, 4, 4), (0, 1, 2, 3, 4, 6)):
        with pytest.raises(ValueError, match="every partial index exactly once"):
            sequential_bf16_accumulate_exact(base, partials, bad_order)


@pytest.mark.parametrize(
    ("bad_order", "message"),
    (
        ([0, 1, 2, 3, 4, 5], "exact tuple"),
        ((False, 1, 2, 3, 4, 5), r"order\[0\].*bool"),
        ((0.0, 1, 2, 3, 4, 5), r"order\[0\].*float"),
        ((0, "1", 2, 3, 4, 5), r"order\[1\].*str"),
    ),
    ids=("list", "bool", "float", "string"),
)
def test_generic_exact_accumulation_rejects_untyped_order_values(
    bad_order: object, message: str
) -> None:
    base = torch.zeros(2, dtype=torch.bfloat16)
    partials = tuple(
        torch.full((2,), float(index + 1), dtype=torch.bfloat16)
        for index in range(6)
    )

    with pytest.raises(TypeError, match=message):
        sequential_bf16_accumulate_exact(base, partials, bad_order)
