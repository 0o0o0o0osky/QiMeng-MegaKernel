"""Numeric criteria for every verifier stage, in one place.

Three comparison families, each with a defined verification depth:

**Elementwise** (`evaluate_numeric`, `max_rel_with_abs_floor`, `assert_max_rel`)
    The yardstick for S1/S2/S3 op and layer depth.  It is the one reference-relative
    piecewise formula: relative above ``abs_floor``, absolute below it.  Tensors named
    by ``ElementwiseGate.near_zero`` use a whole-tensor absolute gate instead.

**Cosine bundle** (`evaluate_numeric`)
    Legitimate only for S1/S2/S3 module depth and S0.  Cosine, max/mean absolute error,
    and norm ratio are always emitted as one bundle; the ratio closes cosine's uniform
    scale blind spot.

``compare_tensor_stats`` remains a compatibility helper for suites that have not
migrated.  New tests should use the depth-specific comparison functions above.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypedDict, cast

import torch
from torch import Tensor

from qimeng_megakernel.models.instruction import Instruction, run_pyvm_sequence

from .verification_schema import CANONICAL_NUMERIC_METRIC_SCHEMA
from .stage_config import CosineGate, ElementwiseGate, NumericDepth
from .tolerance import NORM_RATIO_MAX, NORM_RATIO_MIN

__all__ = [
    "VerificationResult",
    "InstructionTestCase",
    "NUMERIC_SCHEMA_VERSION",
    "NUMERIC_RECORD_KEYS",
    "NUMERIC_THRESHOLD_KEYS",
    "NUMERIC_METRIC_KEYS",
    "NumericRecordContext",
    "NumericThresholdDict",
    "NumericMetricsDict",
    "NumericRecordDict",
    "NumericThresholdReport",
    "NumericMetrics",
    "NumericRecord",
    "NumericGateError",
    "append_numeric_record_jsonl",
    "PiecewiseMetrics",
    "NumericStage",
    "MetricFamily",
    "NumericThresholds",
    "evaluate_numeric",
    "validate_persisted_numeric_pass",
    "assert_numeric",
    "assert_numeric_active_finite",
    "inspect_rel_worst",
    "rel_error_near_zero_report",
    "compare_tensor_stats",
    "verify_instruction_sequence",
    "run_pyvm_sequence",
    "Instruction",
    "cosine",
    "max_abs",
    "mean_abs",
    "norm_ratio",
    "check_norm_ratio",
    "assert_close",
    "max_rel_with_abs_floor",
    "assert_max_rel",
]


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    diffs: list[str]
    stats: dict[str, dict[str, float]]


@dataclass(frozen=True)
class InstructionTestCase:
    name: str
    instructions: list[Instruction]
    handler_map: dict[type[Instruction], Callable[[Any, Instruction], None]]
    expected_fn: Callable[..., dict[str, torch.Tensor]]
    compare_tensors: Iterable[str]
    rtol: float = 1e-2
    atol: float = 1e-2


# ---------------------------------------------------------------------------
# Fixed-schema S0-S3 records
# ---------------------------------------------------------------------------

NUMERIC_SCHEMA_VERSION = "1.2"

NUMERIC_RECORD_KEYS = (
    "schema_version",
    "model_key",
    "model_class",
    "hardware",
    "stage",
    "depth",
    "case_id",
    "actual_route",
    "expected_route",
    "tensor",
    "actual_dtype",
    "expected_dtype",
    "actual_shape",
    "expected_shape",
    "metric_family",
    "thresholds",
    "metrics",
    "pass",
    "command",
    "commit",
)

NUMERIC_THRESHOLD_KEYS = (
    "max_rel",
    "abs_floor",
    "atol",
    "min_cosine",
    "max_abs",
    "norm_ratio_min",
    "norm_ratio_max",
)

NUMERIC_METRIC_KEYS = CANONICAL_NUMERIC_METRIC_SCHEMA.keys

NumericStage = Literal["S0", "S1", "S2", "S3"]
MetricFamily = Literal["elementwise", "near_zero", "cosine_bundle"]
NumericThresholds = ElementwiseGate | CosineGate


class PiecewiseMetrics(TypedDict):
    """Exact output of the sole reference-relative elementwise formula."""

    max_rel_above: float
    max_abs_below: float
    n_above: int
    n_below: int


class NumericThresholdDict(TypedDict):
    max_rel: float | None
    abs_floor: float | None
    atol: float | None
    min_cosine: float | None
    max_abs: float | None
    norm_ratio_min: float | None
    norm_ratio_max: float | None


class NumericMetricsDict(TypedDict):
    shape_match: bool
    nonempty_actual: bool
    nonempty_expected: bool
    finite_actual: bool
    finite_expected: bool
    finite_derived: bool | None
    max_rel_above: float | None
    max_abs_below: float | None
    n_above: int | None
    n_below: int | None
    max_abs: float | None
    mean_abs: float | None
    cosine: float | None
    norm_ratio: float | None


NumericRecordDict = TypedDict(
    "NumericRecordDict",
    {
        "schema_version": str,
        "model_key": str,
        "model_class": str,
        "hardware": str,
        "stage": str,
        "depth": str,
        "case_id": str,
        "actual_route": str,
        "expected_route": str,
        "tensor": str,
        "actual_dtype": str,
        "expected_dtype": str,
        "actual_shape": list[int],
        "expected_shape": list[int],
        "metric_family": str,
        "thresholds": NumericThresholdDict,
        "metrics": NumericMetricsDict,
        "pass": bool,
        "command": str,
        "commit": str,
    },
)


@dataclass(frozen=True)
class NumericRecordContext:
    """Metadata for one shared S0--S3 numeric comparison record."""

    model_key: str
    model_class: str
    hardware: str
    stage: NumericStage
    depth: NumericDepth
    case_id: str
    actual_route: str
    expected_route: str
    tensor: str
    command: str
    commit: str

    def __post_init__(self) -> None:
        if self.stage not in {"S0", "S1", "S2", "S3"}:
            raise ValueError(
                f"numeric record stage must be S0/S1/S2/S3, got {self.stage!r}"
            )
        if self.depth not in {"op", "layer", "module"}:
            raise ValueError(
                f"numeric record depth must be op/layer/module, got {self.depth!r}"
            )
        if self.stage == "S0" and self.depth != "module":
            raise ValueError("legacy S0 numeric records require depth='module'")
        for name, value in (
            ("model_key", self.model_key),
            ("model_class", self.model_class),
            ("hardware", self.hardware),
            ("case_id", self.case_id),
            ("actual_route", self.actual_route),
            ("expected_route", self.expected_route),
            ("tensor", self.tensor),
            ("command", self.command),
            ("commit", self.commit),
        ):
            if not isinstance(value, str):
                raise TypeError(f"NumericRecordContext.{name} must be a string")
            if not value.strip():
                raise ValueError(f"NumericRecordContext.{name} must be non-empty")
            if value != value.strip():
                raise ValueError(
                    f"NumericRecordContext.{name} must not have surrounding whitespace"
                )

@dataclass(frozen=True)
class NumericThresholdReport:
    """Fixed threshold object; values not used by a metric family stay ``None``."""

    max_rel: float | None = None
    abs_floor: float | None = None
    atol: float | None = None
    min_cosine: float | None = None
    max_abs: float | None = None
    norm_ratio_min: float | None = None
    norm_ratio_max: float | None = None

    def as_dict(self) -> NumericThresholdDict:
        return {
            "max_rel": self.max_rel,
            "abs_floor": self.abs_floor,
            "atol": self.atol,
            "min_cosine": self.min_cosine,
            "max_abs": self.max_abs,
            "norm_ratio_min": self.norm_ratio_min,
            "norm_ratio_max": self.norm_ratio_max,
        }


@dataclass(frozen=True)
class NumericMetrics:
    """Fixed metric object; non-applicable metrics stay ``None`` rather than vanish."""

    shape_match: bool
    nonempty_actual: bool
    nonempty_expected: bool
    finite_actual: bool
    finite_expected: bool
    finite_derived: bool | None = None
    max_rel_above: float | None = None
    max_abs_below: float | None = None
    n_above: int | None = None
    n_below: int | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    cosine: float | None = None
    norm_ratio: float | None = None

    def as_dict(self) -> NumericMetricsDict:
        values: dict[str, object] = {
            "shape_match": self.shape_match,
            "nonempty_actual": self.nonempty_actual,
            "nonempty_expected": self.nonempty_expected,
            "finite_actual": self.finite_actual,
            "finite_expected": self.finite_expected,
            "finite_derived": self.finite_derived,
            "max_rel_above": self.max_rel_above,
            "max_abs_below": self.max_abs_below,
            "n_above": self.n_above,
            "n_below": self.n_below,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "cosine": self.cosine,
            "norm_ratio": self.norm_ratio,
        }
        CANONICAL_NUMERIC_METRIC_SCHEMA.validate_payload(values)
        return cast(NumericMetricsDict, values)


@dataclass(frozen=True)
class NumericRecord:
    """One JSON-safe S0-S3 tensor comparison record."""

    context: NumericRecordContext
    actual_dtype: str
    expected_dtype: str
    actual_shape: tuple[int, ...]
    expected_shape: tuple[int, ...]
    metric_family: MetricFamily
    thresholds: NumericThresholdReport
    metrics: NumericMetrics
    passed: bool
    def as_dict(self) -> NumericRecordDict:
        values: dict[str, object] = {
            "schema_version": NUMERIC_SCHEMA_VERSION,
            "model_key": self.context.model_key,
            "model_class": self.context.model_class,
            "hardware": self.context.hardware,
            "stage": self.context.stage,
            "depth": self.context.depth,
            "case_id": self.context.case_id,
            "actual_route": self.context.actual_route,
            "expected_route": self.context.expected_route,
            "tensor": self.context.tensor,
            "actual_dtype": self.actual_dtype,
            "expected_dtype": self.expected_dtype,
            "actual_shape": list(self.actual_shape),
            "expected_shape": list(self.expected_shape),
            "metric_family": self.metric_family,
            "thresholds": self.thresholds.as_dict(),
            "metrics": self.metrics.as_dict(),
            "pass": self.passed,
            "command": self.context.command,
            "commit": self.context.commit,
        }
        # Build from the canonical tuple so accidental dataclass field additions cannot
        # change the persisted schema or its deterministic ordering.
        return cast(
            NumericRecordDict,
            {key: values[key] for key in NUMERIC_RECORD_KEYS},
        )


def validate_persisted_numeric_pass(
    *,
    metric_family: str,
    thresholds: Mapping[str, object],
    metrics: Mapping[str, object],
    actual_shape: Sequence[int],
    expected_shape: Sequence[int],
    context: str = "numeric record",
) -> None:
    """Recompute one claimed persisted PASS from the canonical fixed schema.

    This is the stable shared reader-side counterpart to :func:`evaluate_numeric`.
    It accepts only the fixed threshold/metric unions, validates structural and
    finite evidence, reconstructs the family gate, and raises ``ValueError`` or
    ``TypeError`` unless the persisted metrics independently satisfy that gate.
    Callers must still validate record identity, route, command and commit provenance.
    """
    if metric_family not in {"elementwise", "near_zero", "cosine_bundle"}:
        raise ValueError(f"{context}.metric_family is not canonical")
    threshold_keys: set[str] = set(NUMERIC_THRESHOLD_KEYS)
    observed_threshold_keys: set[str] = set(thresholds)
    if observed_threshold_keys != threshold_keys:
        raise ValueError(
            f"{context}.thresholds has missing="
            f"{sorted(threshold_keys - observed_threshold_keys)}, extra="
            f"{sorted(observed_threshold_keys - threshold_keys)}"
        )
    for key in NUMERIC_THRESHOLD_KEYS:
        value = thresholds[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{context}.thresholds.{key} must be finite or null")
    CANONICAL_NUMERIC_METRIC_SCHEMA.validate_payload(metrics)

    def canonical_shape(value: Sequence[int], *, owner: str) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{owner} must be a sequence of non-negative ints")
        dimensions = tuple(value)
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in dimensions
        ):
            raise ValueError(f"{owner} must contain only non-negative ints")
        return dimensions

    actual_dimensions = canonical_shape(
        actual_shape, owner=f"{context}.actual_shape"
    )
    expected_dimensions = canonical_shape(
        expected_shape, owner=f"{context}.expected_shape"
    )
    structural = (
        "shape_match",
        "nonempty_actual",
        "nonempty_expected",
        "finite_actual",
        "finite_expected",
        "finite_derived",
    )
    if any(metrics[key] is not True for key in structural):
        raise ValueError(f"{context} PASS requires all structural/finite flags true")
    if actual_dimensions != expected_dimensions:
        raise ValueError(f"{context} PASS requires matching actual/expected shapes")
    if math.prod(actual_dimensions) <= 0:
        raise ValueError(f"{context} PASS requires a positive tensor element count")

    element_thresholds = {"max_rel", "abs_floor", "atol"}
    cosine_thresholds = {
        "min_cosine",
        "max_abs",
        "norm_ratio_min",
        "norm_ratio_max",
    }
    element_metrics = {"max_rel_above", "max_abs_below", "n_above", "n_below"}
    cosine_metrics = {"max_abs", "mean_abs", "cosine", "norm_ratio"}

    if metric_family in {"elementwise", "near_zero"}:
        if any(thresholds[key] is None for key in element_thresholds) or any(
            thresholds[key] is not None for key in cosine_thresholds
        ):
            raise ValueError(
                f"{context} {metric_family} requires only "
                "max_rel/abs_floor/atol thresholds"
            )
        gate = ElementwiseGate(
            max_rel=float(cast(Real, thresholds["max_rel"])),
            abs_floor=float(cast(Real, thresholds["abs_floor"])),
            atol=float(cast(Real, thresholds["atol"])),
        )
        assert gate.max_rel is not None and gate.atol is not None
        if metric_family == "elementwise":
            if any(metrics[key] is None for key in element_metrics) or any(
                metrics[key] is not None for key in cosine_metrics
            ):
                raise ValueError(
                    f"{context} elementwise metrics have missing active or "
                    "non-null inactive fields"
                )
            max_rel = cast(Real, metrics["max_rel_above"])
            max_abs_below = cast(Real, metrics["max_abs_below"])
            if float(max_rel) < 0.0 or float(max_abs_below) < 0.0:
                raise ValueError(f"{context} elementwise error metrics must be >= 0")
            if not (
                float(max_rel) < gate.max_rel
                and float(max_abs_below) <= gate.atol
            ):
                raise ValueError(
                    f"{context} elementwise metrics exceed persisted thresholds"
                )
            n_above = cast(int, metrics["n_above"])
            n_below = cast(int, metrics["n_below"])
            if n_above + n_below != math.prod(actual_dimensions):
                raise ValueError(
                    f"{context} elementwise counts do not match tensor shape"
                )
        else:
            if (
                metrics["max_abs"] is None
                or metrics["mean_abs"] is None
                or any(
                    metrics[key] is not None
                    for key in element_metrics | {"cosine", "norm_ratio"}
                )
            ):
                raise ValueError(
                    f"{context} near_zero metrics have missing active or non-null "
                    "inactive fields"
                )
            max_abs = float(cast(Real, metrics["max_abs"]))
            mean_abs = float(cast(Real, metrics["mean_abs"]))
            if mean_abs < 0.0 or max_abs < mean_abs:
                raise ValueError(
                    f"{context} near_zero requires 0 <= mean_abs <= max_abs"
                )
            if max_abs > gate.atol:
                raise ValueError(f"{context} near_zero max_abs exceeds atol")
        return

    if any(thresholds[key] is None for key in cosine_thresholds) or any(
        thresholds[key] is not None for key in element_thresholds
    ):
        raise ValueError(
            f"{context} cosine_bundle requires only cosine-bundle thresholds"
        )
    if any(metrics[key] is None for key in cosine_metrics) or any(
        metrics[key] is not None for key in element_metrics
    ):
        raise ValueError(
            f"{context} cosine_bundle metrics have missing active or non-null "
            "inactive fields"
        )
    gate = CosineGate(
        min_cosine=float(cast(Real, thresholds["min_cosine"])),
        max_abs=float(cast(Real, thresholds["max_abs"])),
        norm_ratio_min=float(cast(Real, thresholds["norm_ratio_min"])),
        norm_ratio_max=float(cast(Real, thresholds["norm_ratio_max"])),
    )
    assert gate.min_cosine is not None
    assert gate.max_abs is not None
    assert gate.norm_ratio_min is not None
    assert gate.norm_ratio_max is not None
    cosine = float(cast(Real, metrics["cosine"]))
    max_abs = float(cast(Real, metrics["max_abs"]))
    mean_abs = float(cast(Real, metrics["mean_abs"]))
    norm_ratio = float(cast(Real, metrics["norm_ratio"]))
    if not (-1.0 <= cosine <= 1.0):
        raise ValueError(f"{context} cosine must be in [-1, 1]")
    if mean_abs < 0.0 or max_abs < mean_abs:
        raise ValueError(f"{context} requires 0 <= mean_abs <= max_abs")
    if norm_ratio < 0.0:
        raise ValueError(f"{context}.metrics.norm_ratio must be >= 0")
    if not (
        cosine >= gate.min_cosine
        and max_abs <= gate.max_abs
        and gate.norm_ratio_min <= norm_ratio <= gate.norm_ratio_max
    ):
        raise ValueError(
            f"{context} cosine_bundle metrics exceed persisted thresholds"
        )


class NumericGateError(AssertionError):
    """Numeric assertion failure with its fixed-schema JSON-safe record attached."""

    record: NumericRecord

    def __init__(self, record: NumericRecord) -> None:
        self.record = record
        super().__init__(
            "numeric gate failed: "
            + json.dumps(record.as_dict(), sort_keys=True, allow_nan=False)
        )


def append_numeric_record_jsonl(
    record: NumericRecord,
    destination: str | Path,
) -> None:
    """Append one strict JSON-safe numeric record."""
    line = json.dumps(record.as_dict(), sort_keys=True, allow_nan=False) + "\n"
    with Path(destination).expanduser().open("a", encoding="utf-8") as handle:
        handle.write(line)


def _dtype_name(tensor: Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _all_finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _finite_or_none(value: float) -> float | None:
    """Keep persisted numeric records strict-JSON-safe without hiding a failure."""
    return value if math.isfinite(value) else None


def _resolved_thresholds(
    gate: NumericThresholds,
) -> NumericThresholdReport:
    if gate.is_placeholder:
        raise ValueError(
            f"cannot evaluate with placeholder {type(gate).__name__}: {gate.reason}"
        )
    if isinstance(gate, ElementwiseGate):
        assert gate.max_rel is not None
        assert gate.abs_floor is not None
        assert gate.atol is not None
        return NumericThresholdReport(
            max_rel=gate.max_rel,
            abs_floor=gate.abs_floor,
            atol=gate.atol,
        )
    assert gate.min_cosine is not None
    assert gate.max_abs is not None
    assert gate.norm_ratio_min is not None
    assert gate.norm_ratio_max is not None
    return NumericThresholdReport(
        min_cosine=gate.min_cosine,
        max_abs=gate.max_abs,
        norm_ratio_min=gate.norm_ratio_min,
        norm_ratio_max=gate.norm_ratio_max,
    )


def _metric_family(
    context: NumericRecordContext,
    gate: NumericThresholds,
) -> MetricFamily:
    if context.stage == "S0":
        if context.depth != "module":
            raise ValueError("S0 numeric records use depth='module'")
        if not isinstance(gate, CosineGate):
            raise TypeError("S0 requires a CosineGate")
        return "cosine_bundle"
    if context.stage not in {"S1", "S2", "S3"}:
        raise ValueError(f"numeric evaluator does not support stage {context.stage!r}")
    if context.depth in {"op", "layer"}:
        if not isinstance(gate, ElementwiseGate):
            raise TypeError(f"{context.stage}/{context.depth} requires ElementwiseGate")
        if context.tensor in gate.near_zero:
            return "near_zero"
        return "elementwise"
    if context.depth == "module":
        if not isinstance(gate, CosineGate):
            raise TypeError(f"{context.stage}/module requires CosineGate")
        return "cosine_bundle"
    raise ValueError(f"unknown numeric depth {context.depth!r}")


def evaluate_numeric(
    actual: Tensor,
    expected: Tensor,
    *,
    context: NumericRecordContext,
    thresholds: NumericThresholds,
) -> NumericRecord:
    """Evaluate one tensor under the shared stage/depth family and fixed schema.

    Shape, non-emptiness, and finiteness are recorded independently for both routes
    before any numeric formula runs.  If a source precondition fails, family-specific
    metrics and ``finite_derived`` remain ``None``.  If finite sources overflow during
    metric computation, every non-finite derived float is persisted as ``None``,
    ``finite_derived`` is false, and the record fails.  Thus ``as_dict()`` remains valid
    under ``json.dumps(..., allow_nan=False)`` on both PASS and failure paths.
    """
    family = _metric_family(context, thresholds)
    threshold_report = _resolved_thresholds(thresholds)
    actual_shape = tuple(actual.shape)
    expected_shape = tuple(expected.shape)
    shape_match = actual_shape == expected_shape
    nonempty_actual = actual.numel() > 0
    nonempty_expected = expected.numel() > 0
    finite_actual = _all_finite(actual)
    finite_expected = _all_finite(expected)
    preconditions_ok = (
        shape_match
        and nonempty_actual
        and nonempty_expected
        and finite_actual
        and finite_expected
    )

    if not preconditions_ok:
        metrics = NumericMetrics(
            shape_match=shape_match,
            nonempty_actual=nonempty_actual,
            nonempty_expected=nonempty_expected,
            finite_actual=finite_actual,
            finite_expected=finite_expected,
        )
        return NumericRecord(
            context=context,
            actual_dtype=_dtype_name(actual),
            expected_dtype=_dtype_name(expected),
            actual_shape=actual_shape,
            expected_shape=expected_shape,
            metric_family=family,
            thresholds=threshold_report,
            metrics=metrics,
            passed=False,
        )

    if family == "elementwise":
        assert isinstance(thresholds, ElementwiseGate)
        assert thresholds.abs_floor is not None
        assert thresholds.max_rel is not None
        assert thresholds.atol is not None
        piecewise = max_rel_with_abs_floor(
            actual,
            expected,
            abs_floor=thresholds.abs_floor,
        )
        finite_derived = math.isfinite(piecewise["max_rel_above"]) and math.isfinite(
            piecewise["max_abs_below"]
        )
        metrics = NumericMetrics(
            shape_match=True,
            nonempty_actual=True,
            nonempty_expected=True,
            finite_actual=True,
            finite_expected=True,
            finite_derived=finite_derived,
            max_rel_above=_finite_or_none(piecewise["max_rel_above"]),
            max_abs_below=_finite_or_none(piecewise["max_abs_below"]),
            n_above=piecewise["n_above"],
            n_below=piecewise["n_below"],
        )
        passed = (
            finite_derived
            and piecewise["max_rel_above"] < thresholds.max_rel
            and piecewise["max_abs_below"] <= thresholds.atol
        )
    else:
        diff = (actual.float() - expected.float()).abs().flatten()
        family_max_abs = float(diff.max().item()) if diff.numel() else 0.0
        family_mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        if family == "near_zero":
            assert isinstance(thresholds, ElementwiseGate)
            assert thresholds.atol is not None
            finite_derived = math.isfinite(family_max_abs) and math.isfinite(
                family_mean_abs
            )
            metrics = NumericMetrics(
                shape_match=True,
                nonempty_actual=True,
                nonempty_expected=True,
                finite_actual=True,
                finite_expected=True,
                finite_derived=finite_derived,
                max_abs=_finite_or_none(family_max_abs),
                mean_abs=_finite_or_none(family_mean_abs),
            )
            passed = finite_derived and family_max_abs <= thresholds.atol
        else:
            assert isinstance(thresholds, CosineGate)
            assert thresholds.min_cosine is not None
            assert thresholds.max_abs is not None
            assert thresholds.norm_ratio_min is not None
            assert thresholds.norm_ratio_max is not None
            ratio = norm_ratio(actual, expected)
            cosine_value = cosine(actual, expected)
            finite_derived = all(
                math.isfinite(value)
                for value in (
                    family_max_abs,
                    family_mean_abs,
                    cosine_value,
                    ratio,
                )
            )
            metrics = NumericMetrics(
                shape_match=True,
                nonempty_actual=True,
                nonempty_expected=True,
                finite_actual=True,
                finite_expected=True,
                finite_derived=finite_derived,
                max_abs=_finite_or_none(family_max_abs),
                mean_abs=_finite_or_none(family_mean_abs),
                cosine=_finite_or_none(cosine_value),
                norm_ratio=_finite_or_none(ratio),
            )
            passed = (
                finite_derived
                and cosine_value >= thresholds.min_cosine
                and family_max_abs <= thresholds.max_abs
                and thresholds.norm_ratio_min <= ratio <= thresholds.norm_ratio_max
            )

    return NumericRecord(
        context=context,
        actual_dtype=_dtype_name(actual),
        expected_dtype=_dtype_name(expected),
        actual_shape=actual_shape,
        expected_shape=expected_shape,
        metric_family=family,
        thresholds=threshold_report,
        metrics=metrics,
        passed=passed,
    )


def assert_numeric(
    actual: Tensor,
    expected: Tensor,
    *,
    context: NumericRecordContext,
    thresholds: NumericThresholds,
    record_sink: Callable[[NumericRecord], None] | None = None,
) -> NumericRecord:
    """Evaluate, optionally write, and assert using the fixed numeric schema.

    ``record_sink`` runs exactly once after evaluation returns, including for a failed
    gate.  Sink exceptions propagate unchanged and take precedence over gate assertion.
    """
    record = evaluate_numeric(
        actual,
        expected,
        context=context,
        thresholds=thresholds,
    )
    if record_sink is not None:
        record_sink(record)
    if not record.passed:
        raise NumericGateError(record)
    return record


def assert_numeric_active_finite(
    actual: Tensor,
    expected: Tensor,
    *,
    context: NumericRecordContext,
    thresholds: NumericThresholds,
    record_sink: Callable[[NumericRecord], None] | None = None,
) -> NumericRecord:
    """Evaluate tensors whose matching signed infinities mark inactive entries.

    A matching ``+/-inf`` mask is structural rather than numeric evidence, so a
    non-empty finite active slice is evaluated and named ``*.active_finite``.  Every
    malformed source pair (shape/empty/NaN, a signed-mask mismatch, or no finite
    active values) is passed through unchanged.  The selected assertion path then owns
    the single fixed-schema failure record and raises :class:`NumericGateError`.
    """
    evaluated_actual = actual
    evaluated_expected = expected
    evaluated_context = context

    if (
        actual.shape == expected.shape
        and actual.numel() > 0
        and expected.numel() > 0
        and not bool(torch.isnan(actual).any().item())
        and not bool(torch.isnan(expected).any().item())
    ):
        actual_posinf = torch.isposinf(actual)
        expected_posinf = torch.isposinf(expected)
        actual_neginf = torch.isneginf(actual)
        expected_neginf = torch.isneginf(expected)
        signed_masks_match = torch.equal(
            actual_posinf,
            expected_posinf,
        ) and torch.equal(actual_neginf, expected_neginf)
        sentinel = actual_posinf | actual_neginf
        if signed_masks_match and bool(sentinel.any().item()):
            active = ~sentinel
            if bool(active.any().item()):
                active_actual = actual[active]
                active_expected = expected[active]
                if bool(torch.isfinite(active_actual).all().item()) and bool(
                    torch.isfinite(active_expected).all().item()
                ):
                    evaluated_actual = active_actual
                    evaluated_expected = active_expected
                    evaluated_context = replace(
                        context,
                        tensor=f"{context.tensor}.active_finite",
                    )

    return assert_numeric(
        evaluated_actual,
        evaluated_expected,
        context=evaluated_context,
        thresholds=thresholds,
        record_sink=record_sink,
    )


# ---------------------------------------------------------------------------
# Elementwise: diagnostics
# ---------------------------------------------------------------------------

def _rel_diff_tensor(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Same denominator as compare_tensor_stats: 2*|a-e|/(|e|+|a|+1e-6)."""
    diff = (actual - expected).float()
    denom = expected.float().abs() + actual.float().abs() + 1e-6
    return (2 * diff.abs() / denom).flatten()


def inspect_rel_worst(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str = "",
    top_k: int = 5,
) -> None:
    """Print flat indices with largest rel error and their actual/expected (check if near-zero)."""
    if actual.shape != expected.shape:
        print(f"[inspect_rel_worst] {name}: shape mismatch {actual.shape} vs {expected.shape}")
        return
    rel = _rel_diff_tensor(actual, expected)
    abs_diff = (actual - expected).float().abs().flatten()
    a_flat = actual.float().flatten()
    e_flat = expected.float().flatten()
    _, worst_idx = rel.topk(min(top_k, rel.numel()), largest=True)
    print(f"[inspect_rel_worst] {name}: top-{worst_idx.numel()} largest rel error")
    for i, idx in enumerate(worst_idx.tolist()):
        ai = a_flat[idx].item()
        ei = e_flat[idx].item()
        di = abs_diff[idx].item()
        ri = rel[idx].item()
        print(f"  [{i}] flat={idx} actual={ai:.6e} expected={ei:.6e} abs_diff={di:.6e} rel={ri:.6e} (near_zero={abs(ei)<1e-5})")
    return


def rel_error_near_zero_report(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str = "",
    near_zero_thresh: float = 1e-5,
    top_k: int = 20,
    print_worst_n: int = 10,
) -> None:
    """Report whether large max_rel is from near-zero data; print actual/expected for worst indices."""
    if actual.shape != expected.shape:
        print(f"[rel_error_near_zero] {name}: shape mismatch")
        return
    a_flat = actual.float().flatten()
    e_flat = expected.float().flatten()
    rel = _rel_diff_tensor(actual, expected)
    abs_diff = (actual - expected).float().abs().flatten()
    near_zero = (a_flat.abs() < near_zero_thresh) & (e_flat.abs() < near_zero_thresh)
    max_rel_all = float(rel.max().item())
    if (~near_zero).any():
        max_rel_excl = float(rel[~near_zero].max().item())
    else:
        max_rel_excl = float("nan")
    k_show = min(top_k, rel.numel())
    _, worst_idx = rel.topk(k_show, largest=True)
    n_near_zero_top = int(near_zero[worst_idx].sum().item())
    print(
        f"[rel_error_near_zero] {name}: max_rel_all={max_rel_all:.4e} max_rel_excl_near_zero={max_rel_excl:.4e} "
        f"(thresh={near_zero_thresh}) | among top-{k_show} worst: {n_near_zero_top} near_zero"
    )
    n_print = min(print_worst_n, k_show)
    print(f"[rel_error_near_zero] {name}: top-{n_print} worst rel (flat_idx  actual    expected  abs_diff   rel)")
    for i in range(n_print):
        idx = int(worst_idx[i].item())
        ai = a_flat[idx].item()
        ei = e_flat[idx].item()
        di = abs_diff[idx].item()
        ri = rel[idx].item()
        print(f"  [{i}] {idx:6d}  {ai:+.6e}  {ei:+.6e}  {di:.6e}  {ri:.4e}")
    return


# ---------------------------------------------------------------------------
# Elementwise: gates
# ---------------------------------------------------------------------------

def compare_tensor_stats(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    atol_pass_if_small: float | None = None,
) -> tuple[bool, dict[str, float]]:
    if actual.shape != expected.shape:
        return False, {"shape_mismatch": 1.0}
    diff = (actual - expected).float()
    abs_diff = diff.abs()
    denom = expected.float().abs() + actual.float().abs() + 1e-6
    rel_diff = 2 * abs_diff / denom
    max_abs = float(abs_diff.max().item())
    max_rel = float(rel_diff.max().item())
    ok = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
    if not ok and atol_pass_if_small is not None and max_abs <= atol_pass_if_small:
        ok = True
    return ok, {"max_abs": max_abs, "max_rel": max_rel}


def max_rel_with_abs_floor(
    actual: Tensor,
    expected: Tensor,
    *,
    abs_floor: float,
) -> PiecewiseMetrics:
    """Split the comparison at `abs_floor` and score each half on its own scale.

    Returns ``{"max_rel_above": ..., "max_abs_below": ..., "n_above": ..., "n_below": ...}``:

    * ``max_rel_above`` -- ``|diff| / |ref|`` over positions with ``|ref| > abs_floor``.
      Small denominators cannot reach this half, so the ratio stays meaningful.
    * ``max_abs_below`` -- ``|diff|`` over positions with ``|ref| <= abs_floor``, where a
      ratio would be noise.

    This piecewise form is deliberately *not* ``torch.allclose``'s additive
    ``atol + rtol*|ref|``: with the same three numbers the additive form only catches an
    overall scaling error of a few percent, while this one catches sub-percent errors.
    It also has no whole-tensor bypass (``max_abs <= atol`` passing everything), which is
    how a tensor whose values are all small used to sail through un-gated.
    """
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )
    if abs_floor < 0.0:
        raise ValueError("abs_floor must be >= 0")
    if not bool(torch.isfinite(actual).all().item()):
        raise ValueError("actual has non-finite values")
    if not bool(torch.isfinite(expected).all().item()):
        raise ValueError("expected has non-finite values")
    a = actual.float()
    e = expected.float()
    diff = (a - e).abs()
    above = e.abs() > abs_floor
    below = ~above
    return {
        "max_rel_above": float((diff[above] / e.abs()[above]).max().item()) if bool(above.any()) else 0.0,
        "max_abs_below": float(diff[below].max().item()) if bool(below.any()) else 0.0,
        "n_above": int(above.sum().item()),
        "n_below": int(below.sum().item()),
    }


def assert_max_rel(
    actual: Tensor,
    expected: Tensor,
    *,
    max_rel: float,
    abs_floor: float,
    atol: float,
    name: str = "tensor",
) -> PiecewiseMetrics:
    """Assert the piecewise elementwise gate; return the metrics either way.

    ``|ref| >  abs_floor``:  ``|diff| / |ref| < max_rel``
    ``|ref| <= abs_floor``:  ``|diff| <= atol``
    """
    assert actual.shape == expected.shape, (
        f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert torch.isfinite(actual).all(), f"{name} actual has non-finite values"
    assert torch.isfinite(expected).all(), f"{name} expected has non-finite values"
    m = max_rel_with_abs_floor(actual, expected, abs_floor=abs_floor)
    assert m["max_rel_above"] < max_rel, (
        f"{name}: max_rel={m['max_rel_above']:.6g} >= {max_rel} "
        f"(over the {m['n_above']} positions with |ref| > {abs_floor})"
    )
    assert m["max_abs_below"] <= atol, (
        f"{name}: max_abs={m['max_abs_below']:.6g} > {atol} "
        f"(over the {m['n_below']} positions with |ref| <= {abs_floor})"
    )
    return m


# ---------------------------------------------------------------------------
# Stage-level: cosine three-part gate
# ---------------------------------------------------------------------------

def cosine(a: Tensor, b: Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def max_abs(a: Tensor, b: Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def mean_abs(a: Tensor, b: Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def norm_ratio(actual: Tensor, expected: Tensor) -> float:
    """``||actual|| / ||expected||`` -- the scale that cosine cannot see.

    Cosine is invariant to a uniform rescale of either side, so a kernel that produces
    ``1.05 * expected`` scores a perfect 1.0.  This ratio is what closes that hole, and
    is why every cosine gate in this framework reports it.
    """
    den = float(expected.float().norm().item())
    if den == 0.0:
        return float("nan")
    return float(actual.float().norm().item()) / den


def check_norm_ratio(
    actual: Tensor,
    expected: Tensor,
    *,
    name: str = "tensor",
    lo: float = NORM_RATIO_MIN,
    hi: float = NORM_RATIO_MAX,
    enforce: bool = False,
) -> float:
    """Norm-ratio guard.  **Report-only by default** (`enforce=False`).

    The [lo, hi] window is a first guess, so it prints rather than fails until a full
    green run has produced a real distribution to set it from.  A suite opts into
    enforcement through its `verifier_config.py` (`CosineGate.enforce_norm_ratio`),
    never by editing this default.
    """
    r = norm_ratio(actual, expected)
    inside = lo <= r <= hi
    if not inside:
        msg = f"{name}: norm_ratio={r:.6f} outside [{lo}, {hi}]"
        if enforce:
            raise AssertionError(msg)
        print(f"[norm-ratio][report-only] {msg}")
    return r


def assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    min_cosine: float,
    max_abs_tol: float | None = None,
    name: str = "tensor",
    norm_ratio_lo: float = NORM_RATIO_MIN,
    norm_ratio_hi: float = NORM_RATIO_MAX,
    enforce_norm_ratio: bool = False,
) -> dict[str, float]:
    """Stage-level cosine gate, three parts: cosine gates, max_abs reports, norm ratio guards.

    Raises AssertionError if the cosine (or the optional `max_abs_tol`) gate fails;
    returns the metrics so the caller can record them.  The norm ratio is report-only
    unless `enforce_norm_ratio=True`.
    """
    assert actual.shape == expected.shape, (
        f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert torch.isfinite(actual).all(), f"{name} has non-finite values"
    c = cosine(actual, expected)
    m = max_abs(actual, expected)
    r = check_norm_ratio(
        actual,
        expected,
        name=name,
        lo=norm_ratio_lo,
        hi=norm_ratio_hi,
        enforce=enforce_norm_ratio,
    )
    assert c >= min_cosine, f"{name} cosine={c:.7f} < {min_cosine}"
    if max_abs_tol is not None:
        assert m <= max_abs_tol, f"{name} max_abs={m:.6g} > {max_abs_tol}"
    return {
        "cosine": c,
        "max_abs": m,
        "mean_abs": mean_abs(actual, expected),
        "norm_ratio": r,
    }


# ---------------------------------------------------------------------------
# Instruction-sequence runner
# ---------------------------------------------------------------------------

def verify_instruction_sequence(
    *,
    globs: Any,
    test_case: InstructionTestCase,
    expected_kwargs: dict[str, Any] | None = None,
) -> VerificationResult:
    expected_kwargs = expected_kwargs or {}
    run_pyvm_sequence(globs, test_case.instructions, test_case.handler_map)
    expected = test_case.expected_fn(globs, **expected_kwargs)

    diffs: list[str] = []
    stats: dict[str, dict[str, float]] = {}
    ok = True
    for name in test_case.compare_tensors:
        actual_tensor = getattr(globs, name, None)
        expected_tensor = expected.get(name)
        if not isinstance(actual_tensor, torch.Tensor) or not isinstance(
            expected_tensor, torch.Tensor
        ):
            ok = False
            diffs.append(f"{name}: missing tensor")
            continue
        pass_ok, pass_stats = compare_tensor_stats(
            name,
            actual_tensor,
            expected_tensor,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )
        stats[name] = pass_stats
        if not pass_ok:
            ok = False
            diffs.append(f"{name}: max_abs={pass_stats['max_abs']}")

    return VerificationResult(ok=ok, diffs=diffs, stats=stats)
