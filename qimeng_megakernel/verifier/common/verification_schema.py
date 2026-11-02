"""Shared depth and numeric-schema names for verifier suites."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import cast

__all__ = [
    "CANONICAL_NUMERIC_METRIC_SCHEMA",
    "NumericMetric",
    "NumericMetricSchema",
    "VerificationDepth",
]


class VerificationDepth(str, Enum):
    """Internal numeric depth for S1--S3."""

    OP = "op"
    LAYER = "layer"
    MODULE = "module"


class NumericMetric(str, Enum):
    """The fixed numeric metric union returned by every S0--S3 comparison."""

    SHAPE_MATCH = "shape_match"
    NONEMPTY_ACTUAL = "nonempty_actual"
    NONEMPTY_EXPECTED = "nonempty_expected"
    FINITE_ACTUAL = "finite_actual"
    FINITE_EXPECTED = "finite_expected"
    FINITE_DERIVED = "finite_derived"
    MAX_REL_ABOVE = "max_rel_above"
    MAX_ABS_BELOW = "max_abs_below"
    N_ABOVE = "n_above"
    N_BELOW = "n_below"
    MAX_ABS = "max_abs"
    MEAN_ABS = "mean_abs"
    COSINE = "cosine"
    NORM_RATIO = "norm_ratio"


_CANONICAL_NUMERIC_METRIC_KEYS = tuple(metric.value for metric in NumericMetric)
_BOOLEAN_METRICS = {
    NumericMetric.SHAPE_MATCH.value,
    NumericMetric.NONEMPTY_ACTUAL.value,
    NumericMetric.NONEMPTY_EXPECTED.value,
    NumericMetric.FINITE_ACTUAL.value,
    NumericMetric.FINITE_EXPECTED.value,
}
_OPTIONAL_BOOLEAN_METRICS = {NumericMetric.FINITE_DERIVED.value}
_COUNT_METRICS = {NumericMetric.N_ABOVE.value, NumericMetric.N_BELOW.value}


@dataclass(frozen=True)
class NumericMetricSchema:
    """Exact metric-key union and strict payload validator.

    A model may override threshold *values* through a named tolerance profile, but it
    cannot remove, rename, or add metric keys.  ``None`` represents a metric that is
    structurally not applicable; NaN and infinity are never valid persisted values.
    """

    keys: tuple[str, ...] = _CANONICAL_NUMERIC_METRIC_KEYS

    def __post_init__(self) -> None:
        if not isinstance(self.keys, tuple):
            raise TypeError("NumericMetricSchema.keys must be a tuple")
        if self.keys != _CANONICAL_NUMERIC_METRIC_KEYS:
            expected = set(_CANONICAL_NUMERIC_METRIC_KEYS)
            actual = set(self.keys)
            missing = tuple(key for key in _CANONICAL_NUMERIC_METRIC_KEYS if key not in actual)
            extra = tuple(key for key in self.keys if key not in expected)
            raise ValueError(
                "numeric metric schema is fixed; "
                f"missing={missing}, extra={extra}, order_matches="
                f"{self.keys == _CANONICAL_NUMERIC_METRIC_KEYS}"
            )

    def validate_payload(self, payload: Mapping[str, object]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("numeric metrics payload must be a mapping")
        expected = set(self.keys)
        actual = set(payload)
        missing = tuple(key for key in self.keys if key not in actual)
        extra = tuple(key for key in payload if key not in expected)
        if missing or extra:
            raise ValueError(
                f"numeric metrics payload has missing={missing}, extra={extra}"
            )

        for key in self.keys:
            value = payload[key]
            if key in _BOOLEAN_METRICS:
                if type(value) is not bool:
                    raise TypeError(f"numeric metric {key!r} must be bool")
                continue
            if key in _OPTIONAL_BOOLEAN_METRICS:
                if value is not None and type(value) is not bool:
                    raise TypeError(f"numeric metric {key!r} must be bool or None")
                continue
            if key in _COUNT_METRICS:
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"numeric metric {key!r} must be a non-negative int or None"
                    )
                continue
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"numeric metric {key!r} must be numeric or None")
            if not math.isfinite(cast(Real, value)):
                raise ValueError(f"numeric metric {key!r} must be finite or None")


CANONICAL_NUMERIC_METRIC_SCHEMA = NumericMetricSchema()
