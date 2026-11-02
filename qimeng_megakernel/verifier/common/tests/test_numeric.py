"""Host tests for the shared S0-S3 numeric verifier core.

These tests exercise schemas and tensor math on the host; they neither implement nor
claim support for CPU inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import mock_open

import pytest
import torch

from qimeng_megakernel.verifier.common.compare import (
    NUMERIC_METRIC_KEYS,
    NUMERIC_RECORD_KEYS,
    NUMERIC_SCHEMA_VERSION,
    NUMERIC_THRESHOLD_KEYS,
    NumericGateError,
    NumericRecord,
    NumericRecordContext,
    append_numeric_record_jsonl,
    assert_numeric,
    assert_numeric_active_finite,
    evaluate_numeric,
    max_rel_with_abs_floor,
    validate_persisted_numeric_pass,
)
from qimeng_megakernel.verifier.common.stage_config import (
    CANONICAL_STAGES,
    ActionGate,
    CosineGate,
    ElementwiseGate,
    ModelVerifierConfig,
    OracleSpec,
    StageDepthGates,
    TensorGateOverride,
    GenerationGate,
)


def _context(
    *,
    stage: str = "S3",
    depth: str = "op",
    tensor: str = "hidden_states",
) -> NumericRecordContext:
    return NumericRecordContext(
        model_key="unit-model",
        model_class="LLM",
        hardware="CPU",
        stage=stage,  # type: ignore[arg-type]
        depth=depth,  # type: ignore[arg-type]
        case_id="case-0",
        actual_route="actual",
        expected_route="expected",
        tensor=tensor,
        command="pytest test_numeric.py",
        commit="deadbeef",
    )


def _elementwise(*, near_zero: tuple[str, ...] = ()) -> ElementwiseGate:
    return ElementwiseGate(
        max_rel=0.2,
        abs_floor=0.1,
        atol=0.25,
        near_zero=near_zero,
    )


def _cosine() -> CosineGate:
    return CosineGate(
        min_cosine=0.99,
        max_abs=10.0,
        norm_ratio_min=0.98,
        norm_ratio_max=1.02,
    )


def test_piecewise_formula_is_reference_relative() -> None:
    metrics = max_rel_with_abs_floor(
        torch.tensor([4.0, 0.25]),
        torch.tensor([2.0, 0.0]),
        abs_floor=0.1,
    )

    # Reference-relative: |4 - 2| / |2| = 1.0.  A symmetric denominator would
    # produce 2/3 and this regression test would fail.
    assert metrics == {
        "max_rel_above": 1.0,
        "max_abs_below": 0.25,
        "n_above": 1,
        "n_below": 1,
    }


@pytest.mark.parametrize(
    ("actual", "expected", "expected_metrics"),
    [
        (
            torch.tensor([2.0, 4.0]),
            torch.tensor([1.0, 2.0]),
            {"max_rel_above": 1.0, "max_abs_below": 0.0, "n_above": 2, "n_below": 0},
        ),
        (
            torch.tensor([0.1, -0.2]),
            torch.tensor([0.0, 0.1]),
            {"max_rel_above": 0.0, "max_abs_below": 0.3, "n_above": 0, "n_below": 2},
        ),
    ],
)
def test_piecewise_empty_partitions_are_zero(
    actual: torch.Tensor,
    expected: torch.Tensor,
    expected_metrics: dict[str, float | int],
) -> None:
    metrics = max_rel_with_abs_floor(actual, expected, abs_floor=0.5)
    assert metrics == pytest.approx(expected_metrics)


@pytest.mark.parametrize(
    ("depth", "tensor", "gate"),
    [
        ("op", "hidden_states", _elementwise()),
        ("layer", "lse", _elementwise(near_zero=("lse",))),
        ("module", "hidden_states", _cosine()),
    ],
    ids=("elementwise", "near-zero", "cosine"),
)
def test_entire_empty_tensor_fails_closed(
    depth: str,
    tensor: str,
    gate: ElementwiseGate | CosineGate,
) -> None:
    actual = torch.empty(0)
    expected = torch.empty(0)
    context = _context(depth=depth, tensor=tensor)

    record = evaluate_numeric(
        actual,
        expected,
        context=context,
        thresholds=gate,
    )

    assert not record.passed
    assert record.metrics.nonempty_actual is False
    assert record.metrics.nonempty_expected is False
    assert record.metrics.finite_actual is True
    assert record.metrics.finite_expected is True
    assert record.metrics.finite_derived is None
    json.dumps(record.as_dict(), allow_nan=False)
    with pytest.raises(AssertionError, match="numeric gate failed"):
        assert_numeric(actual, expected, context=context, thresholds=gate)


def test_elementwise_gate_uses_strict_relative_and_inclusive_absolute_bounds() -> None:
    actual = torch.tensor([3.0, 0.25])
    expected = torch.tensor([2.0, 0.0])
    at_relative_boundary = ElementwiseGate(
        max_rel=0.5,
        abs_floor=0.1,
        atol=0.25,
    )
    just_above_relative_boundary = ElementwiseGate(
        max_rel=0.5001,
        abs_floor=0.1,
        atol=0.25,
    )

    assert not evaluate_numeric(
        actual,
        expected,
        context=_context(),
        thresholds=at_relative_boundary,
    ).passed
    assert evaluate_numeric(
        actual,
        expected,
        context=_context(),
        thresholds=just_above_relative_boundary,
    ).passed


def test_near_zero_tensor_uses_only_absolute_metrics() -> None:
    gate = ElementwiseGate(
        max_rel=1e-6,
        abs_floor=1e-4,
        atol=0.1,
        near_zero=("lse",),
    )
    record = evaluate_numeric(
        torch.tensor([0.05, -0.08]),
        torch.tensor([1e-8, -1e-8]),
        context=_context(depth="layer", tensor="lse"),
        thresholds=gate,
    )

    assert record.metric_family == "near_zero"
    assert record.passed
    assert record.metrics.max_abs == pytest.approx(0.08)
    assert record.metrics.mean_abs == pytest.approx(0.065)
    assert record.metrics.max_rel_above is None
    assert record.metrics.max_abs_below is None
    assert record.metrics.n_above is None
    assert record.metrics.n_below is None


def test_cosine_scale_blind_spot_is_caught_by_norm_ratio() -> None:
    expected = torch.tensor([1.0, -2.0, 3.0])
    actual = 2.0 * expected
    record = evaluate_numeric(
        actual,
        expected,
        context=_context(depth="module"),
        thresholds=_cosine(),
    )

    assert record.metrics.cosine == pytest.approx(1.0)
    assert record.metrics.norm_ratio == pytest.approx(2.0)
    assert not record.passed
    with pytest.raises(AssertionError, match="numeric gate failed"):
        assert_numeric(
            actual,
            expected,
            context=_context(depth="module"),
            thresholds=_cosine(),
        )


def test_cosine_zero_reference_norm_fails_closed() -> None:
    record = evaluate_numeric(
        torch.zeros(3),
        torch.zeros(3),
        context=_context(depth="module"),
        thresholds=_cosine(),
    )

    assert not record.passed
    assert record.metrics.cosine == pytest.approx(0.0)
    assert record.metrics.norm_ratio is None
    assert record.metrics.finite_derived is False
    json.dumps(record.as_dict(), allow_nan=False)


def test_numeric_record_has_fixed_json_schema_and_null_metrics() -> None:
    record = evaluate_numeric(
        torch.tensor([1.1, 0.05]),
        torch.tensor([1.0, 0.0]),
        context=_context(),
        thresholds=_elementwise(),
    )
    payload = record.as_dict()

    assert tuple(payload) == NUMERIC_RECORD_KEYS
    assert tuple(payload["thresholds"]) == NUMERIC_THRESHOLD_KEYS
    assert tuple(payload["metrics"]) == NUMERIC_METRIC_KEYS
    assert payload["actual_shape"] == [2]
    assert payload["expected_shape"] == [2]
    assert payload["metrics"]["cosine"] is None
    assert payload["metrics"]["norm_ratio"] is None
    assert payload["metrics"]["max_abs"] is None
    assert payload["metrics"]["mean_abs"] is None
    assert payload["metrics"]["nonempty_actual"] is True
    assert payload["metrics"]["nonempty_expected"] is True
    assert payload["metrics"]["finite_derived"] is True
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("depth", "tensor", "gate"),
    [
        ("op", "hidden_states", _elementwise()),
        ("layer", "lse", _elementwise(near_zero=("lse",))),
        ("module", "hidden_states", _cosine()),
    ],
    ids=("elementwise", "near-zero", "cosine-bundle"),
)
def test_validate_persisted_numeric_pass_rehydrates_every_metric_family(
    depth: str,
    tensor: str,
    gate: ElementwiseGate | CosineGate,
) -> None:
    values = torch.tensor([1.0, 2.0])
    payload = evaluate_numeric(
        values,
        values.clone(),
        context=_context(depth=depth, tensor=tensor),
        thresholds=gate,
    ).as_dict()
    assert payload["pass"] is True

    validate_persisted_numeric_pass(
        metric_family=payload["metric_family"],
        thresholds=payload["thresholds"],
        metrics=payload["metrics"],
        actual_shape=payload["actual_shape"],
        expected_shape=payload["expected_shape"],
        context="record[0]",
    )


def test_validate_persisted_numeric_pass_rejects_forged_or_noncanonical_payload() -> None:
    values = torch.tensor([1.0, 2.0])
    payload = evaluate_numeric(
        values,
        values.clone(),
        context=_context(),
        thresholds=_elementwise(),
    ).as_dict()
    forged_metrics = dict(payload["metrics"])
    forged_metrics["max_rel_above"] = 0.2
    with pytest.raises(ValueError, match="metrics exceed persisted thresholds"):
        validate_persisted_numeric_pass(
            metric_family=payload["metric_family"],
            thresholds=payload["thresholds"],
            metrics=forged_metrics,
            actual_shape=payload["actual_shape"],
            expected_shape=payload["expected_shape"],
        )

    missing_threshold = dict(payload["thresholds"])
    missing_threshold.pop("atol")
    with pytest.raises(ValueError, match="thresholds has missing"):
        validate_persisted_numeric_pass(
            metric_family=payload["metric_family"],
            thresholds=missing_threshold,
            metrics=payload["metrics"],
            actual_shape=payload["actual_shape"],
            expected_shape=payload["expected_shape"],
        )

    with pytest.raises(ValueError, match="non-negative ints"):
        validate_persisted_numeric_pass(
            metric_family=payload["metric_family"],
            thresholds=payload["thresholds"],
            metrics=payload["metrics"],
            actual_shape=[True],
            expected_shape=[True],
        )


def test_assert_numeric_pass_calls_record_sink_exactly_once() -> None:
    received: list[NumericRecord] = []

    record = assert_numeric(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        context=_context(),
        thresholds=_elementwise(),
        record_sink=received.append,
    )

    assert len(received) == 1
    assert received[0] is record
    assert record.passed
    assert record.as_dict()["schema_version"] == NUMERIC_SCHEMA_VERSION == "1.2"
    json.dumps(record.as_dict(), allow_nan=False)


def test_append_numeric_record_jsonl_uses_one_strict_line_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = evaluate_numeric(
        torch.tensor([float("nan")]),
        torch.tensor([1.0]),
        context=_context(),
        thresholds=_elementwise(),
    )
    opener = mock_open()
    monkeypatch.setattr(Path, "open", opener)

    append_numeric_record_jsonl(record, Path("metrics.jsonl"))

    handle = opener()
    handle.write.assert_called_once()
    line = handle.write.call_args.args[0]
    assert isinstance(line, str)
    assert line.endswith("\n")
    assert line.count("\n") == 1
    payload = json.loads(line)
    json.dumps(payload, allow_nan=False)
    assert payload == record.as_dict()


def test_assert_numeric_failure_calls_sink_then_raises_typed_error() -> None:
    received: list[NumericRecord] = []

    with pytest.raises(NumericGateError) as error:
        assert_numeric(
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            context=_context(),
            thresholds=_elementwise(),
            record_sink=received.append,
        )

    assert len(received) == 1
    record = received[0]
    assert error.value.record is record
    assert isinstance(error.value, AssertionError)
    assert not record.passed
    payload = record.as_dict()
    assert tuple(payload) == NUMERIC_RECORD_KEYS
    assert tuple(payload["thresholds"]) == NUMERIC_THRESHOLD_KEYS
    assert tuple(payload["metrics"]) == NUMERIC_METRIC_KEYS
    assert payload["schema_version"] == NUMERIC_SCHEMA_VERSION == "1.2"
    assert payload["pass"] is False
    json.dumps(payload, allow_nan=False)
    assert str(error.value) == (
        "numeric gate failed: "
        + json.dumps(payload, sort_keys=True, allow_nan=False)
    )


@pytest.mark.parametrize(
    ("actual", "expected", "metric_state"),
    [
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0]]),
            {"shape_match": False, "finite_derived": None},
        ),
        (
            torch.tensor([float("nan")]),
            torch.tensor([1.0]),
            {"finite_actual": False, "finite_derived": None},
        ),
        (
            torch.tensor([3.4e38], dtype=torch.float32),
            torch.tensor([-3.4e38], dtype=torch.float32),
            {"finite_actual": True, "finite_derived": False},
        ),
    ],
    ids=("shape", "nan", "derived-overflow"),
)
def test_assert_numeric_precondition_and_derived_failures_reach_sink_json_safe(
    actual: torch.Tensor,
    expected: torch.Tensor,
    metric_state: dict[str, bool | None],
) -> None:
    received: list[NumericRecord] = []

    with pytest.raises(NumericGateError) as error:
        assert_numeric(
            actual,
            expected,
            context=_context(),
            thresholds=_elementwise(),
            record_sink=received.append,
        )

    assert len(received) == 1
    record = received[0]
    payload = record.as_dict()
    assert error.value.record is record
    assert tuple(payload) == NUMERIC_RECORD_KEYS
    assert tuple(payload["thresholds"]) == NUMERIC_THRESHOLD_KEYS
    assert tuple(payload["metrics"]) == NUMERIC_METRIC_KEYS
    assert payload["schema_version"] == NUMERIC_SCHEMA_VERSION == "1.2"
    assert payload["pass"] is False
    metrics: dict[str, object] = dict(payload["metrics"])
    for key, value in metric_state.items():
        assert metrics[key] is value
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "actual",
    [torch.tensor([1.0]), torch.tensor([2.0])],
    ids=("gate-pass", "gate-fail"),
)
def test_assert_numeric_record_sink_exception_propagates_unchanged(
    actual: torch.Tensor,
) -> None:
    received: list[NumericRecord] = []
    sink_error = RuntimeError("record sink failed")

    def failing_sink(record: NumericRecord) -> None:
        received.append(record)
        raise sink_error

    with pytest.raises(RuntimeError, match="record sink failed") as error:
        assert_numeric(
            actual,
            torch.tensor([1.0]),
            context=_context(),
            thresholds=_elementwise(),
            record_sink=failing_sink,
        )

    assert error.value is sink_error
    assert len(received) == 1


def test_assert_numeric_active_finite_evaluates_one_named_sentinel_slice() -> None:
    received: list[NumericRecord] = []

    record = assert_numeric_active_finite(
        torch.tensor([0.5, float("-inf"), 1.5]),
        torch.tensor([0.5, float("-inf"), 1.5]),
        context=_context(tensor="attn_lse_intermediates"),
        thresholds=_elementwise(),
        record_sink=received.append,
    )

    assert received == [record]
    assert record.passed
    assert record.context.tensor == "attn_lse_intermediates.active_finite"
    assert record.actual_shape == record.expected_shape == (2,)
    json.dumps(record.as_dict(), allow_nan=False)


def test_assert_numeric_active_finite_failed_slice_records_once() -> None:
    received: list[NumericRecord] = []

    with pytest.raises(NumericGateError) as error:
        assert_numeric_active_finite(
            torch.tensor([100.0, float("-inf")]),
            torch.tensor([1.0, float("-inf")]),
            context=_context(tensor="attn_lse_intermediates"),
            thresholds=_elementwise(),
            record_sink=received.append,
        )

    assert received == [error.value.record]
    assert error.value.record.context.tensor == (
        "attn_lse_intermediates.active_finite"
    )
    assert not error.value.record.passed
    json.dumps(error.value.record.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("actual", "expected", "metric_state"),
    [
        (
            torch.ones(2),
            torch.ones(3),
            {"shape_match": False},
        ),
        (
            torch.empty(0),
            torch.empty(0),
            {"nonempty_actual": False, "nonempty_expected": False},
        ),
        (
            torch.tensor([float("nan")]),
            torch.tensor([0.0]),
            {"finite_actual": False, "finite_expected": True},
        ),
        (
            torch.tensor([float("inf")]),
            torch.tensor([0.0]),
            {"finite_actual": False, "finite_expected": True},
        ),
        (
            torch.tensor([float("inf")]),
            torch.tensor([float("-inf")]),
            {"finite_actual": False, "finite_expected": False},
        ),
        (
            torch.tensor([float("-inf")]),
            torch.tensor([float("-inf")]),
            {"finite_actual": False, "finite_expected": False},
        ),
    ],
    ids=(
        "shape",
        "empty",
        "nan",
        "unpaired-infinity",
        "signed-mask-mismatch",
        "no-active",
    ),
)
def test_assert_numeric_active_finite_structural_failures_record_once(
    actual: torch.Tensor,
    expected: torch.Tensor,
    metric_state: dict[str, bool],
) -> None:
    received: list[NumericRecord] = []

    with pytest.raises(NumericGateError) as error:
        assert_numeric_active_finite(
            actual,
            expected,
            context=_context(tensor="attn_lse_intermediates"),
            thresholds=_elementwise(),
            record_sink=received.append,
        )

    assert received == [error.value.record]
    record = error.value.record
    assert record.context.tensor == "attn_lse_intermediates"
    assert not record.passed
    for field, value in metric_state.items():
        assert getattr(record.metrics, field) is value
    json.dumps(record.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("depth", "gate", "overflow_metric"),
    [
        ("op", _elementwise(), "max_rel_above"),
        ("module", _cosine(), "max_abs"),
    ],
    ids=("elementwise", "cosine"),
)
def test_derived_overflow_is_json_safe_and_asserts(
    depth: str,
    gate: ElementwiseGate | CosineGate,
    overflow_metric: str,
) -> None:
    actual = torch.tensor([3.4e38], dtype=torch.float32)
    expected = torch.tensor([-3.4e38], dtype=torch.float32)
    context = _context(depth=depth)

    record = evaluate_numeric(
        actual,
        expected,
        context=context,
        thresholds=gate,
    )
    payload = record.as_dict()

    assert not record.passed
    assert record.metrics.finite_actual is True
    assert record.metrics.finite_expected is True
    assert record.metrics.finite_derived is False
    assert payload["metrics"][overflow_metric] is None
    json.dumps(payload, allow_nan=False)
    with pytest.raises(AssertionError, match="numeric gate failed"):
        assert_numeric(actual, expected, context=context, thresholds=gate)


def test_shape_and_both_finiteness_fields_are_explicit() -> None:
    shape_record = evaluate_numeric(
        torch.tensor([1.0, 2.0]),
        torch.tensor([[1.0, 2.0]]),
        context=_context(),
        thresholds=_elementwise(),
    )
    assert not shape_record.passed
    assert shape_record.metrics.shape_match is False
    assert shape_record.metrics.nonempty_actual is True
    assert shape_record.metrics.nonempty_expected is True
    assert shape_record.metrics.finite_actual is True
    assert shape_record.metrics.finite_expected is True
    assert shape_record.metrics.finite_derived is None
    assert shape_record.actual_shape == (2,)
    assert shape_record.expected_shape == (1, 2)
    assert shape_record.metrics.max_rel_above is None

    finite_record = evaluate_numeric(
        torch.tensor([float("inf")]),
        torch.tensor([float("nan")]),
        context=_context(),
        thresholds=_elementwise(),
    )
    assert not finite_record.passed
    assert finite_record.metrics.shape_match is True
    assert finite_record.metrics.finite_actual is False
    assert finite_record.metrics.finite_expected is False
    assert finite_record.metrics.finite_derived is None
    assert finite_record.metrics.max_rel_above is None


def test_stage_depth_matrix_and_s4_lookup_backcompat() -> None:
    depth_gates = StageDepthGates(
        op=_elementwise(),
        layer=_elementwise(near_zero=("lse",)),
        module=_cosine(),
    )
    s4_gate = GenerationGate(top1_min=0.95)
    config = ModelVerifierConfig(
        model_key="unit-model",
        hardware="CPU",
        model_class="LLM",
        stages=("S1", "S4"),
        disabled_stages={
            "S0": "no independent oracle",
            "S2": "not in this unit fixture",
            "S3": "not in this unit fixture",
        },
        oracle=OracleSpec(kind="hf"),
        gates={"S1": depth_gates, "S4": s4_gate},
    )

    assert config.gate("S1", "op") is depth_gates.op
    assert config.gate("S1", "layer") is depth_gates.layer
    assert config.gate("S1", "module") is depth_gates.module
    assert config.gate("S4") is s4_gate
    with pytest.raises(ValueError, match="depth is only valid"):
        config.gate("S4", "op")


def test_tensor_gate_override_keeps_fixed_metrics_schema_and_exact_fallback() -> None:
    default_module = _cosine()
    hidden_gate = CosineGate(
        min_cosine=0.9995,
        max_abs=5.0,
        norm_ratio_min=0.99,
        norm_ratio_max=1.025,
    )
    logits_gate = CosineGate(
        min_cosine=0.999,
        max_abs=0.75,
        norm_ratio_min=0.98,
        norm_ratio_max=1.005,
    )
    depth_gates = StageDepthGates(
        op=_elementwise(),
        layer=_elementwise(),
        module=default_module,
        tensor_overrides=(
            TensorGateOverride(
                depth="module",
                tensor="hidden_states",
                gate=hidden_gate,
            ),
            TensorGateOverride(depth="module", tensor="logits", gate=logits_gate),
        ),
    )
    config = _stage_config(
        model_class="LLM",
        stages=("S1",),
        gates={"S1": depth_gates},
    )

    assert config.gate("S1", "module") is default_module
    assert config.gate("S1", "module", tensor="hidden_states") is hidden_gate
    assert config.gate("S1", "module", tensor="logits") is logits_gate
    assert config.gate("S1", "module", tensor="unconfigured") is default_module
    assert config.gate("S1", "layer", tensor="hidden_states") is depth_gates.layer

    records = []
    for tensor in ("hidden_states", "logits"):
        gate = config.gate("S1", "module", tensor=tensor)
        assert isinstance(gate, CosineGate)
        records.append(
            evaluate_numeric(
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0]),
                context=_context(stage="S1", depth="module", tensor=tensor),
                thresholds=gate,
            )
        )

    hidden_record, logits_record = records
    assert hidden_record.metric_family == logits_record.metric_family == "cosine_bundle"
    assert tuple(hidden_record.as_dict()) == tuple(logits_record.as_dict()) == NUMERIC_RECORD_KEYS
    assert (
        tuple(hidden_record.as_dict()["thresholds"])
        == tuple(logits_record.as_dict()["thresholds"])
        == NUMERIC_THRESHOLD_KEYS
    )
    assert (
        tuple(hidden_record.as_dict()["metrics"])
        == tuple(logits_record.as_dict()["metrics"])
        == NUMERIC_METRIC_KEYS
    )
    assert hidden_record.thresholds != logits_record.thresholds


@pytest.mark.parametrize(
    ("depth", "tensor", "gate", "error_type", "message"),
    (
        ("stack", "hidden_states", _cosine(), ValueError, "override depth"),
        ("module", "", _cosine(), ValueError, "non-empty"),
        ("module", "   ", _cosine(), ValueError, "non-empty"),
        (
            "module",
            " logits ",
            _cosine(),
            ValueError,
            "surrounding whitespace",
        ),
        ("op", "hidden_states", _cosine(), TypeError, "must use ElementwiseGate"),
        ("layer", "hidden_states", _cosine(), TypeError, "must use ElementwiseGate"),
        ("module", "logits", _elementwise(), TypeError, "must use CosineGate"),
    ),
)
def test_tensor_gate_override_rejects_invalid_keys_and_gate_families(
    depth: str,
    tensor: str,
    gate: ElementwiseGate | CosineGate,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        TensorGateOverride(
            depth=depth,  # type: ignore[arg-type]
            tensor=tensor,
            gate=gate,
        )


def test_tensor_gate_override_rejects_duplicates_and_mutable_container() -> None:
    override = TensorGateOverride(
        depth="module",
        tensor="hidden_states",
        gate=_cosine(),
    )
    defaults = {
        "op": _elementwise(),
        "layer": _elementwise(),
        "module": _cosine(),
    }

    with pytest.raises(ValueError, match="duplicate tensor gate override"):
        StageDepthGates(
            **defaults,  # type: ignore[arg-type]
            tensor_overrides=(override, override),
        )
    with pytest.raises(TypeError, match="must be a tuple"):
        StageDepthGates(
            **defaults,  # type: ignore[arg-type]
            tensor_overrides=[override],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="TensorGateOverride instances"):
        StageDepthGates(
            **defaults,  # type: ignore[arg-type]
            tensor_overrides=("module/hidden_states",),  # type: ignore[arg-type]
        )


def test_same_tensor_name_can_override_two_explicit_depths() -> None:
    op_override = _elementwise(near_zero=("shared",))
    module_override = _cosine()
    depth_gates = StageDepthGates(
        op=_elementwise(),
        layer=_elementwise(),
        module=CosineGate(
            min_cosine=0.9,
            max_abs=20.0,
            norm_ratio_min=0.8,
            norm_ratio_max=1.2,
        ),
        tensor_overrides=(
            TensorGateOverride(depth="op", tensor="shared", gate=op_override),
            TensorGateOverride(
                depth="module",
                tensor="shared",
                gate=module_override,
            ),
        ),
    )

    assert depth_gates.gate("op", tensor="shared") is op_override
    assert depth_gates.gate("module", tensor="shared") is module_override
    assert depth_gates.gate("layer", tensor="shared") is depth_gates.layer


def test_tensor_gate_lookup_rejects_invalid_names_and_legacy_flat_gate() -> None:
    depth_gates = StageDepthGates(
        op=_elementwise(),
        layer=_elementwise(),
        module=_cosine(),
    )
    config = _stage_config(
        model_class="LLM",
        stages=("S1",),
        gates={"S1": depth_gates},
    )
    legacy = _stage_config(
        model_class="LLM",
        stages=(),
        gates={"S1": _elementwise()},
    )

    with pytest.raises(ValueError, match="requires a numeric depth"):
        config.gate("S1", tensor="hidden_states")  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="string tensor name"):
        config.gate("S1", "module", tensor=1)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="non-empty tensor name"):
        config.gate("S1", "module", tensor="")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        config.gate("S1", "module", tensor=" logits ")
    with pytest.raises(ValueError, match="legacy flat gate"):
        legacy.gate("S1", "op", tensor="hidden_states")


def test_placeholder_tensor_override_resolves_exactly_and_fails_evaluation_closed() -> None:
    with pytest.raises(ValueError, match="unset thresholds"):
        CosineGate()
    placeholder = CosineGate(reason="logits calibration is intentionally incomplete")
    depth_gates = StageDepthGates(
        op=_elementwise(),
        layer=_elementwise(),
        module=_cosine(),
        tensor_overrides=(
            TensorGateOverride(depth="module", tensor="logits", gate=placeholder),
        ),
    )
    config = _stage_config(
        model_class="LLM",
        stages=("S1",),
        gates={"S1": depth_gates},
    )

    resolved = config.gate("S1", "module", tensor="logits")
    assert resolved is placeholder
    assert isinstance(resolved, CosineGate) and resolved.is_placeholder
    with pytest.raises(ValueError, match="cannot evaluate with placeholder"):
        evaluate_numeric(
            torch.ones(2),
            torch.ones(2),
            context=_context(stage="S1", depth="module", tensor="logits"),
            thresholds=resolved,
        )


def test_stage_depth_configuration_validation() -> None:
    with pytest.raises(TypeError, match="module"):
        StageDepthGates(  # type: ignore[call-arg]
            op=_elementwise(),
            layer=_elementwise(),
        )

    with pytest.raises(TypeError, match="module must be a CosineGate"):
        StageDepthGates(
            op=_elementwise(),
            layer=_elementwise(),
            module=_elementwise(),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="norm_ratio_min"):
        CosineGate(
            min_cosine=0.99,
            max_abs=1.0,
            norm_ratio_min=1.1,
            norm_ratio_max=1.0,
        )

    with pytest.raises(ValueError, match="no gate configuration"):
        ModelVerifierConfig(
            model_key="unit-model",
            hardware="CPU",
            model_class="LLM",
            stages=("S1",),
            disabled_stages={
                "S0": "not used",
                "S2": "not used",
                "S3": "not used",
                "S4": "not used",
            },
            oracle=OracleSpec(kind="hf"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_rel", float("nan")),
        ("max_rel", float("inf")),
        ("abs_floor", float("nan")),
        ("abs_floor", float("inf")),
        ("atol", float("nan")),
        ("atol", float("inf")),
    ],
)
def test_elementwise_gate_rejects_nonfinite_thresholds(
    field: str,
    value: float,
) -> None:
    values = {"max_rel": 0.1, "abs_floor": 0.1, "atol": 0.1}
    values[field] = value
    with pytest.raises(ValueError, match="finite"):
        ElementwiseGate(
            max_rel=values["max_rel"],
            abs_floor=values["abs_floor"],
            atol=values["atol"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_cosine", float("nan")),
        ("min_cosine", float("inf")),
        ("max_abs", float("nan")),
        ("max_abs", float("inf")),
        ("norm_ratio_min", float("nan")),
        ("norm_ratio_min", float("inf")),
        ("norm_ratio_max", float("nan")),
        ("norm_ratio_max", float("inf")),
    ],
)
def test_cosine_gate_rejects_nonfinite_thresholds(
    field: str,
    value: float,
) -> None:
    values = {
        "min_cosine": 0.9,
        "max_abs": 1.0,
        "norm_ratio_min": 0.9,
        "norm_ratio_max": 1.1,
    }
    values[field] = value
    with pytest.raises(ValueError, match="finite"):
        CosineGate(
            min_cosine=values["min_cosine"],
            max_abs=values["max_abs"],
            norm_ratio_min=values["norm_ratio_min"],
            norm_ratio_max=values["norm_ratio_max"],
        )


def _stage_config(
    *,
    model_class: str,
    stages: tuple[str, ...],
    gates: dict[str, object],
) -> ModelVerifierConfig:
    return ModelVerifierConfig(
        model_key="unit-model",
        hardware="CPU",
        model_class=model_class,
        stages=stages,
        disabled_stages={
            stage: "not used in this fixture"
            for stage in CANONICAL_STAGES
            if stage not in stages
        },
        oracle=OracleSpec(kind="openpi" if model_class == "VLA" else "hf"),
        gates=gates,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("stage", "model_class"),
    [("S0", "VLA"), ("S4", "LLM")],
)
def test_enabled_non_matrix_stage_requires_gate(
    stage: str,
    model_class: str,
) -> None:
    with pytest.raises(ValueError, match="no gate configuration"):
        _stage_config(model_class=model_class, stages=(stage,), gates={})


@pytest.mark.parametrize(
    ("stage", "model_class", "gate"),
    [
        ("S0", "VLA", _elementwise()),
        (
            "S1",
            "LLM",
            GenerationGate(top1_min=0.95),
        ),
        (
            "S4",
            "LLM",
            ActionGate(
                action_cosine=0.99,
                action_max_abs=0.1,
                norm_ratio_min=0.98,
                norm_ratio_max=1.02,
                value_range=(-1.0, 1.0),
            ),
        ),
        (
            "S4",
            "VLA",
            GenerationGate(top1_min=0.95),
        ),
    ],
)
def test_stage_rejects_wrong_gate_type(
    stage: str,
    model_class: str,
    gate: object,
) -> None:
    with pytest.raises(TypeError, match=f"gate {stage!r} must be"):
        _stage_config(model_class=model_class, stages=(stage,), gates={stage: gate})


def test_disabled_stage_gate_still_has_type_checked() -> None:
    with pytest.raises(TypeError, match="gate 'S4' must be GenerationGate"):
        _stage_config(model_class="LLM", stages=(), gates={"S4": _elementwise()})


def test_vla_s4_accepts_action_gate() -> None:
    action_gate = ActionGate(
        action_cosine=0.99,
        action_max_abs=0.1,
        norm_ratio_min=0.98,
        norm_ratio_max=1.02,
        value_range=(-1.0, 1.0),
    )
    config = _stage_config(
        model_class="VLA",
        stages=("S4",),
        gates={"S4": action_gate},
    )

    assert config.gate("S4") is action_gate
