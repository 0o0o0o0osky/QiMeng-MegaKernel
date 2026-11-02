"""Host tests for shared verifier metrics and profiles.

They do not execute or support CPU inference.
"""
from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from qimeng_megakernel.verifier.common.compare import (
    NUMERIC_METRIC_KEYS,
    NumericMetrics,
    NumericRecordContext,
    assert_numeric,
    assert_numeric_active_finite,
)
from qimeng_megakernel.verifier.common.verification_schema import (
    CANONICAL_NUMERIC_METRIC_SCHEMA,
    NumericMetricSchema,
)
from qimeng_megakernel.verifier.common.stage_config import (
    CosineGate,
    ElementwiseGate,
    ModelVerifierConfig,
    OracleSpec,
    StageDepthGates,
    TensorGateOverride,
    TensorMetricFamilyOverride,
    ToleranceProfile,
)


def _depth_gates() -> StageDepthGates:
    return StageDepthGates(
        op=ElementwiseGate(max_rel=0.1, abs_floor=0.05, atol=0.005),
        layer=ElementwiseGate(max_rel=0.2, abs_floor=0.05, atol=0.01),
        module=CosineGate(
            min_cosine=0.99,
            max_abs=0.5,
            norm_ratio_min=0.98,
            norm_ratio_max=1.02,
        ),
    )


def _valid_metric_payload() -> dict[str, object]:
    return {
        "shape_match": True,
        "nonempty_actual": True,
        "nonempty_expected": True,
        "finite_actual": True,
        "finite_expected": True,
        "finite_derived": True,
        "max_rel_above": 0.0,
        "max_abs_below": 0.0,
        "n_above": 1,
        "n_below": 0,
        "max_abs": None,
        "mean_abs": None,
        "cosine": None,
        "norm_ratio": None,
    }


def test_numeric_metric_schema_is_the_existing_fixed_record_union() -> None:
    schema = CANONICAL_NUMERIC_METRIC_SCHEMA
    assert schema.keys == NUMERIC_METRIC_KEYS
    assert {
        "finite_actual",
        "finite_expected",
        "finite_derived",
        "max_abs",
        "mean_abs",
        "cosine",
        "norm_ratio",
    } <= set(schema.keys)
    schema.validate_payload(_valid_metric_payload())


def test_metric_schema_rejects_missing_extra_and_nonfinite_values() -> None:
    missing = _valid_metric_payload()
    missing.pop("finite_expected")
    with pytest.raises(ValueError, match="missing=.*finite_expected"):
        CANONICAL_NUMERIC_METRIC_SCHEMA.validate_payload(missing)

    extra = _valid_metric_payload()
    extra["model_metric"] = 1.0
    with pytest.raises(ValueError, match="extra=.*model_metric"):
        CANONICAL_NUMERIC_METRIC_SCHEMA.validate_payload(extra)

    for value in (float("nan"), float("inf"), float("-inf")):
        nonfinite = _valid_metric_payload()
        nonfinite["max_abs"] = value
        with pytest.raises(ValueError, match="max_abs.*finite"):
            CANONICAL_NUMERIC_METRIC_SCHEMA.validate_payload(nonfinite)

    with pytest.raises(ValueError, match="schema is fixed.*missing"):
        NumericMetricSchema(keys=NUMERIC_METRIC_KEYS[:-1])

    with pytest.raises(ValueError, match="max_abs.*finite"):
        NumericMetrics(
            shape_match=True,
            nonempty_actual=True,
            nonempty_expected=True,
            finite_actual=True,
            finite_expected=True,
            finite_derived=True,
            max_abs=float("nan"),
        ).as_dict()


def test_named_profile_accepts_model_numbers_but_not_model_metrics() -> None:
    profile = ToleranceProfile(name="model-h100-s2", gates=_depth_gates())
    assert profile.metric_keys == NUMERIC_METRIC_KEYS
    assert profile.gate("op") == profile.gates.op
    assert profile.gate("layer") == profile.gates.layer
    assert profile.gate("module") == profile.gates.module

    config = ModelVerifierConfig(
        model_key="model",
        hardware="H100",
        model_class="LLM",
        stages=("S2",),
        disabled_stages={
            "S0": "not applicable",
            "S1": "not migrated",
            "S3": "not migrated",
            "S4": "not migrated",
        },
        oracle=OracleSpec(kind="hf"),
        gates={"S2": profile},
    )
    assert config.tolerance_profile("S2") is profile
    assert config.gate("S2") is profile.gates
    assert config.gate("S2", "op") is profile.gates.op


def test_named_profile_selects_near_zero_family_for_one_exact_tensor() -> None:
    defaults = _depth_gates()
    attention_lse_gate = ElementwiseGate(
        max_rel=0.025,
        abs_floor=0.001,
        atol=0.0005,
        reason="calibrated attention LSE threshold",
    )
    gates = StageDepthGates(
        op=defaults.op,
        layer=defaults.layer,
        module=defaults.module,
        tensor_overrides=(
            TensorGateOverride(
                depth="op",
                tensor="attention_lse",
                gate=attention_lse_gate,
            ),
        ),
    )
    profile = ToleranceProfile(
        name="model-h100-s3-near-zero",
        gates=gates,
        tensor_metric_family_overrides=(
            TensorMetricFamilyOverride(
                depth="op",
                tensor="attention_lse",
                family="near_zero",
            ),
            TensorMetricFamilyOverride(
                depth="op",
                tensor="attn_lse_intermediates.active_finite",
                family="near_zero",
            ),
        ),
    )

    resolved = profile.gate("op", tensor="attention_lse")
    assert isinstance(resolved, ElementwiseGate)
    assert resolved == replace(
        attention_lse_gate,
        near_zero=("attention_lse",),
    )
    assert resolved.reason == attention_lse_gate.reason
    depth_gate = profile.gate("op")
    assert isinstance(depth_gate, ElementwiseGate)
    assert depth_gate == replace(
        gates.op,
        near_zero=(
            "attention_lse",
            "attn_lse_intermediates.active_finite",
        ),
    )
    assert profile.gate("op", tensor="attention_lse.extra") is gates.op
    assert profile.gate("layer", tensor="attention_lse") is gates.layer

    config = ModelVerifierConfig(
        model_key="model",
        hardware="H100",
        model_class="LLM",
        stages=("S3",),
        disabled_stages={
            "S0": "not applicable",
            "S1": "not migrated",
            "S2": "not migrated",
            "S4": "not migrated",
        },
        oracle=OracleSpec(kind="hf"),
        gates={"S3": profile},
    )
    assert config.gate("S3") is gates
    config_depth_gate = config.gate("S3", "op")
    assert config_depth_gate == depth_gate
    assert isinstance(config_depth_gate, ElementwiseGate)
    assert config.gate("S3", "op", tensor="attention_lse.extra") is gates.op
    configured_gate = config.gate("S3", "op", tensor="attention_lse")
    assert configured_gate == resolved
    assert isinstance(configured_gate, ElementwiseGate)

    context = NumericRecordContext(
        model_key="model",
        model_class="LLM",
        hardware="H100",
        stage="S3",
        depth="op",
        case_id="attention-lse",
        actual_route="megakernel",
        expected_route="pyvm",
        tensor="attention_lse",
        command="pytest test_verification.py",
        commit="deadbeef",
    )
    record = assert_numeric(
        torch.zeros(4),
        torch.zeros(4),
        context=context,
        thresholds=configured_gate,
    )
    assert record.metric_family == "near_zero"
    assert record.thresholds.max_rel == attention_lse_gate.max_rel
    assert record.thresholds.abs_floor == attention_lse_gate.abs_floor
    assert record.thresholds.atol == attention_lse_gate.atol
    assert tuple(record.metrics.as_dict()) == NUMERIC_METRIC_KEYS
    CANONICAL_NUMERIC_METRIC_SCHEMA.validate_payload(record.metrics.as_dict())

    active_context = NumericRecordContext(
        model_key="model",
        model_class="LLM",
        hardware="H100",
        stage="S3",
        depth="op",
        case_id="active-attention-lse",
        actual_route="megakernel",
        expected_route="pyvm",
        tensor="attn_lse_intermediates",
        command="pytest test_verification.py",
        commit="deadbeef",
    )
    active_record = assert_numeric_active_finite(
        torch.tensor([float("-inf"), 0.0]),
        torch.tensor([float("-inf"), 0.0]),
        context=active_context,
        thresholds=config_depth_gate,
    )
    assert active_record.context.tensor == "attn_lse_intermediates.active_finite"
    assert active_record.metric_family == "near_zero"

    ordinary_context = replace(
        active_context,
        tensor="ordinary_intermediate",
    )
    ordinary_record = assert_numeric(
        torch.zeros(4),
        torch.zeros(4),
        context=ordinary_context,
        thresholds=config_depth_gate,
    )
    assert ordinary_record.metric_family == "elementwise"


@pytest.mark.parametrize(
    ("depth", "tensor", "family", "error_type", "message"),
    (
        ("module", "attention_lse", "near_zero", ValueError, "'op' or 'layer'"),
        ("stack", "attention_lse", "near_zero", ValueError, "'op' or 'layer'"),
        (1, "attention_lse", "near_zero", TypeError, "depth must be a string"),
        ("op", "", "near_zero", ValueError, "tensor must be non-empty"),
        ("op", "   ", "near_zero", ValueError, "tensor must be non-empty"),
        (
            "op",
            " attention_lse ",
            "near_zero",
            ValueError,
            "surrounding whitespace",
        ),
        ("op", 1, "near_zero", TypeError, "tensor must be a string"),
        (
            "op",
            "attention_lse",
            "elementwise",
            ValueError,
            "only supports 'near_zero'",
        ),
        ("op", "attention_lse", 1, TypeError, "family must be a string"),
    ),
)
def test_tensor_metric_family_override_rejects_invalid_fields(
    depth: object,
    tensor: object,
    family: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        TensorMetricFamilyOverride(
            depth=depth,  # type: ignore[arg-type]
            tensor=tensor,  # type: ignore[arg-type]
            family=family,  # type: ignore[arg-type]
        )


def test_named_profile_rejects_invalid_metric_family_override_collections() -> None:
    override = TensorMetricFamilyOverride(
        depth="op",
        tensor="attention_lse",
        family="near_zero",
    )
    with pytest.raises(ValueError, match="duplicate tensor metric-family override"):
        ToleranceProfile(
            name="model-h100-s2",
            gates=_depth_gates(),
            tensor_metric_family_overrides=(override, override),
        )
    with pytest.raises(ValueError, match="duplicate tensor metric-family override"):
        ToleranceProfile(
            name="model-h100-s2",
            gates=_depth_gates(),
            tensor_metric_family_overrides=(
                override,
                replace(override),
            ),
        )
    with pytest.raises(TypeError, match="must be a tuple"):
        ToleranceProfile(
            name="model-h100-s2",
            gates=_depth_gates(),
            tensor_metric_family_overrides=[override],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="TensorMetricFamilyOverride instances"):
        ToleranceProfile(
            name="model-h100-s2",
            gates=_depth_gates(),
            tensor_metric_family_overrides=("op/attention_lse",),  # type: ignore[arg-type]
        )


def test_named_profile_rejects_missing_or_illegal_thresholds() -> None:
    placeholder = StageDepthGates(
        op=ElementwiseGate(reason="not calibrated"),
        layer=ElementwiseGate(max_rel=0.2, abs_floor=0.05, atol=0.01),
        module=CosineGate(
            min_cosine=0.99,
            max_abs=0.5,
            norm_ratio_min=0.98,
            norm_ratio_max=1.02,
        ),
    )
    with pytest.raises(ValueError, match="placeholder gates"):
        ToleranceProfile(name="model-h100-s1", gates=placeholder)

    base = _depth_gates()
    near_zero_depth = StageDepthGates(
        op=ElementwiseGate(
            max_rel=0.1,
            abs_floor=0.05,
            atol=0.005,
            near_zero=("attention_lse",),
        ),
        layer=base.layer,
        module=base.module,
    )
    with pytest.raises(ValueError, match="metric family.*near_zero"):
        ToleranceProfile(name="model-h100-s1", gates=near_zero_depth)

    near_zero_override = StageDepthGates(
        op=base.op,
        layer=base.layer,
        module=base.module,
        tensor_overrides=(
            TensorGateOverride(
                depth="op",
                tensor="attention_lse",
                gate=ElementwiseGate(
                    max_rel=0.1,
                    abs_floor=0.05,
                    atol=0.005,
                    near_zero=("attention_lse",),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="override:op/attention_lse"):
        ToleranceProfile(name="model-h100-s1", gates=near_zero_override)

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            ElementwiseGate(max_rel=value, abs_floor=0.05, atol=0.01)
    with pytest.raises(TypeError, match="numeric, not bool"):
        ElementwiseGate(max_rel=cast(float, True), abs_floor=0.05, atol=0.01)
    with pytest.raises(ValueError, match="norm_ratio_max must be >= 0"):
        CosineGate(
            min_cosine=0.99,
            max_abs=0.5,
            norm_ratio_min=None,
            norm_ratio_max=-0.1,
            reason="placeholder under validation",
        )
