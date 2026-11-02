"""`ModelVerifierConfig` -- what each suite declares, so the skeleton stays factual-free.

One `verifier_config.py` per `<model>/<hardware>/`.  Every threshold a test asserts is
either here or in `common.tolerance`; a bare literal in a `test_*.py` is a defect,
because it is a number nobody can trace to a decision.

Canonical stages (same names for every model class):

====  ==========================  ===================================================
S0    official_to_baseline        oracle -> in-repo baseline.  Only needed where an
                                  independent official implementation exists (VLA:
                                  openpi).  For LLM the baseline *is* HF, so S0
                                  degenerates to a weight-loading self-check.
S1    baseline_to_fused
S2    fused_to_pyvm
S3    pyvm_to_megakernel
S4    e2e                         token generation or action inference.
====  ==========================  ===================================================

A stage the suite does not run must still be declared, with a reason: `stages` lists
what is enabled and `disabled_stages` maps the rest to why.  "Not in the tuple" is not
a statement; "S4: waiting on action-head opcodes 11-14" is.

Gate values may be `None` where a stage is declared-but-not-yet-calibrated, but only
together with a `reason` -- `__post_init__` enforces that pairing, so a placeholder
cannot quietly become an un-gated stage.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, overload

from .verification_schema import (
    CANONICAL_NUMERIC_METRIC_SCHEMA,
    NumericMetricSchema,
    VerificationDepth,
)

CANONICAL_STAGES = ("S0", "S1", "S2", "S3", "S4")
NUMERIC_STAGES = ("S1", "S2", "S3")
NUMERIC_DEPTHS = tuple(depth.value for depth in VerificationDepth)
NumericDepth = Literal["op", "layer", "module"]
ElementwiseDepth = Literal["op", "layer"]
TensorMetricFamily = Literal["near_zero"]

STAGE_DIRS = {
    "S0": "official_to_baseline",
    "S1": "baseline_to_fused",
    "S2": "fused_to_pyvm",
    "S3": "pyvm_to_megakernel",
    "S4": "e2e_generate",
}

_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")


def _validate_optional_threshold(owner: str, name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{owner}.{name} must be numeric, not {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{owner}.{name} must be finite")


@dataclass(frozen=True)
class OracleSpec:
    """Where the ground truth comes from.

    `kind="hf"`     -- transformers reference; `env` names the local-weights variable.
    `kind="openpi"` -- an external JAX/flax implementation; `env` names what must be set
                       for it to be importable, and `cache` the local self-check file.
    """

    kind: str
    env: tuple[str, ...] = ()
    cache: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class Gate:
    """Base: every gate may be a placeholder, but never a silent one."""

    reason: str = ""

    def __post_init__(self) -> None:
        if self.is_placeholder and not self.reason:
            raise ValueError(
                f"{type(self).__name__} has unset thresholds and no `reason`. "
                "A stage may be declared before it is calibrated, but the config has to "
                "say why, or the stage reads as gated when it is not."
            )

    @property
    def is_placeholder(self) -> bool:
        return False


@dataclass(frozen=True)
class ElementwiseGate(Gate):
    """Op-depth criterion: `max_rel` above `abs_floor`, `max_abs` below it.

    `near_zero` names tensors scored on `max_abs` alone (attention partials, LSE):
    a ratio against a near-zero reference is noise, not a measurement.
    """

    max_rel: float | None = None
    abs_floor: float | None = None
    atol: float | None = None
    near_zero: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("max_rel", self.max_rel),
            ("abs_floor", self.abs_floor),
            ("atol", self.atol),
        ):
            _validate_optional_threshold("ElementwiseGate", name, value)
        if self.max_rel is not None and self.max_rel <= 0.0:
            raise ValueError("ElementwiseGate.max_rel must be > 0")
        if self.abs_floor is not None and self.abs_floor < 0.0:
            raise ValueError("ElementwiseGate.abs_floor must be >= 0")
        if self.atol is not None and self.atol < 0.0:
            raise ValueError("ElementwiseGate.atol must be >= 0")
        if len(set(self.near_zero)) != len(self.near_zero):
            raise ValueError("ElementwiseGate.near_zero must not contain duplicates")

    @property
    def is_placeholder(self) -> bool:
        return self.max_rel is None or self.abs_floor is None or self.atol is None


@dataclass(frozen=True)
class CosineGate(Gate):
    """Module/S0 cosine bundle thresholds.

    Cosine is legal only for S1/S2/S3 ``module`` depth and for S0.  The norm-ratio
    window is mandatory because cosine alone cannot detect a uniform rescale.  The
    evaluator always reports cosine, max/mean absolute error, and norm ratio together.
    """

    min_cosine: float | None = None
    max_abs: float | None = None
    norm_ratio_min: float | None = None
    norm_ratio_max: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("min_cosine", self.min_cosine),
            ("max_abs", self.max_abs),
            ("norm_ratio_min", self.norm_ratio_min),
            ("norm_ratio_max", self.norm_ratio_max),
        ):
            _validate_optional_threshold("CosineGate", name, value)
        if self.min_cosine is not None and not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError("CosineGate.min_cosine must be in [-1, 1]")
        if self.max_abs is not None and self.max_abs < 0.0:
            raise ValueError("CosineGate.max_abs must be >= 0")
        if self.norm_ratio_min is not None and self.norm_ratio_min < 0.0:
            raise ValueError("CosineGate.norm_ratio_min must be >= 0")
        if self.norm_ratio_max is not None and self.norm_ratio_max < 0.0:
            raise ValueError("CosineGate.norm_ratio_max must be >= 0")
        if (
            self.norm_ratio_min is not None
            and self.norm_ratio_max is not None
            and self.norm_ratio_min > self.norm_ratio_max
        ):
            raise ValueError(
                "CosineGate.norm_ratio_min must be <= norm_ratio_max"
            )

    @property
    def is_placeholder(self) -> bool:
        return any(
            value is None
            for value in (
                self.min_cosine,
                self.max_abs,
                self.norm_ratio_min,
                self.norm_ratio_max,
            )
        )

@dataclass(frozen=True)
class TensorGateOverride:
    """One exact tensor threshold override within a canonical numeric depth.

    The gate class remains depth-owned: op/layer tensors can only select another
    :class:`ElementwiseGate`, while module tensors can only select another
    :class:`CosineGate`.  A tuple of these records, rather than a mapping, lets the
    config reject duplicate ``(depth, tensor)`` declarations instead of silently
    keeping whichever value a mapping construction happened to retain.
    """

    depth: NumericDepth
    tensor: str
    gate: ElementwiseGate | CosineGate

    def __post_init__(self) -> None:
        if self.depth not in NUMERIC_DEPTHS:
            raise ValueError(
                f"unknown tensor override depth {self.depth!r}; "
                f"choose one of {NUMERIC_DEPTHS}"
            )
        if not isinstance(self.tensor, str):
            raise TypeError("TensorGateOverride.tensor must be a string")
        if not self.tensor.strip():
            raise ValueError("TensorGateOverride.tensor must be non-empty")
        if self.tensor != self.tensor.strip():
            raise ValueError(
                "TensorGateOverride.tensor must not have surrounding whitespace"
            )

        expected_type = CosineGate if self.depth == "module" else ElementwiseGate
        if not isinstance(self.gate, expected_type):
            raise TypeError(
                f"TensorGateOverride {self.depth}/{self.tensor!r} must use "
                f"{expected_type.__name__}, got {type(self.gate).__name__}"
            )
        if self.gate.is_placeholder and not self.gate.reason:
            # Gate.__post_init__ already enforces this for the built-in gate classes.
            # Keep the invariant explicit here so future gate subclasses cannot turn
            # an exact tensor match into a silently ungated comparison.
            raise ValueError(
                f"placeholder tensor override {self.depth}/{self.tensor!r} "
                "requires a reason"
            )


@dataclass(frozen=True)
class TensorMetricFamilyOverride:
    """Select one non-default metric family for one exact tensor.

    Numeric thresholds remain owned by :class:`StageDepthGates` and
    :class:`TensorGateOverride`.  Keeping this declaration separate prevents a model
    profile from hiding metric-family selection inside a threshold gate.  Near-zero
    is currently the only non-default family and is meaningful only at op/layer
    depth; module depth remains the fixed cosine bundle.
    """

    depth: ElementwiseDepth
    tensor: str
    family: TensorMetricFamily

    def __post_init__(self) -> None:
        if not isinstance(self.depth, str):
            raise TypeError("TensorMetricFamilyOverride.depth must be a string")
        if self.depth not in ("op", "layer"):
            raise ValueError(
                "TensorMetricFamilyOverride.depth must be 'op' or 'layer', "
                f"not {self.depth!r}"
            )
        if not isinstance(self.tensor, str):
            raise TypeError("TensorMetricFamilyOverride.tensor must be a string")
        if not self.tensor.strip():
            raise ValueError(
                "TensorMetricFamilyOverride.tensor must be non-empty"
            )
        if self.tensor != self.tensor.strip():
            raise ValueError(
                "TensorMetricFamilyOverride.tensor must not have surrounding "
                "whitespace"
            )
        if not isinstance(self.family, str):
            raise TypeError("TensorMetricFamilyOverride.family must be a string")
        if self.family != "near_zero":
            raise ValueError(
                "TensorMetricFamilyOverride.family currently only supports "
                f"'near_zero', not {self.family!r}"
            )


@dataclass(frozen=True)
class StageDepthGates:
    """The mandatory S1/S2/S3 depth matrix.

    ``op`` and ``layer`` use the one reference-relative piecewise family.  ``module``
    uses the cosine bundle.  Keeping the three fields explicit makes an omitted depth
    a constructor error instead of a silently un-gated suite.
    """

    op: ElementwiseGate
    layer: ElementwiseGate
    module: CosineGate
    tensor_overrides: tuple[TensorGateOverride, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.op, ElementwiseGate):
            raise TypeError("StageDepthGates.op must be an ElementwiseGate")
        if not isinstance(self.layer, ElementwiseGate):
            raise TypeError("StageDepthGates.layer must be an ElementwiseGate")
        if not isinstance(self.module, CosineGate):
            raise TypeError("StageDepthGates.module must be a CosineGate")

        if not isinstance(self.tensor_overrides, tuple):
            raise TypeError("StageDepthGates.tensor_overrides must be a tuple")
        seen: set[tuple[NumericDepth, str]] = set()
        for override in self.tensor_overrides:
            if not isinstance(override, TensorGateOverride):
                raise TypeError(
                    "StageDepthGates.tensor_overrides entries must be "
                    "TensorGateOverride instances"
                )
            key = (override.depth, override.tensor)
            if key in seen:
                raise ValueError(
                    f"duplicate tensor gate override for "
                    f"{override.depth}/{override.tensor!r}"
                )
            seen.add(key)

    def gate(self, depth: NumericDepth, tensor: str | None = None) -> Gate:
        if depth not in NUMERIC_DEPTHS:
            raise ValueError(
                f"unknown numeric depth {depth!r}; choose one of {NUMERIC_DEPTHS}"
            )
        if tensor is not None:
            if not isinstance(tensor, str):
                raise TypeError("tensor gate lookup requires a string tensor name")
            if not tensor.strip():
                raise ValueError("tensor gate lookup requires a non-empty tensor name")
            if tensor != tensor.strip():
                raise ValueError(
                    "tensor gate lookup name must not have surrounding whitespace"
                )
            for override in self.tensor_overrides:
                if override.depth == depth and override.tensor == tensor:
                    return override.gate
        return getattr(self, depth)


@dataclass(frozen=True)
class ToleranceProfile:
    """A named, calibrated S1--S3 tolerance bundle.

    Suites own the numeric threshold values and their calibration reasons.  The
    framework owns the metric schema, so a profile can override numbers but cannot
    drop or rename metrics.  Placeholder gates are useful migration declarations but
    are deliberately not valid named profiles.
    """

    name: str
    gates: StageDepthGates
    metric_schema: NumericMetricSchema = CANONICAL_NUMERIC_METRIC_SCHEMA
    tensor_metric_family_overrides: tuple[TensorMetricFamilyOverride, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("ToleranceProfile.name must be a string")
        if not _PROFILE_NAME.fullmatch(self.name):
            raise ValueError(
                "ToleranceProfile.name must match [a-z0-9][a-z0-9_.-]*"
            )
        if not isinstance(self.gates, StageDepthGates):
            raise TypeError("ToleranceProfile.gates must be StageDepthGates")
        if not isinstance(self.metric_schema, NumericMetricSchema):
            raise TypeError(
                "ToleranceProfile.metric_schema must be NumericMetricSchema"
            )
        if self.metric_schema != CANONICAL_NUMERIC_METRIC_SCHEMA:
            raise ValueError("ToleranceProfile must use the canonical metric schema")
        if not isinstance(self.tensor_metric_family_overrides, tuple):
            raise TypeError(
                "ToleranceProfile.tensor_metric_family_overrides must be a tuple"
            )
        seen_family_overrides: set[tuple[ElementwiseDepth, str]] = set()
        for override in self.tensor_metric_family_overrides:
            if not isinstance(override, TensorMetricFamilyOverride):
                raise TypeError(
                    "ToleranceProfile.tensor_metric_family_overrides entries must be "
                    "TensorMetricFamilyOverride instances"
                )
            key = (override.depth, override.tensor)
            if key in seen_family_overrides:
                raise ValueError(
                    "duplicate tensor metric-family override for "
                    f"{override.depth}/{override.tensor!r}"
                )
            seen_family_overrides.add(key)

        configured = [self.gates.op, self.gates.layer, self.gates.module]
        configured.extend(override.gate for override in self.gates.tensor_overrides)
        placeholders = [type(gate).__name__ for gate in configured if gate.is_placeholder]
        if placeholders:
            raise ValueError(
                f"ToleranceProfile {self.name!r} has placeholder gates: {placeholders}"
            )
        near_zero_candidates: list[tuple[str, Gate]] = [
            ("op", self.gates.op),
            ("layer", self.gates.layer),
        ]
        near_zero_candidates.extend(
            (f"override:{override.depth}/{override.tensor}", override.gate)
            for override in self.gates.tensor_overrides
        )
        near_zero_owners = [
            owner
            for owner, gate in near_zero_candidates
            if isinstance(gate, ElementwiseGate) and gate.near_zero
        ]
        if near_zero_owners:
            raise ValueError(
                "ToleranceProfile metric family declarations must be explicit; "
                f"ElementwiseGate.near_zero is not allowed in {near_zero_owners}. "
                "Use TensorMetricFamilyOverride instead."
            )

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return self.metric_schema.keys

    def gate(self, depth: NumericDepth, tensor: str | None = None) -> Gate:
        """Resolve thresholds and attach only declared exact-name family selectors.

        A depth-only lookup carries every selector at that depth because comparison
        helpers may refine the record name after gate lookup (for example by adding
        ``.active_finite``).  The evaluator still chooses near-zero only when its
        final context tensor exactly matches one of those names.
        """
        selected = self.gates.gate(depth, tensor=tensor)
        family_tensors = tuple(
            override.tensor
            for override in self.tensor_metric_family_overrides
            if override.depth == depth
            and (tensor is None or override.tensor == tensor)
        )
        if not family_tensors:
            return selected
        if not isinstance(selected, ElementwiseGate):  # pragma: no cover
            raise TypeError(
                "tensor metric-family override resolved a non-elementwise "
                f"gate for {depth}/{tensor!r}"
            )
        return replace(selected, near_zero=family_tensors)


@dataclass(frozen=True)
class GenerationGate(Gate):
    """Thresholds used directly by end-to-end token generation tests."""

    top1_min: float | None = None
    repetition_policy: Literal[
        "diagnostic", "forbid", "allow_baseline_shared"
    ] = "diagnostic"

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (("top1_min", self.top1_min),):
            if isinstance(value, bool):
                raise TypeError(f"GenerationGate.{name} must be numeric, not bool")
            if value is not None and not math.isfinite(value):
                raise ValueError(f"GenerationGate.{name} must be finite")
        if self.top1_min is not None and not 0.0 <= self.top1_min <= 1.0:
            raise ValueError("GenerationGate.top1_min must be in [0, 1]")
        if self.repetition_policy not in (
            "diagnostic",
            "forbid",
            "allow_baseline_shared",
        ):
            raise ValueError(
                "GenerationGate.repetition_policy must be diagnostic, forbid, "
                "or allow_baseline_shared"
            )
        if (
            self.repetition_policy == "allow_baseline_shared"
            and not self.reason.strip()
        ):
            raise ValueError(
                "allow_baseline_shared repetition_policy requires a documented reason"
            )

    @property
    def is_placeholder(self) -> bool:
        return self.top1_min is None

@dataclass(frozen=True)
class ActionGate(Gate):
    """Thresholds used directly by VLA end-to-end action tests."""

    action_cosine: float | None = None
    action_max_abs: float | None = None
    norm_ratio_min: float | None = None
    norm_ratio_max: float | None = None
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("action_cosine", self.action_cosine),
            ("action_max_abs", self.action_max_abs),
            ("norm_ratio_min", self.norm_ratio_min),
            ("norm_ratio_max", self.norm_ratio_max),
        ):
            if isinstance(value, bool):
                raise TypeError(f"ActionGate.{name} must be numeric, not bool")
            if value is not None and not math.isfinite(value):
                raise ValueError(f"ActionGate.{name} must be finite")
        if self.action_cosine is not None and not -1.0 <= self.action_cosine <= 1.0:
            raise ValueError("ActionGate.action_cosine must be in [-1, 1]")
        if self.action_max_abs is not None and self.action_max_abs < 0.0:
            raise ValueError("ActionGate.action_max_abs must be >= 0")
        if self.norm_ratio_min is not None and self.norm_ratio_min < 0.0:
            raise ValueError("ActionGate.norm_ratio_min must be >= 0")
        if self.norm_ratio_max is not None and self.norm_ratio_max < 0.0:
            raise ValueError("ActionGate.norm_ratio_max must be >= 0")
        if (
            self.norm_ratio_min is not None
            and self.norm_ratio_max is not None
            and self.norm_ratio_min > self.norm_ratio_max
        ):
            raise ValueError(
                "ActionGate.norm_ratio_min must be <= norm_ratio_max"
            )
        if self.value_range is not None:
            if len(self.value_range) != 2:
                raise ValueError("ActionGate.value_range must contain (lo, hi)")
            lo, hi = self.value_range
            if isinstance(lo, bool) or isinstance(hi, bool):
                raise TypeError("ActionGate.value_range must be numeric, not bool")
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError("ActionGate.value_range must be finite")
            if lo > hi:
                raise ValueError("ActionGate.value_range must satisfy lo <= hi")

    @property
    def is_placeholder(self) -> bool:
        return self.value_range is None or any(
            value is None
            for value in (
                self.action_cosine,
                self.action_max_abs,
                self.norm_ratio_min,
                self.norm_ratio_max,
            )
        )


@dataclass(frozen=True)
class CopySync:
    """Verifier-kernel and model-production trees checked after migration."""

    model_side: Path
    verifier_side: Path


@dataclass(frozen=True)
class ModelVerifierConfig:
    """One per `<model>/<hardware>/`."""

    model_key: str
    hardware: str
    model_class: str
    stages: tuple[str, ...]
    oracle: OracleSpec
    dtype: str = "bf16"
    gates: Mapping[str, Gate | StageDepthGates | ToleranceProfile] = field(
        default_factory=dict
    )
    #: stage -> why it is not in `stages`.
    disabled_stages: Mapping[str, str] = field(default_factory=dict)
    #: None = this suite has no verifier-side kernel mirror (explicit exemption).
    kernel_dirs: CopySync | None = None
    #: Shape facts demoted from prose docs into data.
    shapes: Mapping[str, Any] = field(default_factory=dict)
    #: pytest markers this suite's conftest should register.
    markers: Sequence[tuple[str, str]] = ()
    #: Free per-test CUDA memory between tests (large-model suites).
    cuda_cleanup_per_test: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if len(set(self.stages)) != len(self.stages):
            raise ValueError(
                f"{self.model_key}/{self.hardware}: `stages` contains duplicates"
            )
        unknown = [s for s in self.stages if s not in CANONICAL_STAGES]
        if unknown:
            raise ValueError(
                f"{self.model_key}/{self.hardware}: unknown stage(s) {unknown}; "
                f"the canonical set is {CANONICAL_STAGES}"
            )
        unknown_disabled = [
            stage for stage in self.disabled_stages if stage not in CANONICAL_STAGES
        ]
        if unknown_disabled:
            raise ValueError(
                f"{self.model_key}/{self.hardware}: disabled_stages contains unknown "
                f"stage(s) {unknown_disabled}"
            )
        overlap = [stage for stage in self.stages if stage in self.disabled_stages]
        if overlap:
            raise ValueError(
                f"{self.model_key}/{self.hardware}: stage(s) {overlap} are both enabled "
                "and disabled"
            )
        undeclared = [
            s for s in CANONICAL_STAGES
            if s not in self.stages and s not in self.disabled_stages
        ]
        if undeclared:
            raise ValueError(
                f"{self.model_key}/{self.hardware}: stage(s) {undeclared} are neither "
                "enabled nor given a reason in `disabled_stages`. Silence reads as "
                "'nobody looked'; say which it is."
            )

        unknown_gate_stages = [
            stage for stage in self.gates if stage not in CANONICAL_STAGES
        ]
        if unknown_gate_stages:
            raise ValueError(
                f"{self.model_key}/{self.hardware}: gates contains unknown stage(s) "
                f"{unknown_gate_stages}"
            )
        for stage, configured_gate in self.gates.items():
            gate = (
                configured_gate.gates
                if isinstance(configured_gate, ToleranceProfile)
                else configured_gate
            )
            if stage == "S0":
                valid = isinstance(gate, CosineGate)
                expected = "CosineGate"
            elif stage in NUMERIC_STAGES:
                # Flat ElementwiseGate is the explicit compatibility path for the
                # three suites that predate the op/layer/module matrix.  A caller
                # selecting a depth still fails loudly in gate() below.
                valid = isinstance(gate, (ElementwiseGate, StageDepthGates))
                expected = (
                    "ElementwiseGate (legacy), StageDepthGates, or "
                    "ToleranceProfile"
                )
            elif stage == "S4":
                if self.model_class == "LLM":
                    valid = isinstance(gate, GenerationGate)
                    expected = "GenerationGate"
                elif self.model_class == "VLA":
                    valid = isinstance(gate, ActionGate)
                    expected = "ActionGate"
                else:
                    raise ValueError(
                        f"{self.model_key}/{self.hardware}: cannot select an S4 gate "
                        f"for unknown model_class {self.model_class!r}"
                    )
            else:  # pragma: no cover - canonical stage validation is exhaustive
                raise AssertionError(f"unhandled canonical stage {stage!r}")
            if not valid:
                raise TypeError(
                    f"{self.model_key}/{self.hardware}: gate {stage!r} must be "
                    f"{expected}, got {type(gate).__name__}"
                )

        missing_enabled_gates = [
            stage for stage in self.stages if stage not in self.gates
        ]
        if missing_enabled_gates:
            raise ValueError(
                f"{self.model_key}/{self.hardware}: enabled stage(s) "
                f"{missing_enabled_gates} have no gate configuration"
            )

    @overload
    def gate(self, stage: str) -> Gate | StageDepthGates | None: ...

    @overload
    def gate(
        self,
        stage: str,
        depth: NumericDepth,
        tensor: str | None = None,
    ) -> Gate | None: ...

    def gate(
        self,
        stage: str,
        depth: NumericDepth | None = None,
        tensor: str | None = None,
    ) -> Gate | StageDepthGates | None:
        """Return one stage gate, optionally selecting a depth and exact tensor.

        ``gate("S4")`` retains the original one-argument behaviour.  During suite
        migration, an old flat S1/S2/S3 gate may still be retrieved without ``depth``;
        asking it for a depth fails loudly rather than pretending one threshold family
        is valid for all three depths.  Tensor lookup is config-owned and exact: an
        override wins when ``(depth, tensor)`` matches, otherwise the depth default is
        returned.
        """
        configured = self.gates.get(stage)
        if depth is None:
            if tensor is not None:
                raise ValueError("tensor gate lookup requires a numeric depth")
            return (
                configured.gates
                if isinstance(configured, ToleranceProfile)
                else configured
            )
        if stage not in NUMERIC_STAGES:
            raise ValueError(
                f"depth is only valid for {NUMERIC_STAGES}, not stage {stage!r}"
            )
        if depth not in NUMERIC_DEPTHS:
            raise ValueError(
                f"unknown numeric depth {depth!r}; choose one of {NUMERIC_DEPTHS}"
            )
        if tensor is not None:
            if not isinstance(tensor, str):
                raise TypeError("tensor gate lookup requires a string tensor name")
            if not tensor.strip():
                raise ValueError("tensor gate lookup requires a non-empty tensor name")
            if tensor != tensor.strip():
                raise ValueError(
                    "tensor gate lookup name must not have surrounding whitespace"
                )
        if configured is None:
            return None
        if isinstance(configured, ToleranceProfile):
            return configured.gate(depth, tensor=tensor)
        if not isinstance(configured, StageDepthGates):
            raise ValueError(
                f"{self.model_key}/{self.hardware}: {stage} still uses a legacy flat "
                "gate; migrate it to StageDepthGates before selecting a depth"
            )
        return configured.gate(depth, tensor=tensor)

    def tolerance_profile(self, stage: str) -> ToleranceProfile | None:
        """Return a stage's named profile without changing legacy ``gate()`` callers."""
        if stage not in NUMERIC_STAGES:
            raise ValueError(
                f"tolerance profiles are only valid for {NUMERIC_STAGES}, not {stage!r}"
            )
        configured = self.gates.get(stage)
        return configured if isinstance(configured, ToleranceProfile) else None
