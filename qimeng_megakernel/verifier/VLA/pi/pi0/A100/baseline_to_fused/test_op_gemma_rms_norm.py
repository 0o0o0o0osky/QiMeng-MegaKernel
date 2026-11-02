"""S1 op: baseline and fused Gemma RMS normalization."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import gemma_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import gemma_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_gemma_rms_norm() -> None:
    x = seeded(8, C.encoder_width, seed=201)
    actual = fused.rms_inv(x)
    expected = gemma_ops.rms_inv(x)
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        actual, expected, max_rel=gate.max_rel, abs_floor=gate.abs_floor,
        atol=gate.atol, name="gemma_rms_inv",
    )
