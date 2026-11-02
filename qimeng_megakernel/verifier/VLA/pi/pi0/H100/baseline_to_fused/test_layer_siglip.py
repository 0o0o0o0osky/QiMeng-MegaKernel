"""S1 layer: baseline and fused SigLIP layer."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.fused import siglip_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.official_to_baseline import siglip_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    make_siglip_globs,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_siglip_layer() -> None:
    globs, weights = make_siglip_globs(views=2, layers=1, seed=110)
    x = globs.vision_x[0, 0]
    expected = siglip_ops.baseline_siglip_layer(x, weights[0], num_views=2)
    actual = fused.fused_siglip_layer(x, weights[0], num_views=2)
    gate = CONFIG.gate("S1", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        actual, expected, max_rel=gate.max_rel, abs_floor=gate.abs_floor,
        atol=gate.atol, name="siglip_layer",
    )
