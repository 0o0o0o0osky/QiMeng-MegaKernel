"""S1 layer: baseline and fused Gemma layer."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import gemma_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import gemma_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    gemma_weights,
    make_gemma_globs,
    make_pack,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_gemma_layer() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_gemma_globs(pack, layers=1, seed=210)
    seq = globs._pyvm_meta.seq_len
    weights = gemma_weights(globs, 1)[0]
    x = globs.encoder_x[0, 0, :seq]
    rope = globs.rope[0, 0, :seq]
    expected = gemma_ops.baseline_gemma_layer(x, *weights, rope, seq)
    actual = fused.fused_gemma_layer(x, *weights, rope, seq)
    gate = CONFIG.gate("S1", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        actual, expected, max_rel=gate.max_rel, abs_floor=gate.abs_floor,
        atol=gate.atol, name="gemma_layer",
    )
