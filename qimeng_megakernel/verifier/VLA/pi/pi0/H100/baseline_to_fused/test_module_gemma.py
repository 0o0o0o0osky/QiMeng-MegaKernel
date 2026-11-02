"""S1 module: baseline and fused Gemma stack."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.fused import gemma_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.official_to_baseline import gemma_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    gemma_weights,
    make_gemma_globs,
    make_pack,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_gemma_module() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_gemma_globs(pack, layers=C.encoder_layers, seed=220)
    seq = globs._pyvm_meta.seq_len
    weights = gemma_weights(globs, C.encoder_layers)
    x = globs.encoder_x[0, 0, :seq]
    rope = globs.rope[0, 0, :seq]
    expected = gemma_ops.baseline_gemma_stack(x, weights, rope, seq)
    actual = fused.fused_gemma_stack(x, weights, rope, seq)
    gate = CONFIG.gate("S1", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        actual, expected, min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="gemma_module",
    )
