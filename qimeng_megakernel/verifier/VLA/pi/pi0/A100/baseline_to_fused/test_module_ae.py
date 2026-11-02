"""S1 module: baseline and fused Action Expert stack."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import ae_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import ae_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    ae_prefix,
    ae_weights,
    make_ae_globs,
    make_pack,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_ae_module() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_ae_globs(pack, layers=C.decoder_layers, seed=320)
    layout = globs._pyvm_meta.layout
    weights = ae_weights(globs, C.decoder_layers)
    prefix = ae_prefix(globs, C.decoder_layers)
    x = globs.x_acc[0, 0].float()
    expected = ae_ops.baseline_ae_stack(
        x, weights, globs.rope[0, 0], prefix,
        layout=layout, n_layers=C.decoder_layers,
    )
    actual = fused.fused_ae_stack(
        x, weights, globs.rope[0, 0], prefix,
        layout=layout, n_layers=C.decoder_layers,
    )
    gate = CONFIG.gate("S1", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        actual, expected, min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="ae_module",
    )
