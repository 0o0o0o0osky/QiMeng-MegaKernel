"""S1 module: baseline and fused SigLIP stack."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import siglip_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import siglip_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    make_siglip_globs,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_siglip_module() -> None:
    globs, weights = make_siglip_globs(
        views=2, layers=C.vision_layers, seed=120, include_head_tail=True
    )
    position = globs.vision_x[0, 0]
    patches = globs.patch_in[0, 0]
    baseline_x = siglip_ops.baseline_gemm(
        patches, globs.patch_w[0], globs.patch_b[0, 0],
        patches.shape[0], patches.shape[1], residual=position,
    ).to(torch.bfloat16)
    actual_x = fused.patch_linear_residual(
        patches, globs.patch_w[0], globs.patch_b[0, 0], position
    )
    expected = siglip_ops.projector(
        siglip_ops.final_norm(
            siglip_ops.baseline_siglip_stack(baseline_x, weights, num_views=2),
            globs.ln_gamma[0, 0, -1], globs.ln_beta[0, 0, -1],
        ),
        globs.proj_w[0], globs.proj_b[0, 0],
    )
    actual = fused.projector(
        fused.final_norm(
            fused.fused_siglip_stack(actual_x, weights, num_views=2),
            globs.ln_gamma[0, 0, -1], globs.ln_beta[0, 0, -1],
        ),
        globs.proj_w[0], globs.proj_b[0, 0],
    )
    gate = CONFIG.gate("S1", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        actual, expected, min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="siglip_module",
    )
