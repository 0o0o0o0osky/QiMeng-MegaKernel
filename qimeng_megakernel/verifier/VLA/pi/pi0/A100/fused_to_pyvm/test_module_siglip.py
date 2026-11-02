"""S2 module: fused and PyVM SigLIP stack."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import siglip_ops
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, make_siglip_globs, run_siglip_fused,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_siglip_module() -> None:
    globs, weights = make_siglip_globs(
        views=2, layers=C.vision_layers, seed=460, include_head_tail=True
    )
    x = siglip_ops.patch_linear_residual(
        globs.patch_in[0, 0], globs.patch_w[0], globs.patch_b[0, 0],
        globs.vision_x[0, 0],
    )
    x = run_siglip_fused(x, weights, num_views=2)
    x = siglip_ops.final_norm(
        x, globs.ln_gamma[0, 0, -1], globs.ln_beta[0, 0, -1]
    )
    expected = siglip_ops.projector(x, globs.proj_w[0], globs.proj_b[0, 0])
    python_vm.run_siglip_pyvm(
        globs, scheduler.schedule_siglip_runtime(num_views=2)
    )
    gate = CONFIG.gate("S2", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        globs.proj_out[0, 0], expected,
        min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="siglip_module",
    )
