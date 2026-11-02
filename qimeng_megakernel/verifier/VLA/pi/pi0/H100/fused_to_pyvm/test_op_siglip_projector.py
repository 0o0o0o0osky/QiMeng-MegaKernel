"""S2 op: fused and PyVM SigLIP projector."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.fused import siglip_ops
from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, make_siglip_globs, seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def test_siglip_projector() -> None:
    globs, _ = make_siglip_globs(views=2, layers=1, seed=401)
    x = seeded(C.siglip_seq_for(2), C.vision_width, seed=402)
    globs.siglip_x_normed[0, 0].copy_(x)
    expected = siglip_ops.projector(x, globs.proj_w[0], globs.proj_b[0, 0])
    instructions = scheduler.schedule_siglip_projector(
        num_views=2, dep=(0, 0, 0), sm_count=C.sm_count
    )
    python_vm.run_siglip_pyvm(globs, instructions)
    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        globs.proj_out[0, 0], expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="siglip_projector",
    )
