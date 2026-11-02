"""S2 layer: fused and PyVM Action Expert layer."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, make_ae_globs, make_pack, run_ae_fused,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def test_ae_layer() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_ae_globs(pack, layers=1, seed=450)
    expected = run_ae_fused(globs, layers=1)
    instructions, _ = scheduler.schedule_ae_layer(
        0, globs._pyvm_meta.layout, sm_count=C.sm_count
    )
    python_vm.run_ae_pyvm(globs, instructions)
    gate = CONFIG.gate("S2", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        globs.x_acc[0, 0].float(), expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="ae_layer",
    )
