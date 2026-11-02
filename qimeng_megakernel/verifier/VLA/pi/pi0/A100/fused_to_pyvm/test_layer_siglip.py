"""S2 layer: fused and PyVM SigLIP layer."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    make_siglip_globs, run_siglip_fused,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def test_siglip_layer() -> None:
    globs, weights = make_siglip_globs(views=2, layers=1, seed=430)
    expected = run_siglip_fused(globs.vision_x[0, 0], weights, num_views=2)
    instructions = scheduler.schedule_siglip_layer(
        layer_idx=0, num_views=2
    )
    python_vm.run_siglip_pyvm(globs, instructions)
    gate = CONFIG.gate("S2", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        globs.vision_x[0, 0].float(), expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="siglip_layer",
    )
