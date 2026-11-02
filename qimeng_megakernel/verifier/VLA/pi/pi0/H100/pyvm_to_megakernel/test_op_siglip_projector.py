"""S3 op: PyVM and megakernel SigLIP projector."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, launch_native, make_pack, make_siglip_globs, seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


@pytest.mark.parametrize("views", (1, 2, 3))
def test_siglip_projector(
    verifier_kernels: dict[str, object], views: int
) -> None:
    pack = make_pack(views=views, chunk=20)
    globs, _ = make_siglip_globs(views=views, layers=1, seed=501)
    x = seeded(C.siglip_seq_for(views), C.vision_width, seed=502)
    globs.siglip_x_normed[0, 0].copy_(x)
    pack.siglip.tensors["siglip_x_normed"].zero_()
    pack.siglip.tensors["siglip_x_normed"][0, 0, :, : C.vision_width] = x
    instructions = scheduler.schedule_siglip_projector(
        num_views=views, dep=(0, 0, 0), sm_count=C.sm_count
    )
    python_vm.run_siglip_pyvm(globs, instructions)
    launch_native(verifier_kernels[f"siglip_view{views}"], pack.siglip, instructions)
    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        pack.siglip.tensors["proj_out"][0, 0], globs.proj_out[0, 0],
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="siglip_projector",
    )
