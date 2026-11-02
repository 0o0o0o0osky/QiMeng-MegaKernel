"""S3 layer: PyVM and megakernel SigLIP layer."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, launch_native, make_pack, make_siglip_globs, prepare_siglip_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


@pytest.mark.parametrize("views", (1, 2, 3))
def test_siglip_layer(verifier_kernels: dict[str, object], views: int) -> None:
    pack = make_pack(views=views, chunk=20)
    prepare_siglip_native(pack, seed=530, include_head_tail=False)
    globs, _ = make_siglip_globs(views=views, layers=1, seed=530)
    instructions = scheduler.schedule_siglip_layer(
        layer_idx=0, num_views=views, sm_count=C.sm_count
    )
    python_vm.run_siglip_pyvm(globs, instructions)
    launch_native(verifier_kernels[f"siglip_view{views}"], pack.siglip, instructions)
    gate = CONFIG.gate("S3", "layer", tensor="siglip")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        pack.siglip.tensors["vision_x"][0, 0, :, : C.vision_width].float(),
        globs.vision_x[0, 0].float(),
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name=f"siglip_layer_view{views}",
    )
