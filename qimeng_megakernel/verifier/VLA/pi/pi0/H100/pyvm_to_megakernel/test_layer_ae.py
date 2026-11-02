"""S3 layer: PyVM and megakernel Action Expert layer."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, launch_native, make_ae_globs, make_pack, prepare_ae_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


@pytest.mark.parametrize("views,chunk", ((1, 20), (3, 50)))
def test_ae_layer(
    verifier_kernels: dict[str, object], views: int, chunk: int
) -> None:
    pack = make_pack(views=views, chunk=chunk)
    prepare_ae_native(pack, seed=550 + views + chunk)
    globs = make_ae_globs(pack, layers=1, seed=550 + views + chunk)
    instructions, _ = scheduler.schedule_ae_layer(
        0, pack.ae_layout, sm_count=C.sm_count
    )
    python_vm.run_ae_pyvm(globs, instructions)
    launch_native(verifier_kernels["ae"], pack.ae, instructions)
    gate = CONFIG.gate("S3", "layer", tensor="ae")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        pack.ae.tensors["x_acc"][0, 0].float(), globs.x_acc[0, 0].float(),
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name=f"ae_layer_view{views}_chunk{chunk}",
    )
