"""S3 layer: PyVM and megakernel Gemma layer."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, launch_native, make_gemma_globs, make_pack, prepare_gemma_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


@pytest.mark.parametrize("views", (1, 3))
def test_gemma_layer(verifier_kernels: dict[str, object], views: int) -> None:
    pack = make_pack(views=views, chunk=20)
    prepare_gemma_native(pack, seed=540 + views)
    globs = make_gemma_globs(pack, layers=1, seed=540 + views)
    seq = globs._pyvm_meta.seq_len
    instructions, _ = scheduler.schedule_gemma_layer(
        seq, layer_idx=0, sm_count=C.sm_count
    )
    python_vm.run_gemma_pyvm(globs, instructions)
    launch_native(verifier_kernels["gemma"], pack.gemma, instructions)
    gate = CONFIG.gate("S3", "layer", tensor="gemma")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        pack.gemma.tensors["encoder_x"][0, 0, :seq].float(),
        globs.encoder_x[0, 0, :seq].float(),
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name=f"gemma_layer_view{views}",
    )
