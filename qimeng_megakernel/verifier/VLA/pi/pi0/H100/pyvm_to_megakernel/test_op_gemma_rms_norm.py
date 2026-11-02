"""S3 op: PyVM and megakernel Gemma RMS normalization."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import (
    instruction_defs as I, python_vm, scheduler,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, launch_native, make_gemma_globs, make_pack, prepare_gemma_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def test_gemma_rms_norm(verifier_kernels: dict[str, object]) -> None:
    pack = make_pack(views=2, chunk=20)
    prepare_gemma_native(pack, seed=511)
    globs = make_gemma_globs(pack, layers=1, seed=511)
    seq = globs._pyvm_meta.seq_len
    layer, _ = scheduler.schedule_gemma_layer(seq, layer_idx=0, sm_count=C.sm_count)
    instructions = [item for item in layer if isinstance(item, I.GemmaRmsnAttn)]
    python_vm.run_gemma_pyvm(globs, instructions)
    launch_native(verifier_kernels["gemma"], pack.gemma, instructions)
    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        pack.gemma.tensors["rms_factor"][0, 0, 0, :seq],
        globs.rms_factor[0, 0, 0, :seq],
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="gemma_rms_inv",
    )
