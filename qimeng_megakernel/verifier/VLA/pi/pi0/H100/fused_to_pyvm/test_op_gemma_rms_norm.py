"""S2 op: fused and PyVM Gemma RMS normalization."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.fused import gemma_ops
from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import (
    instruction_defs as I, python_vm, scheduler,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, make_gemma_globs, make_pack,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def test_gemma_rms_norm() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_gemma_globs(pack, layers=1, seed=411)
    seq = globs._pyvm_meta.seq_len
    expected = gemma_ops.rms_inv(globs.encoder_x[0, 0, :seq]).squeeze(-1)
    layer, _ = scheduler.schedule_gemma_layer(seq, layer_idx=0, sm_count=C.sm_count)
    instructions = [item for item in layer if isinstance(item, I.GemmaRmsnAttn)]
    python_vm.run_gemma_pyvm(globs, instructions)
    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        globs.rms_factor[0, 0, 0, :seq], expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="gemma_rms_inv",
    )
