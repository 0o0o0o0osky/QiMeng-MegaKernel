"""S2 layer: fused and PyVM Gemma layer."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, gemma_weights, make_gemma_globs, make_pack, run_gemma_fused,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def test_gemma_layer() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_gemma_globs(pack, layers=1, seed=440)
    seq = globs._pyvm_meta.seq_len
    expected = run_gemma_fused(
        globs.encoder_x[0, 0, :seq], gemma_weights(globs, 1),
        globs.rope[0, 0, :seq], seq_len=seq,
    )
    instructions, _ = scheduler.schedule_gemma_layer(
        seq, layer_idx=0, sm_count=C.sm_count
    )
    python_vm.run_gemma_pyvm(globs, instructions)
    gate = CONFIG.gate("S2", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        globs.encoder_x[0, 0, :seq].float(), expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="gemma_layer",
    )
