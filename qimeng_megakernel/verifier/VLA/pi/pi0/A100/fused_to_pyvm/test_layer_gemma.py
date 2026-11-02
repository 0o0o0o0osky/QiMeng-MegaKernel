"""S2 layer: fused and PyVM Gemma layer."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, gemma_weights, make_gemma_globs, make_pack, run_gemma_fused,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
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
        seq, layer_idx=0
    )
    python_vm.run_gemma_pyvm(globs, instructions)
    actual = globs.encoder_x[0, 0, :seq].float()
    expected = expected.float()
    gate = CONFIG.gate("S2", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    actual_rms = actual.square().mean(dim=-1, keepdim=True).sqrt()
    expected_rms = expected.square().mean(dim=-1, keepdim=True).sqrt()
    assert_max_rel(
        actual_rms, expected_rms,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="gemma_layer_hidden_rms",
    )
    actual = actual * torch.rsqrt(actual_rms.square() + C.rms_eps)
    expected = expected * torch.rsqrt(expected_rms.square() + C.rms_eps)
    assert_max_rel(
        actual, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="gemma_layer_normalized_hidden",
    )
