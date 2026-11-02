"""S3 layer: PyVM and the production-form Gemma layer."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.prefill import build_prefill_schedules
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import GemmaProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, make_gemma_globs, make_pack, prepare_gemma_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


@pytest.mark.parametrize("views", (1, 2, 3))
def test_gemma_layer(verifier_dsos: dict[str, object], views: int) -> None:
    pack = make_pack(views=views, chunk=20)
    prepare_gemma_native(pack, seed=540 + views)
    globs = make_gemma_globs(pack, layers=1, seed=540 + views)
    seq = globs._pyvm_meta.seq_len
    instructions, _ = scheduler.schedule_gemma_layer(
        seq, layer_idx=0
    )
    python_vm.run_gemma_pyvm(globs, instructions)
    GemmaProgram(
        pack, build_prefill_schedules(pack), verifier_dsos["gemma"], instructions
    ).run()
    torch.cuda.synchronize(pack.gemma.tensors["encoder_x"].device)
    actual = pack.gemma.tensors["encoder_x"][0, 0, :seq].float()
    expected = globs.encoder_x[0, 0, :seq].float()
    gate = CONFIG.gate("S3", "layer", tensor="gemma")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    actual_rms = actual.square().mean(dim=-1, keepdim=True).sqrt()
    expected_rms = expected.square().mean(dim=-1, keepdim=True).sqrt()
    assert_max_rel(
        actual_rms, expected_rms,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name=f"gemma_layer_hidden_rms_view{views}",
    )
    actual = actual * torch.rsqrt(actual_rms.square() + C.rms_eps)
    expected = expected * torch.rsqrt(expected_rms.square() + C.rms_eps)
    assert_max_rel(
        actual,
        expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name=f"gemma_layer_view{views}",
    )
