"""S3 op: PyVM and the production-form native Action Expert input projection."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.decode import build_decode_schedule
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import (
    instruction_defs as I, python_vm, scheduler,
)
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import ActionExpertProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    make_ae_globs, make_pack, prepare_ae_native, seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def test_ae_act_in(verifier_dsos: dict[str, object]) -> None:
    pack = make_pack(views=2, chunk=20)
    prepare_ae_native(pack, seed=521)
    globs = make_ae_globs(pack, layers=1, seed=521)
    noise = seeded(*globs.noise.shape, seed=522)
    globs.noise.copy_(noise)
    pack.ae.tensors["noise"].copy_(noise)
    instructions = [
        item
        for item in scheduler.schedule_ae_flow(pack.ae_layout)
        if isinstance(item, I.AeActIn) and item.bias_row == 0
    ]
    python_vm.run_ae_pyvm(globs, instructions)
    ActionExpertProgram(
        pack, build_decode_schedule(pack), verifier_dsos["ae"], instructions
    ).run()
    torch.cuda.synchronize(pack.ae.tensors["x"].device)
    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    query_rows = pack.chunk + 1
    assert_max_rel(
        pack.ae.tensors["act_h"][0, 0, :query_rows],
        globs.act_h[0, 0, :query_rows],
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="ae_act_in",
    )
