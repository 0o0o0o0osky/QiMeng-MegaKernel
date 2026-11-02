"""S3 layer: PyVM and the production-form Action Expert layer."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.decode import build_decode_schedule
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import ActionExpertProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    make_ae_globs, make_pack, prepare_ae_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


@pytest.mark.parametrize("views,chunk", ((1, 20), (3, 50)))
def test_ae_layer(
    verifier_dsos: dict[str, object], views: int, chunk: int
) -> None:
    pack = make_pack(views=views, chunk=chunk)
    prepare_ae_native(pack, seed=550 + views + chunk)
    globs = make_ae_globs(pack, layers=1, seed=550 + views + chunk)
    instructions, _ = scheduler.schedule_ae_layer(
        0, pack.ae_layout
    )
    python_vm.run_ae_pyvm(globs, instructions)
    ActionExpertProgram(
        pack, build_decode_schedule(pack), verifier_dsos["ae"], instructions
    ).run()
    torch.cuda.synchronize(pack.ae.tensors["x"].device)
    gate = CONFIG.gate("S3", "layer", tensor="ae")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        pack.ae.tensors["x_acc"][0, 0, : pack.chunk + 1].float(),
        globs.x_acc[0, 0, : pack.chunk + 1].float(),
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name=f"ae_layer_view{views}_chunk{chunk}",
    )


def test_ae_layer_masks_nonfinite_kv_padding(
    verifier_dsos: dict[str, object],
) -> None:
    pack = make_pack(views=1, chunk=20)
    seed = 591
    prepare_ae_native(pack, seed=seed)
    globs = make_ae_globs(pack, layers=1, seed=seed)
    prefix = pack.ae_layout["prefix_len"]
    state_key = pack.ae_layout["state_key"]
    pack.ae.tensors["k_cache"][0, 0, prefix:state_key].fill_(float("nan"))
    pack.ae.tensors["v_cache"][0, 0, prefix:state_key].fill_(float("nan"))
    instructions, _ = scheduler.schedule_ae_layer(
        0, pack.ae_layout
    )

    python_vm.run_ae_pyvm(globs, instructions)
    ActionExpertProgram(
        pack, build_decode_schedule(pack), verifier_dsos["ae"], instructions
    ).run()
    torch.cuda.synchronize(pack.ae.tensors["x"].device)
    actual = pack.ae.tensors["x_acc"][0, 0, : pack.ae_layout["seq"]].float()
    expected = globs.x_acc[0, 0, : pack.ae_layout["seq"]].float()
    assert torch.isfinite(actual).all()
    gate = CONFIG.gate("S3", "layer", tensor="ae")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        actual,
        expected,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name="ae_layer_nonfinite_kv_padding",
    )
