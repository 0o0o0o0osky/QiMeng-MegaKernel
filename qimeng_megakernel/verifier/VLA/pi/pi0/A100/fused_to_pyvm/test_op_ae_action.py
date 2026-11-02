"""S2 op: fused and PyVM Action Expert action head."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import ae_ops
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import (
    instruction_defs as I,
    python_vm,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    make_ae_globs,
    make_pack,
    seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def _dependency(instruction_type: type) -> dict[str, int]:
    return {
        "dep_layer": 0,
        "dep_row": I.ae_barrier_row(instruction_type.opcode()),
        "dep_expected": 1,
    }


def test_ae_action() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_ae_globs(pack, layers=1, seed=421)
    layout = globs._pyvm_meta.layout
    globs.noise.copy_(seeded(*globs.noise.shape, seed=422))
    initial_noise = globs.noise[0, 0].clone()

    expected_act_h = ae_ops.fused_ae_act_in(
        initial_noise,
        globs.act_w_in[0, 0],
        globs.act_bias[0, 0, 0],
    )
    expected_x_acc = ae_ops.fused_ae_act_mlp(
        expected_act_h,
        globs.act_w_mlp[0, 0],
        globs.act_bias[0, 0, 10],
        globs.act_bias[0, 0, 12],
        n_valid_row=layout["seq"],
    )
    expected_landing, factor = ae_ops.fused_ae_rmsn_landing(expected_x_acc)
    expected_noise = ae_ops.fused_ae_act_out(
        initial_noise[:, : globs.act_w_out.shape[-2]],
        expected_landing,
        globs.act_w_out[0, 0],
        globs.act_bias[0, 0, 11, : globs.act_w_out.shape[-2]],
        factor,
    )

    instructions = [
        I.AeActIn(plane=0, bar_layer=0, bias_row=0),
        I.AeActMlp(
            plane=0,
            bar_layer=0,
            bias_row=10,
            n_valid_row=layout["seq"],
            **_dependency(I.AeActIn),
        ),
        I.AeRmsnOut(
            bar_layer=0,
            token_row=0,
            num_tokens=layout["m_pad"],
            **_dependency(I.AeActMlp),
        ),
        I.AeActOut(
            plane=0,
            bar_layer=0,
            bias_row=11,
            norm_sel=2,
            **_dependency(I.AeRmsnOut),
        ),
    ]
    python_vm.run_ae_pyvm(globs, instructions)

    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    for name, actual, expected in (
        ("ae_action_input", globs.act_h[0, 0], expected_act_h),
        ("ae_action_mlp", globs.x_acc[0, 0], expected_x_acc),
        ("ae_action_landing", globs.x[0, 0], expected_landing),
        (
            "ae_action_output",
            globs.noise[0, 0, :, : expected_noise.shape[-1]],
            expected_noise,
        ),
    ):
        assert_max_rel(
            actual,
            expected,
            max_rel=gate.max_rel,
            abs_floor=gate.abs_floor,
            atol=gate.atol,
            name=name,
        )
    assert torch.equal(
        globs.noise[0, 0, :, expected_noise.shape[-1] :],
        initial_noise[:, expected_noise.shape[-1] :],
    )
