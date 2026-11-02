"""S1 op: baseline and fused Action Expert input projection."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import ae_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    make_ae_globs,
    make_pack,
    seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_ae_act_in() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_ae_globs(pack, layers=1, seed=301)
    noise = seeded(C.ae_m, 256, seed=302)
    weight = globs.act_w_in[0, 0]
    bias = globs.act_bias[0, 0, 0]
    pre = noise.float() @ weight.float().T + bias.float()
    expected = (pre / (1.0 + torch.exp(-pre))).to(torch.bfloat16)
    actual = fused.fused_ae_act_in(noise, weight, bias)
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        actual, expected, max_rel=gate.max_rel, abs_floor=gate.abs_floor,
        atol=gate.atol, name="ae_act_in",
    )
