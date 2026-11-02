"""S1 op: AE RMS projection preserves the SM80 BF16 landing order."""
from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import ae_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    require_checkpoint,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_ae_rms_linear_lands_before_gemm_not_after_normalization() -> None:
    with safe_open(str(require_checkpoint()), framework="pt", device="cpu") as handle:
        seed = handle.get_tensor("action_time_mlp_in.bias").float()
        weight = handle.get_tensor(
            "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
        )[:64].cuda()
    offsets = torch.tensor([0.001, -0.003, 0.007, -0.011])[:, None]
    x_acc = (seed[None] + offsets).cuda()
    actual = fused.fused_ae_rms_linear(x_acc, weight)

    landed = x_acc.bfloat16().float()
    inv = torch.rsqrt((x_acc * x_acc).mean(-1) + C.rms_eps)
    expected = inv[:, None] * (landed @ weight.float().T)
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert_max_rel(
        actual,
        expected,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name="ae_rms_linear",
    )

    premature = (x_acc * inv[:, None]).bfloat16().float() @ weight.float().T
    assert (actual - premature).abs().max().item() > 1e-3

