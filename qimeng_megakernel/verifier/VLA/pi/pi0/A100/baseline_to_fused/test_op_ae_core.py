"""S1 ops: remaining Action Expert attention and action-head boundaries."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import ae_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import ae_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    ae_weights,
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


@pytest.fixture(scope="module")
def inputs():
    pack = make_pack(views=1, chunk=20)
    return make_ae_globs(pack, layers=1, seed=330)


def _assert(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        actual,
        expected,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name=name,
    )


def test_ae_action_mlp(inputs) -> None:
    globs = inputs
    act_h = seeded(C.ae_m, C.decoder_width, seed=331)
    state_h = globs.act_bias[0, 0, 12]
    actual = fused.fused_ae_act_mlp(
        act_h,
        globs.act_w_mlp[0, 0],
        globs.act_bias[0, 0, 10],
        state_h,
        n_valid_row=21,
    )
    expected = act_h.float() @ globs.act_w_mlp[0, 0].float().T + globs.act_bias[0, 0, 10].float()
    expected[0] = state_h.float()
    expected[21:] = 0
    _assert(actual, expected, "ae_action_mlp")


def test_ae_attention_mask(inputs) -> None:
    globs = inputs
    layout = globs._pyvm_meta.layout
    weights = ae_weights(globs, 1)
    qkv = fused.fused_ae_rms_linear(globs.x_acc[0, 0], weights["qkv"][0])
    qw, kw = C.num_heads * C.head_dim, C.head_dim
    q = fused.apply_rope_interleaved(qkv[:, :qw], globs.rope[0, 0])
    k = globs.k_cache[0, 0].clone()
    v = globs.v_cache[0, 0].clone()
    k[layout["state_key"] : layout["state_key"] + C.ae_m] = fused.apply_rope_interleaved(
        qkv[:, qw : qw + kw], globs.rope[0, 0]
    ).bfloat16()
    v[layout["state_key"] : layout["state_key"] + C.ae_m] = qkv[:, qw + kw :].bfloat16()
    actual = fused.fused_ae_attn(
        q.bfloat16(),
        k,
        v,
        prefix_len=layout["prefix_len"],
        state_key=layout["state_key"],
        self_end=layout["self_end"],
    )
    expected = ae_ops.baseline_ae_attn(
        q.bfloat16(),
        k,
        v,
        prefix_len=layout["prefix_len"],
        state_key=layout["state_key"],
        self_end=layout["self_end"],
    )
    _assert(actual, expected, "ae_attention")


def test_ae_action_out(inputs) -> None:
    globs = inputs
    noise = seeded(C.ae_m, C.action_dim, seed=332)
    x = seeded(C.ae_m, C.decoder_width, seed=333)
    rms = torch.rsqrt((x.float() ** 2).mean(-1) + C.rms_eps)
    actual = fused.fused_ae_act_out(
        noise,
        x,
        globs.act_w_out[0, 0],
        globs.act_bias[0, 0, 11, : C.action_dim],
        rms,
    )
    delta = rms[:, None] * (x.float() @ globs.act_w_out[0, 0].float().T)
    delta += globs.act_bias[0, 0, 11, : C.action_dim].float()
    expected = (noise.float() + delta.bfloat16().float()).bfloat16()
    assert actual.dtype == expected.dtype == torch.bfloat16
    _assert(actual, expected, "ae_action_out")
