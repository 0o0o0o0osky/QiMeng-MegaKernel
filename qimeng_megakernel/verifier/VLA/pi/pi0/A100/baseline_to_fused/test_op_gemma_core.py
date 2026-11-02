"""S1 ops: Gemma QKV/RoPE, attention, output projection, and gated FFN."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import gemma_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import gemma_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    gemma_weights,
    make_gemma_globs,
    make_pack,
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
    globs = make_gemma_globs(pack, layers=1, seed=230)
    seq = globs._pyvm_meta.seq_len
    return globs.encoder_x[0, 0, :seq], globs.rope[0, 0, :seq], gemma_weights(globs, 1)[0]


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


def test_gemma_qkv_rope(inputs) -> None:
    x, rope, weights = inputs
    expected = gemma_ops.qkv_rope(x, weights[0], rope)
    actual = fused.fused_qkv_rope(x, weights[0], rope)
    for name, got, want in zip(("q", "k", "v"), actual, expected, strict=True):
        _assert(got, want, f"gemma_qkv_rope_{name}")


def test_gemma_mqa(inputs) -> None:
    x, rope, weights = inputs
    q, k, v = gemma_ops.qkv_rope(x, weights[0], rope)
    expected = gemma_ops.baseline_mqa_bidirectional(q.bfloat16(), k.bfloat16(), v.bfloat16(), x.shape[0])
    actual = fused.fused_mqa_bidirectional(q.bfloat16(), k.bfloat16(), v.bfloat16(), x.shape[0])
    _assert(actual, expected, "gemma_mqa")


def test_gemma_oproj_residual(inputs) -> None:
    x, rope, weights = inputs
    q, k, v = gemma_ops.qkv_rope(x, weights[0], rope)
    attn = gemma_ops.baseline_mqa_bidirectional(q.bfloat16(), k.bfloat16(), v.bfloat16(), x.shape[0])
    actual = fused.fused_oproj_residual(x, attn, weights[1])
    expected = gemma_ops.oproj_residual(x, attn, weights[1])
    _assert(actual, expected, "gemma_oproj_residual")


def test_gemma_gated_ffn(inputs) -> None:
    x, _, weights = inputs
    actual = fused.fused_ffn_gate_up_down(x, weights[2], weights[3], weights[4])
    expected = gemma_ops.ffn_gate_up_down(x, weights[2], weights[3], weights[4])
    _assert(actual, expected, "gemma_gated_ffn")

