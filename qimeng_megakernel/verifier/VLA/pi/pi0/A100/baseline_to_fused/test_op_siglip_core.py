"""S1 ops: remaining SigLIP fusion boundaries on real checkpoint tensors."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import siglip_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import siglip_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
    make_siglip_globs,
    seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


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


def test_siglip_layernorm() -> None:
    _, weights = make_siglip_globs(views=1, layers=1, seed=130)
    x = seeded(8, C.vision_width, seed=131)
    actual = fused.layernorm(x, weights[0]["g1"], weights[0]["b1"])
    expected = siglip_ops.layernorm(x, weights[0]["g1"], weights[0]["b1"])
    _assert(actual, expected, "siglip_layernorm")


def test_siglip_attention_dense_and_head_padded() -> None:
    globs, weights = make_siglip_globs(views=1, layers=1, seed=132)
    x = globs.vision_x[0, 0]
    normed = siglip_ops.layernorm(x, weights[0]["g1"], weights[0]["b1"])
    qkv = (normed.float() @ weights[0]["Wqkv"].float().T + weights[0]["bqkv"].float()).bfloat16()
    dense_expected = siglip_ops.baseline_siglip_attention(qkv, num_views=1)
    dense_actual = fused.fused_siglip_attention(qkv, num_views=1)
    _assert(dense_actual, dense_expected, "siglip_attention_dense")

    padded = torch.zeros(
        qkv.shape[0],
        3 * C.vision_num_heads * C.siglip_head_dim_pad,
        device=qkv.device,
        dtype=qkv.dtype,
    )
    for section in range(3):
        for head in range(C.vision_num_heads):
            source = section * C.vision_width + head * C.siglip_head_dim
            target = (
                section * C.vision_num_heads * C.siglip_head_dim_pad
                + head * C.siglip_head_dim_pad
            )
            padded[:, target : target + C.siglip_head_dim] = qkv[
                :, source : source + C.siglip_head_dim
            ]
    padded_actual = fused.fused_siglip_attention_head_padded(padded, num_views=1)
    padded_expected = siglip_ops.baseline_siglip_attention_head_padded(padded, num_views=1)
    _assert(padded_actual, padded_expected, "siglip_attention_head_padded")


def test_siglip_mlp() -> None:
    _, weights = make_siglip_globs(views=1, layers=1, seed=133)
    x = seeded(8, C.vision_width, seed=134)
    w = weights[0]
    actual = fused.fused_mlp_gelu_tanh(x, w["W1"], w["b1f"], w["W2"], w["b2f"])
    expected = siglip_ops.mlp_gelu_tanh(x, w["W1"], w["b1f"], w["W2"], w["b2f"])
    _assert(actual, expected, "siglip_mlp")


def test_siglip_patch_and_final_norm() -> None:
    globs, _ = make_siglip_globs(
        views=1,
        layers=1,
        seed=135,
        include_head_tail=True,
    )
    patch_rows = globs.patch_in[0, 0]
    position = globs.vision_x[0, 0]
    actual_patch = fused.patch_linear_residual(
        patch_rows,
        globs.patch_w[0],
        globs.patch_b[0, 0],
        position,
    )
    expected_patch = siglip_ops.baseline_gemm(
        patch_rows,
        globs.patch_w[0],
        globs.patch_b[0, 0],
        patch_rows.shape[0],
        3 * 14 * 14,
        residual=position,
    ).to(position.dtype)
    _assert(actual_patch, expected_patch, "siglip_patch")

    gamma, beta = globs.ln_gamma[0, 0, -1], globs.ln_beta[0, 0, -1]
    actual_norm = fused.final_norm(actual_patch, gamma, beta)
    expected_norm = siglip_ops.final_norm(expected_patch, gamma, beta)
    _assert(actual_norm, expected_norm, "siglip_final_norm")

