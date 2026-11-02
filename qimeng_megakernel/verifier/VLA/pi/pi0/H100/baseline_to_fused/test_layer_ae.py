"""S1 layer: baseline and fused Action Expert layer."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.fused import ae_ops as fused
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.official_to_baseline import ae_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    ae_prefix,
    ae_weights,
    make_ae_globs,
    make_pack,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_ae_layer() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_ae_globs(pack, layers=1, seed=310)
    layout = globs._pyvm_meta.layout
    weights = ae_weights(globs, 1)
    prefix = ae_prefix(globs, 1)[0]
    x = globs.x_acc[0, 0].float()
    expected_k = torch.zeros_like(globs.k_cache[0, 0])
    expected_v = torch.zeros_like(globs.v_cache[0, 0])
    actual_k = torch.zeros_like(expected_k)
    actual_v = torch.zeros_like(expected_v)
    expected_k[: layout["prefix_len"]], expected_v[: layout["prefix_len"]] = prefix
    actual_k[: layout["prefix_len"]], actual_v[: layout["prefix_len"]] = prefix
    expected = ae_ops.baseline_ae_layer(
        x, weights, 0, globs.rope[0, 0], expected_k, expected_v, layout=layout
    )
    actual = fused.fused_ae_layer(
        x, weights, 0, globs.rope[0, 0], actual_k, actual_v, layout=layout
    )
    gate = CONFIG.gate("S1", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        actual, expected, max_rel=gate.max_rel, abs_floor=gate.abs_floor,
        atol=gate.atol, name="ae_layer",
    )
