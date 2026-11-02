"""S0 op: official SigLIP layer normalization and baseline math."""
from __future__ import annotations

import os

import pytest
import torch
from safetensors import safe_open

from qimeng_megakernel.verifier.VLA.pi.pi0.H100.official_to_baseline import siglip_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_siglip_layernorm() -> None:
    with safe_open(os.environ["PI0_CKPT"], framework="pt", device="cpu") as handle:
        prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
        gamma = handle.get_tensor(
            f"{prefix}.encoder.layers.0.layer_norm1.weight"
        ).float()
        beta = handle.get_tensor(
            f"{prefix}.encoder.layers.0.layer_norm1.bias"
        ).float()
        x = handle.get_tensor(
            f"{prefix}.embeddings.position_embedding.weight"
        )[:8].float()
    expected = torch.nn.functional.layer_norm(
        x, (x.shape[-1],), gamma, beta, 1e-6
    )
    gate = CONFIG.gate("S0")
    torch.testing.assert_close(
        siglip_ops.layernorm(x, gamma, beta), expected, rtol=0, atol=gate.max_abs
    )
