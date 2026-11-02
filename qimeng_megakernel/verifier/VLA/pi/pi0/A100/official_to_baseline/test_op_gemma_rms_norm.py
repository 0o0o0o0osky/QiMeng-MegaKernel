"""S0 op: official Gemma RMS normalization and baseline math."""
from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import gemma_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    require_checkpoint,
)


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_gemma_rms_norm() -> None:
    with safe_open(str(require_checkpoint()), framework="pt", device="cpu") as handle:
        x = handle.get_tensor(
            "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight"
        )[None].float()
    expected = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(gemma_ops.rms_norm(x), expected)
