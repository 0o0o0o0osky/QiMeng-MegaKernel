"""S0 op: official Action Expert GELU and baseline math."""
from __future__ import annotations

import os

import pytest
import torch
from safetensors import safe_open

from qimeng_megakernel.verifier.VLA.pi.pi0.H100.official_to_baseline import ae_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
)


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_ae_gelu() -> None:
    with safe_open(os.environ["PI0_CKPT"], framework="pt", device="cpu") as handle:
        x = handle.get_tensor("action_time_mlp_in.bias").float()
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    torch.testing.assert_close(ae_ops.gelu_tanh(x), expected)
