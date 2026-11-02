"""S0 op: A100 interleaved RoPE table and official Gemma rotary embedding."""
from __future__ import annotations

import pytest
import torch
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import gemma_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    LOCAL_CHECKPOINT_AVAILABLE,
)


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_gemma_rope_table() -> None:
    config = GemmaConfig(
        head_dim=C.head_dim,
        hidden_size=C.encoder_width,
        intermediate_size=C.encoder_mlp_width,
        num_attention_heads=C.num_heads,
        num_key_value_heads=C.num_kv_heads,
        num_hidden_layers=C.encoder_layers,
        rope_theta=C.rope_theta,
    )
    official = GemmaRotaryEmbedding(config, device="cpu")
    rows = 32
    x = torch.zeros(1, rows, C.head_dim, dtype=torch.bfloat16)
    position_ids = torch.arange(rows, dtype=torch.int64).unsqueeze(0)
    cos, sin = official(x, position_ids)
    expected = torch.empty(rows, C.head_dim, dtype=torch.bfloat16)
    expected[:, 0::2] = cos[0, :, : C.head_dim // 2]
    expected[:, 1::2] = sin[0, :, : C.head_dim // 2]
    torch.testing.assert_close(gemma_ops.make_rope_table(rows), expected, rtol=0, atol=0)

