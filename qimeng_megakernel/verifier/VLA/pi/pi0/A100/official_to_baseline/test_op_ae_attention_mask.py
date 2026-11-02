"""S0 op: Action Expert prefix/self visibility matches official suffix masking."""
from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from qimeng_megakernel.verifier.VLA.pi.pi0.A100.official_to_baseline import ae_ops
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    require_checkpoint,
)


pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set PI0_CKPT to the local Pi0 safetensors checkpoint",
)


def test_ae_state_row_cannot_see_action_keys() -> None:
    heads, head_dim, rows, keys = 2, 4, 4, 8
    with safe_open(str(require_checkpoint()), framework="pt", device="cpu") as handle:
        q_all = handle.get_tensor("action_in_proj.weight")[: heads * head_dim, :rows].T
        source = handle.get_tensor("action_out_proj.weight")[:head_dim, : 2 * keys].T
    k = source[:keys].contiguous()
    v = source[keys:].contiguous()
    prefix_len, state_key, self_end = 3, 4, 8

    actual = ae_ops.baseline_ae_attn(
        q_all,
        k,
        v,
        prefix_len=prefix_len,
        state_key=state_key,
        self_end=self_end,
        m_pad=rows,
        num_heads=heads,
        head_dim=head_dim,
    )

    key_index = torch.arange(keys)
    visible = (key_index < prefix_len) | (
        (key_index >= state_key) & (key_index < self_end)
    )
    mask = visible.unsqueeze(0).expand(rows, -1).clone()
    mask[0] = (key_index < prefix_len) | (key_index == state_key)
    q = q_all.float().reshape(rows, heads, head_dim).permute(1, 0, 2).unsqueeze(0)
    kh = k.float().expand(heads, -1, -1).unsqueeze(0)
    vh = v.float().expand(heads, -1, -1).unsqueeze(0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        kh,
        vh,
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
    ).squeeze(0).permute(1, 0, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    unmasked_state = ae_ops.baseline_ae_attn(
        q_all,
        k,
        v,
        prefix_len=prefix_len,
        state_key=state_key,
        self_end=self_end,
        m_pad=rows,
        mask_state_row=False,
        num_heads=heads,
        head_dim=head_dim,
    )[0]
    assert not torch.allclose(actual[0], unmasked_state)

