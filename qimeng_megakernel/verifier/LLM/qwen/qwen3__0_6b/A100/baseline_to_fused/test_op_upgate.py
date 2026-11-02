"""
S1 op: baseline and fused UpGate projection with SiLU.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    SHORT_PROMPTS,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.fused.operations import (
    fused_rms_qkvproj_gemv,
    fused_qknorm_rope_writecache,
    fused_decode_attention,
    fused_oproj_gemv_residual,
    fused_mlp_upgate,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.verifier_config import CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set QWEN3_TEST_MODEL to a local checkpoint directory",
)

@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_upgate_silu_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(
        prompt=prompt,
        return_model=True,
        use_chat_template=False,
    )
    layer_idx = 0
    fused_rms_qkvproj_gemv(globs, layer_idx)
    fused_qknorm_rope_writecache(globs, layer_idx)
    fused_decode_attention(globs, layer_idx)
    fused_oproj_gemv_residual(globs, layer_idx)
    backup = backup_globs(globs)
    fused_mlp_upgate(globs, layer_idx)
    fused_silu = globs.silu_out.clone()
    restore_globs(globs, backup)
    layer = model.model.layers[layer_idx]
    hidden_states = globs.hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states.unsqueeze(0)).squeeze(0)
    gate = F.linear(hidden_states.unsqueeze(0), layer.mlp.gate_proj.weight).squeeze(0)
    up = F.linear(hidden_states.unsqueeze(0), layer.mlp.up_proj.weight).squeeze(0)
    baseline_silu = F.silu(gate) * up
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        fused_silu, baseline_silu,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="silu_out",
    )
