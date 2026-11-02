"""S1 op: baseline and fused MoE router."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.H100.single_batch.decode_schedule.fused.operations import (
    fused_decode_attention,
    fused_oproj_gemv_residual,
    fused_rms_qkvproj_gemv_rope,
    fused_router,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.H100.single_batch.decode_schedule.model_config import (
    QwenMoeConfig,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    SHORT_PROMPTS,
    make_globs_from_real_prefill,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.H100.verifier_config import (
    CONFIG,
)

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set QWEN_MOE_A2_7B_MODEL to a local checkpoint directory",
)


@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_router_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True)
    fused_rms_qkvproj_gemv_rope(globs, 0)
    fused_decode_attention(globs, 0)
    fused_oproj_gemv_residual(globs, 0)
    layer = model.model.layers[0]
    normed = layer.post_attention_layernorm(globs.hidden_states.view(1, 1, -1))[0, 0]
    logits = F.linear(normed.float(), layer.mlp.gate.weight.float())
    probabilities = torch.softmax(logits, dim=-1)
    expected_weights, expected_indices = torch.topk(
        probabilities, QwenMoeConfig.num_experts_per_tok,
    )
    fused_router(globs, 0)

    assert torch.equal(globs.expert_indices, expected_indices)
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.expert_weights, expected_weights,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="expert_weights",
    )
