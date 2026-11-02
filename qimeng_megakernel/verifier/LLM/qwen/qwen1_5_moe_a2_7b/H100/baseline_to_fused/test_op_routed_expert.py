"""S1 op: baseline and fused routed experts."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.H100.single_batch.decode_schedule.fused.operations import (
    fused_decode_attention,
    fused_oproj_gemv_residual,
    fused_rms_qkvproj_gemv_rope,
    fused_routed_expert_downproj,
    fused_routed_expert_upgate,
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
def test_routed_expert_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True)
    fused_rms_qkvproj_gemv_rope(globs, 0)
    fused_decode_attention(globs, 0)
    fused_oproj_gemv_residual(globs, 0)
    fused_router(globs, 0)

    layer = model.model.layers[0]
    normed = layer.post_attention_layernorm(globs.hidden_states.view(1, 1, -1))
    expected = torch.zeros_like(globs.routed_out_f32)
    globs.routed_out_f32.zero_()
    for slot in range(QwenMoeConfig.num_experts_per_tok):
        expert_id = int(globs.expert_indices[slot].item())
        expert_out = layer.mlp.experts[expert_id](normed)[0, 0]
        expected.add_(
            (
                expert_out
                * globs.expert_weights[slot].to(expert_out.dtype)
            ).float()
        )
        fused_routed_expert_upgate(globs, 0, slot)
        fused_routed_expert_downproj(globs, 0, slot, accum_f32=globs.routed_out_f32)

    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.routed_out_f32, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="routed_out_f32",
    )
