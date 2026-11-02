"""S1 op: baseline and fused attention output projection."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.fused.operations import (
    fused_decode_attention,
    fused_oproj_gemv_residual,
    fused_rms_qkvproj_gemv_rope,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    SHORT_PROMPTS,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.verifier_config import (
    CONFIG,
)

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set QWEN_MOE_A2_7B_MODEL to a local checkpoint directory",
)


@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_o_proj_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True)
    fused_rms_qkvproj_gemv_rope(globs, 0)
    fused_decode_attention(globs, 0)
    source = backup_globs(globs)
    fused_oproj_gemv_residual(globs, 0)
    actual = globs.hidden_states.clone()
    restore_globs(globs, source)
    weight = model.model.layers[0].self_attn.o_proj.weight
    projected = torch.nn.functional.linear(globs.attn_out.unsqueeze(0), weight).squeeze(0)
    expected = globs.hidden_states + projected.to(globs.hidden_states.dtype)

    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        actual, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="hidden_states",
    )
