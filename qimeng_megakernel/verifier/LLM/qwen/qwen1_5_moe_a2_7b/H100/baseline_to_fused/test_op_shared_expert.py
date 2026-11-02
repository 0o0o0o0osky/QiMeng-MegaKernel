"""S1 op: baseline and fused shared expert."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.H100.single_batch.decode_schedule.fused.operations import (
    fused_decode_attention,
    fused_oproj_gemv_residual,
    fused_rms_qkvproj_gemv_rope,
    fused_shared_mlp,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.H100.baseline_to_fused._baseline_globs import (
    expected_layer0_shared_expert_from_baseline_hf,
)
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
def test_shared_expert_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True)
    expected = expected_layer0_shared_expert_from_baseline_hf(globs, model)["shared_out"]
    fused_rms_qkvproj_gemv_rope(globs, 0)
    fused_decode_attention(globs, 0)
    fused_oproj_gemv_residual(globs, 0)
    fused_shared_mlp(globs, 0)

    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.shared_out, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="shared_out",
    )
