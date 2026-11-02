"""S1 layer: baseline and fused layer-0 decode."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.fused.operations import (
    fused_decode_attention,
    fused_moe_block,
    fused_oproj_gemv_residual,
    fused_rms_qkvproj_gemv_rope,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.baseline_to_fused._baseline_globs import (
    expected_layer0_full_from_baseline_hf,
)
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
def test_layer_decode(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True)
    source = backup_globs(globs)
    expected = expected_layer0_full_from_baseline_hf(globs, model)["hidden_states"]
    restore_globs(globs, source)
    fused_rms_qkvproj_gemv_rope(globs, 0)
    fused_decode_attention(globs, 0)
    fused_oproj_gemv_residual(globs, 0)
    fused_moe_block(globs, 0)

    gate = CONFIG.gate("S1", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.hidden_states, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="hidden_states",
    )
