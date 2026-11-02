"""S1 op: baseline and fused QKV projection with RoPE."""
from __future__ import annotations

import pytest

from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.fused.operations import (
    fused_rms_qkvproj_gemv_rope,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.baseline_to_fused._baseline_globs import (
    expected_layer0_from_baseline_hf,
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
def test_qkv_rope_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True)
    source = backup_globs(globs)
    expected = expected_layer0_from_baseline_hf(globs, model)
    restore_globs(globs, source)
    fused_rms_qkvproj_gemv_rope(globs, 0)

    position = int(globs.position_id)
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    for name, actual, reference in (
        ("q_post_rope", globs.q_post_rope, expected["q_post_rope"]),
        ("k_cache", globs.k_cache[0, 0, position], expected["k_cache"][0, 0, position]),
        ("v_cache", globs.v_cache[0, 0, position], expected["v_cache"][0, 0, position]),
    ):
        assert_max_rel(
            actual, reference,
            max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name=name,
        )
