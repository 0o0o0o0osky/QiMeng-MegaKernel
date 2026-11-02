"""S1 op: baseline and fused QKV projection with QK-Norm and RoPE."""
from __future__ import annotations

import pytest

from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    SHORT_PROMPTS,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.baseline_to_fused._baseline_globs import (
    expected_layer0_from_baseline_hf,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.fused.operations import (
    fused_rms_qkvproj_gemv,
    fused_qknorm_rope_writecache,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.verifier_config import CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set QWEN3_TEST_MODEL to a local checkpoint directory",
)

@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_qkv_rope_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True, use_chat_template=False)
    backup = backup_globs(globs)
    baseline_outputs = expected_layer0_from_baseline_hf(globs, model)
    restore_globs(globs, backup)
    layer_idx = 0
    fused_rms_qkvproj_gemv(globs, layer_idx)
    fused_qknorm_rope_writecache(globs, layer_idx)

    k_slice = globs.k_cache[layer_idx, 0, globs.position_id]
    k_baseline = baseline_outputs["k_cache"][layer_idx, 0, globs.position_id]
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    for name, actual, expected in (
        ("q_post_rope", globs.q_post_rope, baseline_outputs["q_post_rope"]),
        ("k_cache", k_slice, k_baseline),
        (
            "v_cache",
            globs.v_cache[layer_idx, 0, globs.position_id],
            baseline_outputs["v_cache"][layer_idx, 0, globs.position_id],
        ),
    ):
        assert_max_rel(
            actual, expected,
            max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
            name=name,
        )
