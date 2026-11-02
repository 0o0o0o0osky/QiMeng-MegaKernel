"""S1 op: baseline and fused decode attention."""
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
    expected_layer0_attention_from_baseline_hf,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.fused.operations import (
    fused_rms_qkvproj_gemv,
    fused_qknorm_rope_writecache,
    fused_decode_attention,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.verifier_config import CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set QWEN3_TEST_MODEL to a local checkpoint directory",
)

@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_attention_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(prompt=prompt, return_model=True, use_chat_template=False)
    backup = backup_globs(globs)
    baseline_attn = expected_layer0_attention_from_baseline_hf(globs, model)
    restore_globs(globs, backup)
    layer_idx = 0
    fused_rms_qkvproj_gemv(globs, layer_idx)
    fused_qknorm_rope_writecache(globs, layer_idx)
    fused_decode_attention(globs, layer_idx)

    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.attn_out, baseline_attn["attn_out"],
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="attn_out",
    )
