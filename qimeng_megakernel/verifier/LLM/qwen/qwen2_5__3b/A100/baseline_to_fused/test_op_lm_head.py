"""
S1 op: baseline and fused final RMSNorm plus LM head.
"""
from __future__ import annotations

import pytest

from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    SHORT_PROMPTS,
    make_globs_from_real_prefill,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.A100.baseline_to_fused._baseline_globs import (
    expected_logits_same_formula,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.single_batch.decode_schedule.fused.operations import (
    fused_rms_lmhead_gemv,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.A100.verifier_config import CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set QWEN_TEST_MODEL to a local checkpoint directory",
)

@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_lm_head_real_data(prompt: str) -> None:
    globs = make_globs_from_real_prefill(prompt=prompt)
    baseline_logits = expected_logits_same_formula(globs)["logits"]
    fused_rms_lmhead_gemv(globs)
    fused_logits = globs.logits.clone()
    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        fused_logits, baseline_logits,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="logits",
    )
