"""S1 module: baseline and fused full decode step."""
from __future__ import annotations

import pytest
import torch

from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    SHORT_PROMPTS,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.A100.baseline_to_fused._baseline_globs import (
    run_baseline_decode_step,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.single_batch.decode_schedule.fused.layer_scheduler import (
    run_fused_decode_step,
)
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate
from qimeng_megakernel.verifier.LLM.qwen.qwen2_5__3b.A100.verifier_config import CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set QWEN_TEST_MODEL to a local checkpoint directory",
)

@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_full_decode_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(
        prompt=prompt,
        return_model=True,
    )
    backup = backup_globs(globs)
    baseline_logits = run_baseline_decode_step(globs, model)
    baseline_hidden = globs.hidden_states.clone()
    baseline_token = torch.argmax(baseline_logits).item()
    restore_globs(globs, backup)
    run_fused_decode_step(globs)
    fused_logits = globs.logits.clone()
    fused_hidden = globs.hidden_states.clone()
    fused_token = torch.argmax(fused_logits).item()

    token_match = baseline_token == fused_token
    assert token_match, f"Token mismatch: baseline={baseline_token}, fused={fused_token}"
    gate = CONFIG.gate("S1", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        fused_logits,
        baseline_logits,
        min_cosine=gate.min_cosine,
        max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min,
        norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True,
        name="logits",
    )
    assert_close(
        fused_hidden,
        baseline_hidden,
        min_cosine=gate.min_cosine,
        max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min,
        norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True,
        name="hidden_states",
    )
