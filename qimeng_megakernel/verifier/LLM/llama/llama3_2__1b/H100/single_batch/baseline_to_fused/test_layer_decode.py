"""S1 layer: baseline and fused layer-0 decode."""
from __future__ import annotations

import pytest

from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    SHORT_PROMPTS,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.single_batch.baseline_to_fused._baseline_globs import (
    expected_layer0_full_from_baseline_hf,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.fused.operations import (
    fused_rms_qkvproj_gemv_rope,
    fused_decode_attention,
    fused_oproj_gemv_residual,
    fused_mlp,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.verifier_config import CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set MODEL_LLAMA3_2_1B to a local checkpoint directory",
)

@pytest.mark.parametrize("prompt", SHORT_PROMPTS[:1])
def test_single_layer0_real_data(prompt: str) -> None:
    globs, model = make_globs_from_real_prefill(
        prompt=prompt,
        return_model=True,
    )
    backup = backup_globs(globs)
    baseline_outputs = expected_layer0_full_from_baseline_hf(globs, model)
    restore_globs(globs, backup)
    layer_idx = 0
    fused_rms_qkvproj_gemv_rope(globs, layer_idx)
    fused_decode_attention(globs, layer_idx)
    fused_oproj_gemv_residual(globs, layer_idx)
    fused_mlp(globs, layer_idx)

    gate = CONFIG.gate("S1", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.hidden_states, baseline_outputs["hidden_states"],
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol,
        name="hidden_states",
    )
