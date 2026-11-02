"""S1 op: baseline and multi-batch fused final RMSNorm plus LM head."""
import pytest
import torch

from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule.fused import operations as fused
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    distinct_prompts_for_batch,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import MULTI_BATCH_CONFIG as CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set MODEL_LLAMA3_2_1B to a local checkpoint directory",
)


@torch.inference_mode()
def test_lm_head_real_data() -> None:
    globs, model = make_globs_from_real_prefill_multi(
        distinct_prompts_for_batch(2), return_model=True
    )
    expected = model.lm_head(model.model.norm(globs.hidden_states)).to(globs.logits.dtype)
    fused.fused_rms_lmhead_gemv(globs)

    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.logits,
        expected,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name="logits",
    )
