"""S1 op: baseline and multi-batch fused QKV projection with RoPE."""
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
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.single_batch.baseline_to_fused._baseline_globs import (
    expected_layer0_from_baseline_hf,
    select_batch_row,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import MULTI_BATCH_CONFIG as CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set MODEL_LLAMA3_2_1B to a local checkpoint directory",
)


@torch.inference_mode()
def test_qkv_rope_layer0_real_data() -> None:
    globs, model = make_globs_from_real_prefill_multi(
        distinct_prompts_for_batch(2), return_model=True
    )
    rows = [select_batch_row(globs, batch_idx) for batch_idx in range(2)]
    expected = [expected_layer0_from_baseline_hf(row, model) for row in rows]
    fused.fused_rms_qkvproj_gemv_rope(globs, 0)

    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    tensors = [
        ("q_post_rope", globs.q_post_rope, torch.stack([item["q_post_rope"] for item in expected])),
        (
            "k_cache",
            torch.stack([globs.k_cache[0, b, int(globs.position_ids[b])] for b in range(2)]),
            torch.stack([item["k_cache"][0, 0, row.position_id] for item, row in zip(expected, rows)]),
        ),
        (
            "v_cache",
            torch.stack([globs.v_cache[0, b, int(globs.position_ids[b])] for b in range(2)]),
            torch.stack([item["v_cache"][0, 0, row.position_id] for item, row in zip(expected, rows)]),
        ),
    ]
    for name, actual, reference in tensors:
        assert_max_rel(
            actual,
            reference,
            max_rel=gate.max_rel,
            abs_floor=gate.abs_floor,
            atol=gate.atol,
            name=name,
        )
