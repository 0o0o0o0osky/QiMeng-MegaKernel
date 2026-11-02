"""S1 module: baseline and multi-batch fused full decode step."""
import pytest
import torch

from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule.fused import layer_scheduler
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.real_data import backup_all_tensors, restore_globs
from qimeng_megakernel.verifier.common.stage_config import CosineGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    distinct_prompts_for_batch,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.single_batch.baseline_to_fused._baseline_globs import (
    run_baseline_decode_step,
    select_batch_row,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.verifier_config import MULTI_BATCH_CONFIG as CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set MODEL_LLAMA3_2_1B to a local checkpoint directory",
)


@torch.inference_mode()
def test_full_decode_real_data() -> None:
    globs, model = make_globs_from_real_prefill_multi(
        distinct_prompts_for_batch(2), return_model=True
    )
    source = backup_all_tensors(globs)
    baseline_rows = [select_batch_row(globs, batch_idx) for batch_idx in range(2)]
    expected_logits = torch.stack(
        [run_baseline_decode_step(row, model) for row in baseline_rows]
    )
    expected_hidden = torch.stack([row.hidden_states for row in baseline_rows])

    restore_globs(globs, source)
    layer_scheduler.run_fused_decode_step(globs)
    assert torch.equal(globs.logits.argmax(dim=-1), expected_logits.argmax(dim=-1))

    gate = CONFIG.gate("S1", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    for name, actual, expected in (
        ("logits", globs.logits, expected_logits),
        ("hidden_states", globs.hidden_states, expected_hidden),
    ):
        assert_close(
            actual,
            expected,
            min_cosine=gate.min_cosine,
            max_abs_tol=gate.max_abs,
            norm_ratio_lo=gate.norm_ratio_min,
            norm_ratio_hi=gate.norm_ratio_max,
            enforce_norm_ratio=True,
            name=name,
        )
