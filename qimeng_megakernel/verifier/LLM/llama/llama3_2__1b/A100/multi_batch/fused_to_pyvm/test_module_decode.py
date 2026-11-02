"""S2 module: multi-batch fused and PyVM full decode step."""

import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule import (
    layer_scheduler,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.real_data import backup_all_tensors, restore_globs
from qimeng_megakernel.verifier.common.stage_config import CosineGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import (
    MULTI_BATCH_CONFIG as CONFIG,
)

pytestmark = pytest.mark.slow


@torch.inference_mode()
def test_module_decode() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    globs = make_globs_from_real_prefill_multi(
        ["Hello", "The capital of France is"], max_new_tokens=16
    )
    source = backup_all_tensors(globs)
    layer_scheduler.run_fused_decode_step(globs)
    expected = globs.logits.clone()
    expected_tokens = expected.argmax(dim=-1)

    restore_globs(globs, source)
    run_pyvm_sequence(
        globs,
        scheduler.build_instruction_list(globs),
        python_vm.INSTRUCTION_TO_HANDLER,
    )

    assert torch.equal(globs.logits.argmax(dim=-1), expected_tokens)
    gate = CONFIG.gate("S2", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        globs.logits,
        expected,
        min_cosine=gate.min_cosine,
        max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min,
        norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True,
        name="multi_batch_logits",
    )
