"""S2 op: multi-batch fused and PyVM QKV projection with RoPE."""

import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule import (
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule.fused import (
    operations as fused,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.real_data import backup_all_tensors, restore_globs
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.verifier_config import MULTI_BATCH_CONFIG as CONFIG

pytestmark = pytest.mark.slow


@torch.inference_mode()
def test_op_qkv_rope() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    globs = make_globs_from_real_prefill_multi(
        ["Hello", "The capital of France is"], max_new_tokens=16
    )
    source = backup_all_tensors(globs)
    fused.fused_rms_qkvproj_gemv_rope(globs, 0)
    expected_q = globs.q_post_rope.clone()
    expected_k = torch.stack(
        [globs.k_cache[0, b, int(globs.position_ids[b])] for b in range(2)]
    )

    restore_globs(globs, source)
    run_pyvm_sequence(
        globs,
        scheduler.schedule_qkvproj_gemm(0, 2, 132),
        python_vm.INSTRUCTION_TO_HANDLER,
    )
    actual_k = torch.stack(
        [globs.k_cache[0, b, int(globs.position_ids[b])] for b in range(2)]
    )

    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        globs.q_post_rope,
        expected_q,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name="q_post_rope",
    )
    assert_max_rel(
        actual_k,
        expected_k,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name="k_cache_slot",
    )
