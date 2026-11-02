"""S2 op: multi-batch fused and PyVM final normalization with LM head."""

import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule import (
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule.fused import (
    operations as fused,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule.instructions import (
    DownProj_GEMM_Residual,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule.model_config import (
    LlamaConfig,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.real_data import backup_all_tensors, restore_globs
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import (
    MULTI_BATCH_CONFIG as CONFIG,
)

pytestmark = pytest.mark.slow


@torch.inference_mode()
def test_op_lm_head() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    globs = make_globs_from_real_prefill_multi(
        ["Hello", "The capital of France is"], max_new_tokens=16
    )
    globs.barriers[
        LlamaConfig.num_hidden_layers - 1,
        DownProj_GEMM_Residual.opcode() - 1,
        0,
    ] = LlamaConfig.hidden_size // DownProj_GEMM_Residual.tile_N_granularity
    source = backup_all_tensors(globs)
    fused.fused_rms_lmhead_gemv(globs)
    expected = globs.logits.clone()

    restore_globs(globs, source)
    worker_count = scheduler._get_worker_count(globs.hidden_states.device)
    run_pyvm_sequence(
        globs,
        scheduler.schedule_lmhead_rms(2)
        + scheduler.schedule_lmhead_gemm(2, worker_count),
        python_vm.INSTRUCTION_TO_HANDLER,
    )

    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        globs.logits,
        expected,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name="logits",
    )
