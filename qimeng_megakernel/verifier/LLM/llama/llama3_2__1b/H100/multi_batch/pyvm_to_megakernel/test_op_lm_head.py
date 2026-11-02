"""S3 op: multi-batch megakernel final normalization with LM head."""

import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule import (
    mk_kernel,
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule.instructions import (
    DownProj_GEMM_Residual,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule.model_config import (
    LlamaConfig,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.real_data import backup_all_tensors, restore_globs
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.verifier_config import (
    MULTI_BATCH_CONFIG as CONFIG,
)


@torch.inference_mode()
def test_op_lm_head(verifier_kernel) -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    globs = make_globs_from_real_prefill_multi(
        ["Hello", "The capital of France is"], max_new_tokens=16
    )
    source = backup_all_tensors(globs)
    queues = scheduler.build_runtime_instruction_queues(globs)

    # First run only the 16 transformer layers.  This gives the LM-head oracle
    # the exact hidden state produced by the same verifier kernel, so the op
    # gate does not include accumulated layer error.
    scheduler.tensorize_instructions(globs, [queue[:-1] for queue in queues])
    mk_kernel.call_mk(globs, mk=verifier_kernel)
    globs.barriers[
        LlamaConfig.num_hidden_layers - 1,
        DownProj_GEMM_Residual.opcode() - 1,
        0,
    ] = LlamaConfig.hidden_size // DownProj_GEMM_Residual.tile_N_granularity
    run_pyvm_sequence(
        globs,
        scheduler.schedule_lmhead_gemm(2, 132),
        python_vm.INSTRUCTION_TO_HANDLER,
    )
    expected = globs.logits.clone()

    scheduler.tensorize_instructions(globs, queues)
    restore_globs(globs, source)
    mk_kernel.call_mk(globs, mk=verifier_kernel)

    gate = CONFIG.gate("S3", "op")
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
