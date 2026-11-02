"""S3 op: PyVM and megakernel final normalization with LM head."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule import (
    mk_kernel,
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule.instructions import (
    BARRIER_ROW,
    DownProj_GEMV_Residual,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule.instructions.instruction_defs import (
    DOWNPROJ_BARRIER_COUNT,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule.model_config import LlamaConfig
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__3b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__3b.H100.verifier_config import CONFIG


def test_op_lm_head(verifier_kernel) -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set LLAMA_32_3B_MODEL_PATH to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    sm_count = torch.cuda.get_device_properties("cuda:0").multi_processor_count
    if sm_count < 132:
        pytest.skip("Llama 3.2 3B H100 megakernel requires 132 SMs")

    globs = make_globs_from_real_prefill("Hello", device="cuda:0", max_new_tokens=256)
    instructions = scheduler.schedule_lmhead_gemv(min(sm_count, 132))
    globs.barriers[
        LlamaConfig.num_hidden_layers - 1,
        BARRIER_ROW(DownProj_GEMV_Residual.opcode()),
        0,
    ] = DOWNPROJ_BARRIER_COUNT
    boundary = backup_globs(globs)
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)
    expected = globs.logits.clone()
    expected_barriers = globs.barriers.clone()
    restore_globs(globs, boundary)
    mk_kernel.run_instructions(globs, instructions, mk=verifier_kernel)

    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.logits, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="logits",
    )
    assert torch.equal(globs.barriers, expected_barriers)
