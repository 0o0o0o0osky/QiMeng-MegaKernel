"""S3 op: PyVM and megakernel attention output projection."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule import (
    mk_kernel,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.H100.verifier_config import CONFIG


def test_op_o_proj(verifier_kernel) -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set QWEN3_TEST_MODEL to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 132:
        pytest.skip("Qwen3 0.6B H100 megakernel requires 132 SMs")

    globs = make_globs_from_real_prefill(
        "Hello",
        device="cuda:0",
        max_seq_len=512,
        use_chat_template=False,
        for_mk=True,
    )
    layer = scheduler.build_instruction_list(globs, num_layers=1)
    setup = [instruction for instruction in layer if instruction.opcode() < 4]
    instructions = [instruction for instruction in layer if instruction.opcode() == 4]
    run_pyvm_sequence(globs, setup, python_vm.INSTRUCTION_TO_HANDLER)
    boundary = backup_globs(globs)
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)
    expected = globs.hidden_states.clone()
    expected_barriers = globs.barriers.clone()
    restore_globs(globs, boundary)
    mk_kernel.run_instructions(globs, instructions, mk=verifier_kernel)

    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.hidden_states, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="hidden_states",
    )
    assert torch.equal(globs.barriers, expected_barriers)
