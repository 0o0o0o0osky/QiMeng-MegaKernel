"""S3 op: PyVM and megakernel MoE router."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule import (
    mk_kernel,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.verifier_config import (
    CONFIG,
)


def test_op_router(verifier_kernel) -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set QWEN_MOE_A2_7B_MODEL to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 108:
        pytest.skip("Qwen1.5-MoE A100 megakernel requires 108 SMs")

    globs = make_globs_from_real_prefill("Hello", device="cuda:0", max_new_tokens=256)
    layer = scheduler.build_instruction_list(globs, num_layers=1)
    setup = [instruction for instruction in layer if instruction.opcode() < 8]
    instructions = [instruction for instruction in layer if instruction.opcode() in (8, 9)]
    run_pyvm_sequence(globs, setup, python_vm.INSTRUCTION_TO_HANDLER)
    boundary = backup_globs(globs)
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)
    expected = {
        name: getattr(globs, name).clone()
        for name in ("normed_hidden_states", "router_logits_buf", "expert_weights")
    }
    expected_indices = globs.expert_indices.clone()
    expected_barriers = globs.barriers.clone()
    restore_globs(globs, boundary)
    mk_kernel.run_instructions(globs, instructions, mk=verifier_kernel)

    assert torch.equal(globs.expert_indices, expected_indices)
    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    for name, reference in expected.items():
        assert_max_rel(
            getattr(globs, name), reference,
            max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name=name,
        )
    assert torch.equal(globs.barriers, expected_barriers)
