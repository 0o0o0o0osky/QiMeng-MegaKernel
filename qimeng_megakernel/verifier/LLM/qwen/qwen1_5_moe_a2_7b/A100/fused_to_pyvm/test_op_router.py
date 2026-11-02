"""S2 op: fused and PyVM MoE router."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule import (
    fused_operations as fused,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule import (
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

pytestmark = pytest.mark.slow


def test_op_router() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set QWEN_MOE_A2_7B_MODEL to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 108:
        pytest.skip("Qwen1.5-MoE A100 schedule requires 108 SMs")

    globs = make_globs_from_real_prefill("Hello", device="cuda:0", max_new_tokens=256)
    source = backup_globs(globs)
    fused.fused_rms_qkvproj_gemv_rope(globs, 0)
    fused.fused_decode_attention(globs, 0)
    fused.fused_oproj_gemv_residual(globs, 0)
    fused.fused_router(globs, 0)
    expected_indices = globs.expert_indices.clone()
    expected_weights = globs.expert_weights.clone()
    restore_globs(globs, source)
    instructions = [
        instruction for instruction in scheduler.build_instruction_list(globs, num_layers=1)
        if instruction.opcode() <= 9
    ]
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)

    assert torch.equal(globs.expert_indices, expected_indices)
    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.expert_weights, expected_weights,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="expert_weights",
    )
