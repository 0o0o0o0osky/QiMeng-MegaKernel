"""S2 op: fused formula and PyVM MoE combine."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule import (
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    make_globs_from_real_prefill,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen1_5_moe_a2_7b.A100.verifier_config import (
    CONFIG,
)

pytestmark = pytest.mark.slow


def test_op_moe_combine() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set QWEN_MOE_A2_7B_MODEL to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 108:
        pytest.skip("Qwen1.5-MoE A100 schedule requires 108 SMs")

    globs = make_globs_from_real_prefill("Hello", device="cuda:0", max_new_tokens=256)
    layer = scheduler.build_instruction_list(globs, num_layers=1)
    setup = [instruction for instruction in layer if instruction.opcode() < 14]
    instructions = [instruction for instruction in layer if instruction.opcode() == 14]
    run_pyvm_sequence(globs, setup, python_vm.INSTRUCTION_TO_HANDLER)
    normed = globs.normed_hidden_states.reshape(-1).float()
    gate_scale = torch.sigmoid(
        (globs.shared_expert_gate_weights[0].reshape(-1).float() * normed).sum()
    )
    expected = (
        globs.hidden_states.float()
        + globs.routed_out_f32.reshape_as(globs.hidden_states)
        + gate_scale * globs.shared_out.float()
    ).to(globs.hidden_states.dtype)
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)

    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.hidden_states, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="hidden_states",
    )
    assert torch.count_nonzero(globs.routed_out_f32) == 0
