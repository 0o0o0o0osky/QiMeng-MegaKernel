"""S2 op: fused and PyVM QKV projection with RoPE."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.A100.single_batch.decode_schedule import (
    fused_operations as fused,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__3b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__3b.A100.verifier_config import CONFIG

pytestmark = pytest.mark.slow


def test_op_qkv_rope() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set LLAMA_32_3B_MODEL_PATH to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 108:
        pytest.skip("Llama 3.2 3B A100 schedule requires 108 SMs")

    globs = make_globs_from_real_prefill("Hello", device="cuda:0", max_new_tokens=256)
    source = backup_globs(globs)
    fused.fused_rms_qkvproj_gemv_rope(globs, 0)
    expected = {
        name: getattr(globs, name).clone()
        for name in ("q_post_rope", "k_cache", "v_cache")
    }
    restore_globs(globs, source)
    instructions = [
        instruction
        for instruction in scheduler.build_instruction_list(globs, num_layers=1)
        if instruction.opcode() == 1
    ]
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)

    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    for name, reference in expected.items():
        assert_max_rel(
            getattr(globs, name), reference,
            max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name=name,
        )
