"""S2 op: fused and PyVM split-KV attention with reduction."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule import (
    fused_operations as fused,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.A100.verifier_config import CONFIG

pytestmark = pytest.mark.slow


def test_op_attention() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set QWEN3_TEST_MODEL to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 108:
        pytest.skip("Qwen3 0.6B A100 schedule requires 108 SMs")

    globs = make_globs_from_real_prefill(
        "Hello", device="cuda:0", max_seq_len=512, use_chat_template=False
    )
    source = backup_globs(globs)
    fused.fused_rms_qkvproj_gemv(globs, 0)
    fused.fused_qknorm_rope_writecache(globs, 0)
    fused.fused_decode_attention(globs, 0)
    expected = {
        name: getattr(globs, name).clone()
        for name in ("q_post_rope", "k_cache", "v_cache", "attn_out")
    }
    restore_globs(globs, source)
    layer = scheduler.build_instruction_list(globs, num_layers=1)
    instructions = [instruction for instruction in layer if instruction.opcode() <= 3]
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)

    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    for name, reference in expected.items():
        assert_max_rel(
            getattr(globs, name), reference,
            max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name=name,
        )
