"""S2 module: fused and PyVM full decode step."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule import (
    layer_scheduler,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.qwen.qwen3__0_6b.H100.verifier_config import CONFIG

pytestmark = pytest.mark.slow


def test_module_decode() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set QWEN3_TEST_MODEL to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 132:
        pytest.skip("Qwen3 0.6B H100 schedule requires 132 SMs")

    globs = make_globs_from_real_prefill(
        "Hello", device="cuda:0", max_seq_len=512, use_chat_template=False
    )
    source = backup_globs(globs)
    layer_scheduler.run_fused_decode_step(globs)
    expected = globs.logits.clone()
    restore_globs(globs, source)
    instructions = scheduler.build_instruction_list(globs)
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)

    gate = CONFIG.gate("S2", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        globs.logits, expected,
        min_cosine=gate.min_cosine,
        max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min,
        norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True,
        name="logits",
    )
