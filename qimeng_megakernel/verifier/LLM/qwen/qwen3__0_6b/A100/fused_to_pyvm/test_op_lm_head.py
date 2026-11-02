"""S2 op: fused and PyVM final normalization with LM head."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule import (
    fused_operations as fused,
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.instructions import (
    BARRIER_ROW,
    NUM_DOWNPROJ_SPLIT_K,
    TILE_N_GRANULARITY,
    DownProj_GEMV_Residual,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.model_config import Qwen3Config
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


def test_op_lm_head() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set QWEN3_TEST_MODEL to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    sm_count = torch.cuda.get_device_properties("cuda:0").multi_processor_count
    if sm_count < 108:
        pytest.skip("Qwen3 0.6B A100 schedule requires 108 SMs")

    globs = make_globs_from_real_prefill(
        "Hello", device="cuda:0", max_seq_len=512, use_chat_template=False
    )
    globs.barriers[
        Qwen3Config.num_hidden_layers - 1,
        BARRIER_ROW(DownProj_GEMV_Residual.opcode()),
        0,
    ] = (
        Qwen3Config.hidden_size // TILE_N_GRANULARITY
    ) * NUM_DOWNPROJ_SPLIT_K
    source = backup_globs(globs)
    fused.fused_rms_lmhead_gemv(globs)
    expected = globs.logits.clone()
    restore_globs(globs, source)
    instructions = scheduler.schedule_lmhead_gemv(min(sm_count, 108))
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)

    gate = CONFIG.gate("S2", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.logits, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="logits",
    )
