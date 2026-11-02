"""S2 layer: fused and PyVM one-layer decode."""
import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule import (
    layer_scheduler,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    backup_globs,
    make_globs_from_real_prefill,
    restore_globs,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.verifier_config import CONFIG

pytestmark = pytest.mark.slow


def test_layer_decode() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 132:
        pytest.skip("Llama 3.2 1B H100 schedule requires 132 SMs")

    globs = make_globs_from_real_prefill("Hello", device="cuda:0", max_new_tokens=256)
    source = backup_globs(globs)
    layer_scheduler.run_fused_decode_step(globs, num_layers=1)
    expected = globs.hidden_states.clone()
    restore_globs(globs, source)
    instructions = scheduler.build_instruction_list(globs, num_layers=1)
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)

    gate = CONFIG.gate("S2", "layer")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.hidden_states, expected,
        max_rel=gate.max_rel, abs_floor=gate.abs_floor, atol=gate.atol, name="hidden_states",
    )
