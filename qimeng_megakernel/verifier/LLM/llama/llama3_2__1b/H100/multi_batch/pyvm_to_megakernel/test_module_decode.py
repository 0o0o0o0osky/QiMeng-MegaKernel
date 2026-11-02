"""S3 module: multi-batch PyVM and megakernel full decode step."""

import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule import (
    mk_kernel,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.real_data import backup_all_tensors, restore_globs
from qimeng_megakernel.verifier.common.stage_config import CosineGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    distinct_prompts_for_batch,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.H100.verifier_config import (
    MULTI_BATCH_CONFIG as CONFIG,
)

pytestmark = pytest.mark.slow


def _require_h100() -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_properties("cuda:0").multi_processor_count < 132:
        pytest.skip("Llama 3.2 1B H100 megakernel requires 132 SMs")


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    gate = CONFIG.gate("S3", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        actual,
        expected,
        min_cosine=gate.min_cosine,
        max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min,
        norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True,
        name="multi_batch_logits",
    )


@pytest.mark.parametrize("batch_size", (2, 16))
@torch.inference_mode()
def test_module_decode_and_replay(batch_size: int, verifier_kernel) -> None:
    _require_h100()
    globs = make_globs_from_real_prefill_multi(
        distinct_prompts_for_batch(batch_size), max_new_tokens=16
    )
    source = backup_all_tensors(globs)

    run_pyvm_sequence(
        globs,
        scheduler.build_instruction_list(globs),
        python_vm.INSTRUCTION_TO_HANDLER,
    )
    expected = globs.logits.clone()
    expected_tokens = expected.argmax(dim=-1)

    restore_globs(globs, source)
    mk_kernel.call_mk(globs, mk=verifier_kernel)
    actual = globs.logits.clone()
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.argmax(dim=-1), expected_tokens)
    _assert_close(actual, expected)

    restore_globs(globs, source)
    mk_kernel.call_mk(globs, mk=verifier_kernel)
    replay = globs.logits.clone()
    restore_globs(globs, source)
    mk_kernel.call_mk(globs, mk=verifier_kernel)
    assert torch.equal(globs.logits, replay)


@torch.inference_mode()
def test_ragged_inactive_rows(verifier_kernel) -> None:
    _require_h100()
    batch_size = 4
    globs = make_globs_from_real_prefill_multi(
        distinct_prompts_for_batch(batch_size), max_new_tokens=16
    )
    globs.active_mask.copy_(
        torch.tensor([True, False, True, False], device=globs.active_mask.device)
    )
    for name in (
        "hidden_states_backing",
        "q_post_rope_backing",
        "attn_out_backing",
        "silu_out_backing",
        "logits_backing",
        "attn_rms_out_backing",
        "mlp_rms_out_backing",
        "lm_head_rms_out_backing",
    ):
        getattr(globs, name)[batch_size:].fill_(float("nan"))
    source = backup_all_tensors(globs)
    initial_hidden = globs.hidden_states.clone()
    initial_barriers = globs.barriers.clone()

    run_pyvm_sequence(
        globs,
        scheduler.build_instruction_list(globs),
        python_vm.INSTRUCTION_TO_HANDLER,
    )
    expected = globs.logits.clone()

    restore_globs(globs, source)
    mk_kernel.call_mk(globs, mk=verifier_kernel)
    active = globs.active_mask.bool()
    assert torch.isfinite(globs.logits).all()
    _assert_close(globs.logits[active], expected[active])
    assert torch.equal(globs.logits[~active], expected[~active])
    assert torch.equal(globs.hidden_states[~active], initial_hidden[~active])
    assert torch.isnan(globs.hidden_states_backing[batch_size:]).all()
    assert torch.equal(globs.barriers, initial_barriers)
