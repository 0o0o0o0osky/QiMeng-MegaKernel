"""S3 op: multi-batch PyVM and megakernel QKV projection with RoPE."""

import pytest
import torch

from qimeng_megakernel.models.instruction import run_pyvm_sequence
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule import (
    mk_kernel,
    python_vm,
    scheduler,
)
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.real_data import backup_all_tensors, restore_globs
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import (
    MULTI_BATCH_CONFIG as CONFIG,
)


@torch.inference_mode()
def test_op_qkv_rope(verifier_kernel) -> None:
    if not LOCAL_CHECKPOINT_AVAILABLE:
        pytest.skip("set MODEL_LLAMA3_2_1B to a local checkpoint directory")
    globs = make_globs_from_real_prefill_multi(
        ["Hello", "The capital of France is"], max_new_tokens=16
    )
    source = backup_all_tensors(globs)
    instructions = [
        instruction
        for instruction in scheduler.build_instruction_list(globs, num_layers=1)
        if instruction.opcode() in {1, 2}
    ]
    run_pyvm_sequence(globs, instructions, python_vm.INSTRUCTION_TO_HANDLER)
    expected_q = globs.q_post_rope.clone()
    expected_k = torch.stack(
        [globs.k_cache[0, b, int(globs.position_ids[b])] for b in range(2)]
    )

    restore_globs(globs, source)
    queues = scheduler.build_runtime_instruction_queues(globs, num_layers=1)
    scheduler.tensorize_instructions(
        globs,
        [
            [item if item.opcode() in {1, 2} else scheduler.NoOp() for item in queue]
            for queue in queues
        ],
    )
    mk_kernel.call_mk(globs, mk=verifier_kernel)
    actual_k = torch.stack(
        [globs.k_cache[0, b, int(globs.position_ids[b])] for b in range(2)]
    )

    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    for name, actual, expected in (
        ("q_post_rope", globs.q_post_rope, expected_q),
        ("k_cache_slot", actual_k, expected_k),
    ):
        assert_max_rel(
            actual,
            expected,
            max_rel=gate.max_rel,
            abs_floor=gate.abs_floor,
            atol=gate.atol,
            name=name,
        )
