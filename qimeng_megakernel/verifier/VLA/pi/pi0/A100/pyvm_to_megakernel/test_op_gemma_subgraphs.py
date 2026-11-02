"""S3 ops: PyVM slices and shape-specialized Gemma static subgraph MegaKernels."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.prefill import build_prefill_schedules
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.fused import gemma_ops
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import (
    instruction_defs as I,
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import GemmaProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    make_gemma_globs,
    make_pack,
    prepare_gemma_native,
    seeded,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate


def _open_first_dependency(instructions: list[object], first_type: type) -> list[object]:
    return [
        replace(item, dep_layer=0, dep_row=0, dep_expected=0)
        if isinstance(item, first_type)
        else item
        for item in instructions
    ]


def _assert_op(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    gate = CONFIG.gate("S3", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None
    assert gate.atol is not None
    assert_max_rel(
        actual,
        expected,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name=name,
    )


def _score_view(tensor: torch.Tensor, seq_pad: int) -> torch.Tensor:
    count = C.num_heads * seq_pad * seq_pad
    return tensor.view(-1)[:count].view(C.num_heads, seq_pad, seq_pad)


def _make_case(views: int, seed: int):
    pack = make_pack(views=views, chunk=20, prompt_len=48)
    prepare_gemma_native(pack, seed=seed)
    globs = make_gemma_globs(pack, layers=1, seed=seed)
    return pack, globs


@pytest.mark.parametrize("views", (2, 3))
def test_gemma_qkv_qk_subgraph(
    verifier_dsos: dict[str, object], views: int
) -> None:
    pack, globs = _make_case(views, seed=630 + views)
    seq = C.encoder_seq_for(views, pack.prompt_len)
    factor = gemma_ops.rms_inv(globs.encoder_x[0, 0, :seq]).squeeze(-1)
    pack.gemma.tensors["rms_factor"][0, 0, 0, :seq].copy_(factor)
    globs.rms_factor[0, 0, 0, :seq].copy_(factor)
    layer, _ = scheduler.schedule_gemma_layer(seq, layer_idx=0, sm_count=1)
    instructions = _open_first_dependency(
        [
            item
            for item in layer
            if isinstance(item, (I.GemmaQkvRope, I.GemmaAttnQk))
        ],
        I.GemmaQkvRope,
    )

    python_vm.run_gemma_pyvm(globs, instructions)
    GemmaProgram(
        pack, build_prefill_schedules(pack), verifier_dsos["gemma"], instructions
    ).run()
    torch.cuda.synchronize(pack.gemma.tensors["encoder_x"].device)
    seq_pad = pack.gemma.tensors["encoder_x"].shape[-2]
    _assert_op(
        pack.gemma.tensors["qkv_buf"][0, 0, :seq].float(),
        globs.qkv_buf[0, 0, :seq].float(),
        name=f"gemma_qkv_qk_qkv_view{views}",
    )
    _assert_op(
        pack.gemma.tensors["k_cache"][0, :seq].float(),
        globs.k_cache[0, :seq].float(),
        name=f"gemma_qkv_qk_k_view{views}",
    )
    _assert_op(
        pack.gemma.tensors["v_cache"][0, :seq].float(),
        globs.v_cache[0, :seq].float(),
        name=f"gemma_qkv_qk_v_view{views}",
    )
    native_scores = _score_view(pack.gemma.tensors["attn_scores"], seq_pad)
    expected_scores = _score_view(globs.attn_scores, seq_pad)
    _assert_op(
        native_scores[:, :seq, :seq],
        expected_scores[:, :seq, :seq],
        name=f"gemma_qkv_qk_scores_view{views}",
    )
    assert torch.isneginf(native_scores[:, :seq, seq:]).all()
    assert torch.isneginf(expected_scores[:, :seq, seq:]).all()
    _assert_op(
        pack.gemma.tensors["attn_vt"][0, 0, :, :seq],
        globs.attn_vt[0, 0, :, :seq],
        name=f"gemma_qkv_qk_vt_view{views}",
    )
    assert torch.count_nonzero(
        pack.gemma.tensors["attn_vt"][0, 0, :, seq:]
    ).item() == 0
    assert torch.count_nonzero(globs.attn_vt[0, 0, :, seq:]).item() == 0


@pytest.mark.parametrize("views", (1, 2, 3))
def test_gemma_pv_oproj_subgraph(
    verifier_dsos: dict[str, object], views: int
) -> None:
    pack, globs = _make_case(views, seed=640 + views)
    seq = C.encoder_seq_for(views, pack.prompt_len)
    seq_pad = pack.gemma.tensors["encoder_x"].shape[-2]
    logical_probabilities = torch.softmax(
        seeded(
            C.num_heads,
            seq,
            seq,
            seed=641 + views,
            dtype=torch.float32,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    probabilities = torch.zeros(
        C.num_heads,
        seq_pad,
        seq_pad,
        device=logical_probabilities.device,
        dtype=torch.bfloat16,
    )
    probabilities[:, :seq, :seq].copy_(logical_probabilities)
    value_t = torch.zeros(
        C.head_dim,
        seq_pad,
        device=logical_probabilities.device,
        dtype=torch.bfloat16,
    )
    value_t[:, :seq].copy_(seeded(C.head_dim, seq, seed=642 + views))
    native_scores = _score_view(pack.gemma.tensors["attn_scores"], seq_pad)
    pyvm_scores = _score_view(globs.attn_scores, seq_pad)
    native_scores.copy_(probabilities)
    pyvm_scores.copy_(probabilities)
    pack.gemma.tensors["attn_vt"][0, 0].copy_(value_t)
    globs.attn_vt[0, 0].copy_(value_t)
    layer, _ = scheduler.schedule_gemma_layer(seq, layer_idx=0, sm_count=1)
    instructions = _open_first_dependency(
        [
            item
            for item in layer
            if isinstance(item, (I.GemmaAttnPv, I.GemmaOproj))
        ],
        I.GemmaAttnPv,
    )

    python_vm.run_gemma_pyvm(globs, instructions)
    GemmaProgram(
        pack, build_prefill_schedules(pack), verifier_dsos["gemma"], instructions
    ).run()
    torch.cuda.synchronize(pack.gemma.tensors["encoder_x"].device)
    _assert_op(
        pack.gemma.tensors["attn_out"][0, 0, :seq].float(),
        globs.attn_out[0, 0, :seq].float(),
        name=f"gemma_pv_oproj_attention_view{views}",
    )
    _assert_op(
        pack.gemma.tensors["encoder_x"][0, 0, :seq].float(),
        globs.encoder_x[0, 0, :seq].float(),
        name=f"gemma_pv_oproj_view{views}",
    )
