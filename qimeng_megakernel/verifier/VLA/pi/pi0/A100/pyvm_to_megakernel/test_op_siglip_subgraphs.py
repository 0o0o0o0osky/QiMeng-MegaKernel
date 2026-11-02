"""S3 ops: PyVM slices and shape-specialized SigLIP static subgraph MegaKernels."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.prefill import build_prefill_schedules
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import (
    instruction_defs as I,
    python_vm,
    scheduler,
)
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import SiglipProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C,
    make_pack,
    make_siglip_globs,
    prepare_siglip_native,
    siglip_folded_linear_reference,
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


def _assert_final(actual: torch.Tensor, expected: torch.Tensor, *, name: str) -> None:
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


def _assert_bf16_landing(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    tensor: str,
) -> None:
    gate = CONFIG.gate("S3", "op", tensor=tensor)
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


def _make_case(views: int, seed: int, *, norm: str):
    pack = make_pack(views=views, chunk=20)
    prepare_siglip_native(pack, seed=seed, include_head_tail=False)
    globs, _ = make_siglip_globs(views=views, layers=1, seed=seed)

    native_x = pack.siglip.tensors["vision_x"][0, 0, :, : C.vision_width]
    pyvm_x = globs.vision_x[0, 0]
    if not torch.equal(native_x, pyvm_x):
        raise AssertionError("native and PyVM SigLIP boundary activations differ")

    if norm == "attention":
        mean_row, inv_row = (40, 41) if views == 1 else (20, 21)
    elif norm == "mlp":
        mean_row, inv_row = 0, 10
    else:
        raise ValueError(f"unsupported SigLIP boundary norm {norm!r}")

    x = native_x.float()
    mean = x.mean(dim=-1)
    second_moment = (x * x).mean(dim=-1)
    inverse_std = torch.rsqrt(
        (second_moment - mean * mean).clamp_min_(0.0) + C.ln_eps
    )
    stats = pack.siglip.tensors["ln_stats"][0, 0]
    stats[mean_row].copy_(mean)
    stats[inv_row].copy_(inverse_std)
    return pack, globs, native_x.clone(), mean, inverse_std


@pytest.mark.parametrize("views", (1, 2, 3))
def test_siglip_attention_chain_subgraph(
    verifier_dsos: dict[str, object], views: int
) -> None:
    pack, globs, input_x, mean, inverse_std = _make_case(
        views, seed=610 + views, norm="attention"
    )
    layer = scheduler.schedule_siglip_layer(
        layer_idx=0, num_views=views, sm_count=1
    )
    instructions = _open_first_dependency(
        [
            item
            for item in layer
            if isinstance(item, (I.SiglipQkv, I.SiglipAttn, I.SiglipOproj))
        ],
        I.SiglipQkv,
    )

    expected_qkv = siglip_folded_linear_reference(
        pack, input_x, mean, inverse_std, projection="qkv"
    )
    globs.qkv_buf[0, 0].copy_(expected_qkv)
    downstream = _open_first_dependency(
        [item for item in instructions if not isinstance(item, I.SiglipQkv)],
        I.SiglipAttn,
    )
    python_vm.run_siglip_pyvm(globs, downstream)
    SiglipProgram(
        pack, build_prefill_schedules(pack), verifier_dsos["siglip"], instructions
    ).run()
    torch.cuda.synchronize(pack.siglip.tensors["vision_x"].device)
    _assert_bf16_landing(
        pack.siglip.tensors["qkv_buf"][0, 0].float(),
        expected_qkv.float(),
        name=f"siglip_attention_chain_qkv_view{views}",
        tensor="siglip_folded_projection_landing",
    )
    _assert_bf16_landing(
        pack.siglip.tensors["attn_out"][0, 0].float(),
        globs.attn_out[0, 0].float(),
        name=f"siglip_attention_chain_attention_view{views}",
        tensor="siglip_attention_landing",
    )
    _assert_final(
        pack.siglip.tensors["vision_x"][0, 0, :, : C.vision_width].float(),
        globs.vision_x[0, 0].float(),
        name=f"siglip_attention_chain_view{views}",
    )


@pytest.mark.parametrize("views", (1, 2, 3))
def test_siglip_mlp_subgraph(
    verifier_dsos: dict[str, object], views: int
) -> None:
    pack, globs, input_x, mean, inverse_std = _make_case(
        views, seed=620 + views, norm="mlp"
    )
    layer = scheduler.schedule_siglip_layer(
        layer_idx=0, num_views=views, sm_count=1
    )
    instructions = _open_first_dependency(
        [item for item in layer if isinstance(item, (I.SiglipFc1, I.SiglipFc2))],
        I.SiglipFc1,
    )

    expected_fc1 = siglip_folded_linear_reference(
        pack, input_x, mean, inverse_std, projection="fc1"
    )
    globs.fc1_buf[0, 0, :, : C.vision_mlp_width].copy_(expected_fc1)
    downstream = _open_first_dependency(
        [item for item in instructions if not isinstance(item, I.SiglipFc1)],
        I.SiglipFc2,
    )
    python_vm.run_siglip_pyvm(globs, downstream)
    SiglipProgram(
        pack, build_prefill_schedules(pack), verifier_dsos["siglip"], instructions
    ).run()
    torch.cuda.synchronize(pack.siglip.tensors["vision_x"].device)
    _assert_bf16_landing(
        pack.siglip.tensors["fc1_buf"][0, 0, :, : C.vision_mlp_width].float(),
        expected_fc1.float(),
        name=f"siglip_mlp_fc1_view{views}",
        tensor="siglip_folded_projection_landing",
    )
    _assert_final(
        pack.siglip.tensors["vision_x"][0, 0, :, : C.vision_width].float(),
        globs.vision_x[0, 0].float(),
        name=f"siglip_mlp_view{views}",
    )
