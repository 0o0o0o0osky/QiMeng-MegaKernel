"""S3 module: PyVM and megakernel SigLIP stack."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, launch_native, make_pack, make_siglip_globs, prepare_siglip_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_siglip_module(verifier_kernels: dict[str, object]) -> None:
    views = 2
    pack = make_pack(views=views, chunk=20)
    prepare_siglip_native(pack, seed=560, include_head_tail=True)
    globs, _ = make_siglip_globs(
        views=views, layers=C.vision_layers, seed=560, include_head_tail=True
    )
    instructions = scheduler.schedule_siglip_runtime(
        num_views=views, sm_count=C.sm_count
    )
    python_vm.run_siglip_pyvm(globs, instructions)
    launch_native(verifier_kernels["siglip_view2"], pack.siglip, instructions)
    gate = CONFIG.gate("S3", "module", tensor="siglip")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        pack.siglip.tensors["proj_out"][0, 0], globs.proj_out[0, 0],
        min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="siglip_module",
    )
