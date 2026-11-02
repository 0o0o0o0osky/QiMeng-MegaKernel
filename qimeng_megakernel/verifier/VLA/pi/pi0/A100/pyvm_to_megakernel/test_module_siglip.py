"""S3 module: PyVM and the production-form SigLIP stack."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.prefill import build_prefill_schedules
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import SiglipProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, make_pack, make_siglip_globs, prepare_siglip_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_siglip_module(verifier_dsos: dict[str, object]) -> None:
    views = 2
    pack = make_pack(views=views, chunk=20)
    prepare_siglip_native(pack, seed=560, include_head_tail=True)
    globs, _ = make_siglip_globs(
        views=views, layers=C.vision_layers, seed=560, include_head_tail=True
    )
    instructions = scheduler.schedule_siglip_runtime(
        num_views=views
    )
    python_vm.run_siglip_pyvm(globs, instructions)
    SiglipProgram(
        pack, build_prefill_schedules(pack), verifier_dsos["siglip"], instructions
    ).run()
    torch.cuda.synchronize(pack.siglip.tensors["vision_x"].device)
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
