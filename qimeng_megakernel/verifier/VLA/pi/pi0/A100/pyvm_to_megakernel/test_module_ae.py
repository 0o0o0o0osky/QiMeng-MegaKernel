"""S3 module: PyVM and the production-form Action Expert stack."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.decode import build_decode_schedule
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import ActionExpertProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, make_ae_globs, make_pack, prepare_ae_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_ae_module(verifier_dsos: dict[str, object]) -> None:
    pack = make_pack(views=2, chunk=20)
    prepare_ae_native(pack, seed=580)
    globs = make_ae_globs(pack, layers=C.decoder_layers, seed=580)
    instructions = scheduler.schedule_ae_stack(
        C.decoder_layers, pack.ae_layout, n_steps=1
    )
    python_vm.run_ae_pyvm(globs, instructions)
    ActionExpertProgram(
        pack, build_decode_schedule(pack), verifier_dsos["ae"], instructions
    ).run()
    torch.cuda.synchronize(pack.ae.tensors["x"].device)
    gate = CONFIG.gate("S3", "module", tensor="ae")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    query_rows = pack.chunk + 1
    assert_close(
        pack.ae.tensors["x_acc"][0, 0, :query_rows].float(),
        globs.x_acc[0, 0, :query_rows].float(),
        min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="ae_module",
    )
