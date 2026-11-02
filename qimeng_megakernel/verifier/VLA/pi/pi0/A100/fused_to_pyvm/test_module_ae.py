"""S2 module: fused and PyVM Action Expert stack."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, make_ae_globs, make_pack, run_ae_fused,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_ae_module() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_ae_globs(pack, layers=C.decoder_layers, seed=480)
    expected = run_ae_fused(globs, layers=C.decoder_layers)
    python_vm.run_ae_pyvm(
        globs,
        scheduler.schedule_ae_stack(
            C.decoder_layers, globs._pyvm_meta.layout,
            n_steps=1,
        ),
    )
    gate = CONFIG.gate("S2", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        globs.x_acc[0, 0].float(), expected,
        min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="ae_module",
    )
