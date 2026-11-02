"""S3 module: PyVM and megakernel Gemma stack."""
from __future__ import annotations

from qimeng_megakernel.models.VLA.pi.pi0.H100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.real_data_utils import (
    C, launch_native, make_gemma_globs, make_pack, prepare_gemma_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.H100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_gemma_module(verifier_kernels: dict[str, object]) -> None:
    pack = make_pack(views=2, chunk=20)
    prepare_gemma_native(pack, seed=570)
    globs = make_gemma_globs(pack, layers=C.encoder_layers, seed=570)
    seq = globs._pyvm_meta.seq_len
    instructions = scheduler.schedule_gemma_encoder(
        seq, C.encoder_layers, kv_only_last=False, sm_count=C.sm_count
    )
    python_vm.run_gemma_pyvm(globs, instructions)
    launch_native(verifier_kernels["gemma"], pack.gemma, instructions)
    gate = CONFIG.gate("S3", "module", tensor="gemma")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    assert_close(
        pack.gemma.tensors["encoder_x"][0, 0, :seq].float(),
        globs.encoder_x[0, 0, :seq].float(),
        min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="gemma_module",
    )
