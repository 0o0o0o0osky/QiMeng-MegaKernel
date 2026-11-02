"""S3 module: PyVM and the production-form Gemma stack."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.prefill import build_prefill_schedules
from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.models.VLA.pi.pi0.A100.subgraphs import GemmaProgram
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, make_gemma_globs, make_pack, prepare_gemma_native,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_gemma_module(verifier_dsos: dict[str, object]) -> None:
    pack = make_pack(views=2, chunk=20)
    prepare_gemma_native(pack, seed=570)
    globs = make_gemma_globs(pack, layers=C.encoder_layers, seed=570)
    seq = globs._pyvm_meta.seq_len
    instructions = scheduler.schedule_gemma_encoder(
        seq, C.encoder_layers, kv_only_last=True
    )
    python_vm.run_gemma_pyvm(globs, instructions)
    GemmaProgram(
        pack, build_prefill_schedules(pack), verifier_dsos["gemma"], instructions
    ).run()
    torch.cuda.synchronize(pack.gemma.tensors["encoder_x"].device)
    gate = CONFIG.gate("S3", "module", tensor="gemma")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    actual = pack.gemma.tensors["encoder_x"][0, 0, :seq].float()
    expected = globs.encoder_x[0, 0, :seq].float()
    actual = actual * pack.gemma.tensors["rms_factor"][0, 0, 0, :seq, None]
    expected = expected * globs.rms_factor[0, 0, 0, :seq, None]
    assert_close(
        actual, expected,
        min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="gemma_module_normalized_hidden",
    )
