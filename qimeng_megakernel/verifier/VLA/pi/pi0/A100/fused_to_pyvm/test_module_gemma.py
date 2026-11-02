"""S2 module: fused and PyVM Gemma stack."""
from __future__ import annotations

import torch

from qimeng_megakernel.models.VLA.pi.pi0.A100.schedule.instructions import python_vm, scheduler
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.real_data_utils import (
    C, gemma_weights, make_gemma_globs, make_pack, run_gemma_fused,
)
from qimeng_megakernel.verifier.VLA.pi.pi0.A100.verifier_config import CONFIG
from qimeng_megakernel.verifier.common.compare import assert_close
from qimeng_megakernel.verifier.common.stage_config import CosineGate


def test_gemma_module() -> None:
    pack = make_pack(views=2, chunk=20)
    globs = make_gemma_globs(pack, layers=C.encoder_layers, seed=470)
    seq = globs._pyvm_meta.seq_len
    expected = run_gemma_fused(
        globs.encoder_x[0, 0, :seq], gemma_weights(globs, C.encoder_layers),
        globs.rope[0, 0, :seq], seq_len=seq,
    )
    python_vm.run_gemma_pyvm(
        globs,
        scheduler.schedule_gemma_encoder(
            seq, C.encoder_layers, kv_only_last=False
        ),
    )
    gate = CONFIG.gate("S2", "module")
    assert isinstance(gate, CosineGate)
    assert gate.min_cosine is not None and gate.max_abs is not None
    assert gate.norm_ratio_min is not None and gate.norm_ratio_max is not None
    actual = globs.encoder_x[0, 0, :seq].float()
    actual = actual * torch.rsqrt(
        actual.square().mean(dim=-1, keepdim=True) + C.rms_eps
    )
    expected = expected.float()
    expected = expected * torch.rsqrt(
        expected.square().mean(dim=-1, keepdim=True) + C.rms_eps
    )
    assert_close(
        actual, expected,
        min_cosine=gate.min_cosine, max_abs_tol=gate.max_abs,
        norm_ratio_lo=gate.norm_ratio_min, norm_ratio_hi=gate.norm_ratio_max,
        enforce_norm_ratio=True, name="gemma_module_normalized_hidden",
    )
