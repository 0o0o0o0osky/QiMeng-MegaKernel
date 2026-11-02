"""Llama 3.2 3B H100 verifier configuration."""
from __future__ import annotations

from pathlib import Path

from qimeng_megakernel.verifier.common.stage_config import (
    CopySync,
    CosineGate,
    ElementwiseGate,
    ModelVerifierConfig,
    OracleSpec,
    StageDepthGates,
    GenerationGate,
)
from qimeng_megakernel.verifier.common.tolerance import (
    E2E_TOP1_MIN_AGREE,
)

_QIMENG_ROOT = Path(__file__).resolve().parents[5]
_SUITE_ROOT = Path(__file__).resolve().parent


CONFIG = ModelVerifierConfig(
    model_key="llama-3.2-3b",
    hardware="H100",
    model_class="LLM",
    dtype="bf16",
    stages=("S1", "S2", "S3", "S4"),
    disabled_stages={
        "S0": "HuggingFace is the baseline, so there is no independent S0 oracle",
    },
    oracle=OracleSpec(kind="hf", env=("LLAMA_32_3B_MODEL_PATH",)),
    gates={
        "S1": StageDepthGates(
            op=ElementwiseGate(max_rel=1e-2, abs_floor=5e-2, atol=5e-3),
            layer=ElementwiseGate(max_rel=1e-2, abs_floor=5e-2, atol=5e-3),
            module=CosineGate(
                min_cosine=0.99,
                max_abs=5.0,
                norm_ratio_min=0.96,
                norm_ratio_max=1.04,
            ),
        ),
        "S2": StageDepthGates(
            op=ElementwiseGate(max_rel=1e-1, abs_floor=5e-2, atol=7e-2),
            layer=ElementwiseGate(max_rel=5e-2, abs_floor=5e-2, atol=1e-2),
            module=CosineGate(
                min_cosine=0.99,
                max_abs=5.0,
                norm_ratio_min=0.96,
                norm_ratio_max=1.04,
            ),
        ),
        "S3": StageDepthGates(
            op=ElementwiseGate(
                max_rel=1e-1,
                abs_floor=5e-2,
                atol=1e-2,
                near_zero=("attn_out_intermediates", "attn_lse_intermediates"),
            ),
            layer=ElementwiseGate(max_rel=2e-2, abs_floor=5e-2, atol=2e-3),
            module=CosineGate(
                min_cosine=0.999,
                max_abs=5.0,
                norm_ratio_min=0.98,
                norm_ratio_max=1.01,
            ),
        ),
        "S4": GenerationGate(
            top1_min=E2E_TOP1_MIN_AGREE,
        ),
    },
    kernel_dirs=CopySync(
        model_side=(
            _QIMENG_ROOT
            / "models/LLM/llama/llama3_2__3b/H100/single_batch/decode_kernel"
        ),
        verifier_side=_SUITE_ROOT / "pyvm_to_megakernel/decode_kernel",
    ),
    cuda_cleanup_per_test=False,
)


__all__ = ["CONFIG"]
