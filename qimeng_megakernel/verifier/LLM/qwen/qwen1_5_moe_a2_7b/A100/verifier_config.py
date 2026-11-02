"""Qwen1.5 MoE A2.7B A100 verifier configuration."""
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
    model_key="qwen1.5-moe-a2.7b",
    hardware="A100",
    model_class="LLM",
    dtype="bf16",
    stages=("S1", "S2", "S3", "S4"),
    disabled_stages={
        "S0": "HuggingFace is the baseline, so there is no independent S0 oracle",
    },
    oracle=OracleSpec(kind="hf", env=("QWEN_MOE_A2_7B_MODEL",)),
    gates={
        "S1": StageDepthGates(
            op=ElementwiseGate(max_rel=5e-2, abs_floor=5e-2, atol=5e-3),
            layer=ElementwiseGate(max_rel=2e-2, abs_floor=5e-2, atol=5e-3),
            module=CosineGate(
                min_cosine=0.99,
                max_abs=3.5,
                norm_ratio_min=0.96,
                norm_ratio_max=1.04,
            ),
        ),
        "S2": StageDepthGates(
            op=ElementwiseGate(max_rel=2e-2, abs_floor=5e-2, atol=5e-3),
            layer=ElementwiseGate(max_rel=1e-2, abs_floor=5e-2, atol=5e-3),
            module=CosineGate(
                min_cosine=0.9995,
                max_abs=0.35,
                norm_ratio_min=0.98,
                norm_ratio_max=1.02,
            ),
        ),
        "S3": StageDepthGates(
            op=ElementwiseGate(max_rel=4e-1, abs_floor=5e-2, atol=3.2e-2),
            layer=ElementwiseGate(max_rel=1.25e-1, abs_floor=5e-2, atol=8e-3),
            module=CosineGate(
                min_cosine=0.9998,
                max_abs=3e-1,
                norm_ratio_min=0.99,
                norm_ratio_max=1.006,
            ),
        ),
        "S4": GenerationGate(
            top1_min=E2E_TOP1_MIN_AGREE,
            repetition_policy="allow_baseline_shared",
            reason="Allow repetition only when the same checkpoint baseline also repeats",
        ),
    },
    kernel_dirs=CopySync(
        model_side=(
            _QIMENG_ROOT
            / "models/LLM/qwen/qwen1_5_moe_a2_7b/A100/single_batch/decode_kernel"
        ),
        verifier_side=_SUITE_ROOT / "pyvm_to_megakernel/decode_kernel",
    ),
    cuda_cleanup_per_test=False,
)


__all__ = ["CONFIG"]
