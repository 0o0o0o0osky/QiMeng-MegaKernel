"""Pi0 A100 verifier configuration."""

from __future__ import annotations

from pathlib import Path

from qimeng_megakernel.verifier.common.stage_config import (
    ActionGate,
    CopySync,
    CosineGate,
    ElementwiseGate,
    ModelVerifierConfig,
    OracleSpec,
    StageDepthGates,
    TensorGateOverride,
)


_QIMENG_ROOT = Path(__file__).resolve().parents[5]
_SUITE_ROOT = Path(__file__).resolve().parent


def _depths(
    op: tuple[float, float, float],
    layer: tuple[float, float, float],
    module: tuple[float, float, float, float],
) -> StageDepthGates:
    return StageDepthGates(
        op=ElementwiseGate(max_rel=op[0], abs_floor=op[1], atol=op[2]),
        layer=ElementwiseGate(max_rel=layer[0], abs_floor=layer[1], atol=layer[2]),
        module=CosineGate(
            min_cosine=module[0],
            max_abs=module[1],
            norm_ratio_min=module[2],
            norm_ratio_max=module[3],
        ),
    )


CONFIG = ModelVerifierConfig(
    model_key="pi0",
    hardware="A100",
    model_class="VLA",
    dtype="bf16",
    stages=("S0", "S1", "S2", "S3", "S4"),
    disabled_stages={},
    oracle=OracleSpec(
        kind="openpi",
        env=(
            "OPENPI_ROOT",
            "PI0_CKPT",
            "PI0_LIBERO_HDF5",
            "PI0_TOKENIZER",
        ),
    ),
    gates={
        "S0": CosineGate(
            min_cosine=0.998,
            max_abs=0.2,
            norm_ratio_min=0.98,
            norm_ratio_max=1.02,
        ),
        "S1": _depths(
            (0.05, 0.05, 0.005), (0.08, 0.05, 0.01), (0.9999, 0.05, 0.99, 1.01)
        ),
        "S2": _depths(
            (0.125, 0.05, 0.01), (0.25, 0.05, 0.03), (0.999, 1.0, 0.995, 1.005)
        ),
        "S3": StageDepthGates(
            op=ElementwiseGate(max_rel=0.25, abs_floor=0.10, atol=0.025),
            layer=ElementwiseGate(max_rel=0.25, abs_floor=0.05, atol=0.06),
            module=CosineGate(
                min_cosine=0.999,
                max_abs=1.0,
                norm_ratio_min=0.995,
                norm_ratio_max=1.005,
            ),
            tensor_overrides=(
                TensorGateOverride(
                    depth="op",
                    tensor="siglip_folded_projection_landing",
                    gate=ElementwiseGate(max_rel=0.02, abs_floor=0.10, atol=0.001),
                ),
                TensorGateOverride(
                    depth="op",
                    tensor="siglip_attention_landing",
                    gate=ElementwiseGate(max_rel=0.05, abs_floor=0.10, atol=0.01),
                ),
                TensorGateOverride(
                    depth="layer",
                    tensor="siglip",
                    gate=ElementwiseGate(max_rel=0.25, abs_floor=0.25, atol=0.075),
                ),
                TensorGateOverride(
                    depth="layer",
                    tensor="gemma",
                    gate=ElementwiseGate(max_rel=0.25, abs_floor=0.50, atol=0.125),
                ),
            ),
        ),
        "S4": ActionGate(
            action_cosine=0.999,
            action_max_abs=0.06,
            norm_ratio_min=0.98,
            norm_ratio_max=1.02,
            value_range=(-10.0, 10.0),
        ),
    },
    kernel_dirs=CopySync(
        model_side=_QIMENG_ROOT / "models/VLA/pi/pi0/A100/kernel",
        verifier_side=_SUITE_ROOT / "kernel",
    ),
    shapes={
        "views": (1, 2, 3),
        "chunks": (20, 50),
        "action_dim": 32,
    },
    cuda_cleanup_per_test=False,
    notes=(
        "Dedicated S3 op checks cover AE input projection, Gemma RMS "
        "normalization, and SigLIP projector direct leaves, plus the SigLIP "
        "attention-chain/MLP and Gemma QKV-QK/PV-OProj static subgraph "
        "MegaKernel entrypoints at their compiled view specializations; other "
        "direct leaves currently have only layer/module coverage."
    ),
)


__all__ = ["CONFIG"]
