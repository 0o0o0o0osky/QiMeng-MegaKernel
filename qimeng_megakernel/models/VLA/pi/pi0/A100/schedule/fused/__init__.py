"""Fused (full-op, untiled) torch references for pi0 A100 modules.

Same role as LLM ``decode_schedule/fused/operations.py``: check that the
math of each logical operation matches the MegaKernel and PyVM paths without tiling.

This package is named ``schedule/fused`` (not ``decode_schedule``) — pi0 is
not an autoregressive decode model; the LLM-style *layout* is kept.
"""
from __future__ import annotations

from . import ae_ops, attention_ops, gemma_ops, siglip_ops
from .operations import *  # noqa: F403

__all__ = [
    "siglip_ops",
    "gemma_ops",
    "ae_ops",
    "attention_ops",
    "fused_siglip_layer",
    "fused_siglip_stack",
    "fused_siglip_attention",
    "fused_gemm",
    "patch_linear_residual",
    "final_norm",
    "projector",
    "fused_gemma_layer",
    "fused_gemma_stack",
    "fused_mqa_bidirectional",
    "rms_inv",
    "fused_ae_layer",
    "fused_ae_stack",
    "fused_ae_attn",
    "fused_ae_rms_linear",
]
