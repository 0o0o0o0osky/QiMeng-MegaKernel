"""
Layer-level scheduler: runs fused ops in order per layer (no instructions).

Used for two-level verification:
  1. HF -> fused: compare HF forward vs run_fused_decode_step(globs).
  2. Fused -> instruction schedule: compare run_fused_decode_step(globs) vs
     run_pyvm_sequence(globs, instructions, handler_map).

This module does not emit or tensorize instructions; it only invokes fused_* in
the same order as the instruction schedule (per-layer: QKV+RoPE, attention, o_proj, mlp;
then lm_head).
"""

from __future__ import annotations

from ..globs import Globals
from ..model_config import LlamaConfig
from . import operations as fused


def run_fused_prefill_layer(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """
    Run one transformer layer for prefill: full sequence in one forward.
    Order: fused_prefill_rms_qkv_rope_write_cache -> fused_prefill_attention ->
    fused_prefill_oproj_residual -> fused_prefill_mlp.
    """
    fused.fused_prefill_rms_qkv_rope_write_cache(globs, layer_idx, seq_len)
    fused.fused_prefill_attention(globs, layer_idx, seq_len)
    fused.fused_prefill_oproj_residual(globs, layer_idx, seq_len)
    fused.fused_prefill_mlp(globs, layer_idx, seq_len)


def run_fused_decode_step(globs: Globals, num_layers: int | None = None) -> None:
    """
    Run one decode step using fused (full-layer) ops only.

    Order per layer: fused_rms_qkvproj_gemv_rope -> fused_decode_attention ->
    fused_oproj_gemv_residual -> fused_mlp. If num_layers is None or equals
    LlamaConfig.num_hidden_layers, append fused_rms_lmhead_gemv after the last layer.
    """
    layers = num_layers if num_layers is not None else LlamaConfig.num_hidden_layers
    for layer_idx in range(layers):
        fused.fused_rms_qkvproj_gemv_rope(globs, layer_idx)
        fused.fused_decode_attention(globs, layer_idx)
        fused.fused_oproj_gemv_residual(globs, layer_idx)
        fused.fused_mlp(globs, layer_idx)
    if layers == LlamaConfig.num_hidden_layers:
        fused.fused_rms_lmhead_gemv(globs)
