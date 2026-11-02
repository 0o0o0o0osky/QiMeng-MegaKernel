"""
Layer-level scheduler: runs fused ops in order per layer (no instructions).
Used for baseline->fused verification.
Prefill: run_fused_prefill_layer for full-sequence forward per layer (H100 layout).
"""
from __future__ import annotations

from ..globs import Globals
from ..model_config import Qwen2Config
from . import operations as fused


def run_fused_prefill_layer(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """
    Run one transformer layer for prefill: full sequence in one forward (H100 layout).
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
    fused_oproj_gemv_residual -> fused_mlp. Then fused_rms_lmhead_gemv.
    """
    layers = num_layers if num_layers is not None else Qwen2Config.num_hidden_layers
    for layer_idx in range(layers):
        fused.fused_rms_qkvproj_gemv_rope(globs, layer_idx)
        fused.fused_decode_attention(globs, layer_idx)
        fused.fused_oproj_gemv_residual(globs, layer_idx)
        fused.fused_mlp(globs, layer_idx)
    if layers == Qwen2Config.num_hidden_layers:
        fused.fused_rms_lmhead_gemv(globs)
