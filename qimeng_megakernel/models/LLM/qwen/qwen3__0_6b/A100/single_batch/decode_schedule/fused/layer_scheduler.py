"""Layer-level scheduler for Qwen3-0.6B decode: runs fused ops per layer."""
from __future__ import annotations

from ..globs import Globals
from ..model_config import Qwen3Config
from . import operations as fused


def run_fused_decode_step(globs: Globals, num_layers: int | None = None) -> None:
    """Run one full decode step using fused Python ops.

    Order per layer: fused_rms_qkvproj_gemv → fused_qknorm_rope_writecache →
    fused_decode_attention → fused_oproj_gemv_residual → fused_mlp. Then fused_rms_lmhead_gemv.
    """
    layers = num_layers if num_layers is not None else Qwen3Config.num_hidden_layers
    for layer_idx in range(layers):
        fused.fused_rms_qkvproj_gemv(globs, layer_idx)
        fused.fused_qknorm_rope_writecache(globs, layer_idx)
        fused.fused_decode_attention(globs, layer_idx)
        fused.fused_oproj_gemv_residual(globs, layer_idx)
        fused.fused_mlp(globs, layer_idx)
    if layers == Qwen3Config.num_hidden_layers:
        fused.fused_rms_lmhead_gemv(globs)


def run_fused_prefill_layer(globs: Globals, layer_idx: int, seq_len: int) -> None:
    """Run one transformer layer for prefill: full sequence in one forward.

    Order: fused_prefill_rms_qkv_qknorm_rope_write_cache -> fused_prefill_attention ->
    fused_prefill_oproj_residual -> fused_prefill_mlp.
    """
    fused.fused_prefill_rms_qkv_qknorm_rope_write_cache(globs, layer_idx, seq_len)
    fused.fused_prefill_attention(globs, layer_idx, seq_len)
    fused.fused_prefill_oproj_residual(globs, layer_idx, seq_len)
    fused.fused_prefill_mlp(globs, layer_idx, seq_len)
