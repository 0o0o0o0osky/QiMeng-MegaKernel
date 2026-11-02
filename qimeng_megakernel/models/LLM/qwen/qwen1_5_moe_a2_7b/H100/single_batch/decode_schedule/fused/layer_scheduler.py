from __future__ import annotations

from ..globs import Globals
from ..model_config import QwenMoeConfig
from . import operations as fused


def _resolve_layer_count(num_layers: int | None) -> int:
    return (
        num_layers
        if num_layers is not None
        else QwenMoeConfig.num_hidden_layers
    )


def run_fused_decode_step(
    globs: Globals, num_layers: int | None = None
) -> None:
    layers = _resolve_layer_count(num_layers)
    for layer_idx in range(layers):
        fused.fused_rms_qkvproj_gemv_rope(globs, layer_idx)
        fused.fused_decode_attention(globs, layer_idx)
        fused.fused_oproj_gemv_residual(globs, layer_idx)
        fused.fused_moe_block(globs, layer_idx)
    if layers == QwenMoeConfig.num_hidden_layers:
        fused.fused_rms_lmhead_gemv(globs)


def run_fused_prefill_layer(
    globs: Globals,
    layer_idx: int,
    seq_len: int,
) -> None:
    """Run one layer of fused prefill: RMS+QKV+RoPE+write cache -> attention -> oproj+residual -> MoE."""
    fused.fused_prefill_rms_qkv_rope_write_cache(globs, layer_idx, seq_len)
    fused.fused_prefill_attention(globs, layer_idx, seq_len)
    fused.fused_prefill_oproj_residual(globs, layer_idx, seq_len)
    fused.fused_prefill_moe_block(globs, layer_idx, seq_len)
