from __future__ import annotations

from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.model_config import (
    QwenMoeConfig,
)
from . import operations as fused


def _resolve_layer_count(num_layers: int | None) -> int:
    return (
        num_layers
        if num_layers is not None
        else QwenMoeConfig.num_hidden_layers
    )


def run_fused_decode_layer(globs: Globals, layer_idx: int) -> None:
    """Run one decode layer with prefill-matching stage boundaries."""
    fused.fused_rms_qkvproj_gemv_rope(globs, layer_idx)
    fused.fused_decode_attention(globs, layer_idx)
    fused.fused_oproj_gemv_residual(globs, layer_idx)
    fused.fused_moe_block(globs, layer_idx)


def run_fused_decode_layers_only(
    globs: Globals, num_layers: int | None = None
) -> None:
    layers = _resolve_layer_count(num_layers)
    for layer_idx in range(layers):
        run_fused_decode_layer(globs, layer_idx)


def run_fused_decode_step(
    globs: Globals, num_layers: int | None = None
) -> None:
    run_fused_decode_layers_only(globs, num_layers)
    layers = _resolve_layer_count(num_layers)
    if layers == QwenMoeConfig.num_hidden_layers:
        fused.fused_rms_lmhead_gemv(globs)


def run_fused_prefill_layer(
    globs: Globals,
    layer_idx: int,
    seq_len: int,
    debug_capture_post_attn: set | None = None,
    debug_post_attn_list: list | None = None,
    debug_capture_moe: set | None = None,
    debug_moe_list: list | None = None,
    debug_capture_router: set | None = None,
    debug_router_list: list | None = None,
) -> None:
    """Run one layer of fused prefill: RMS+QKV+RoPE+write cache -> attention -> oproj+residual -> MoE."""
    fused.fused_prefill_rms_qkv_rope_write_cache(globs, layer_idx, seq_len)
    fused.fused_prefill_attention(globs, layer_idx, seq_len)
    fused.fused_prefill_oproj_residual(globs, layer_idx, seq_len)
    if debug_capture_post_attn is not None and debug_post_attn_list is not None and layer_idx in debug_capture_post_attn:
        debug_post_attn_list.append((layer_idx, globs.hidden_states_seq[seq_len - 1].clone().cpu().float()))
    fused.fused_prefill_moe_block(
        globs, layer_idx, seq_len,
        debug_capture_moe=debug_capture_moe,
        debug_moe_list=debug_moe_list,
        debug_capture_router=debug_capture_router,
        debug_router_list=debug_router_list,
    )
