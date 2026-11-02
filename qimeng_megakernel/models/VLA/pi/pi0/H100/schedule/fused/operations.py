"""Re-export fused module ops (LLM ``fused.operations`` analog for multi-module pi0)."""
from __future__ import annotations

from .ae_ops import (
    ae_layout,
    fused_ae_attn,
    fused_ae_layer,
    fused_ae_stack,
    fused_flow_step,
)
from .gemma_ops import (
    apply_rope_interleaved,
    fused_ffn_gate_up_down,
    fused_gemma_layer,
    fused_gemma_stack,
    fused_mqa_bidirectional,
    fused_oproj_residual,
    fused_qkv_rope,
    make_rope_table,
    rms_inv,
)
from .siglip_ops import (
    final_norm,
    fused_gemm,
    fused_mlp_gelu_tanh,
    fused_siglip_attention,
    fused_siglip_attention_head_padded,
    fused_siglip_layer,
    fused_siglip_stack,
    patch_linear_residual,
    projector,
)

__all__ = [
    "fused_siglip_layer",
    "fused_siglip_stack",
    "fused_siglip_attention",
    "fused_siglip_attention_head_padded",
    "fused_gemm",
    "fused_mlp_gelu_tanh",
    "patch_linear_residual",
    "final_norm",
    "projector",
    "fused_gemma_layer",
    "fused_gemma_stack",
    "fused_mqa_bidirectional",
    "fused_qkv_rope",
    "fused_oproj_residual",
    "fused_ffn_gate_up_down",
    "rms_inv",
    "apply_rope_interleaved",
    "make_rope_table",
    "fused_ae_layer",
    "fused_ae_stack",
    "fused_ae_attn",
    "fused_flow_step",
    "ae_layout",
]
