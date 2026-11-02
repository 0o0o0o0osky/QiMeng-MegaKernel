"""
Fused decode: full-layer ops (no tiling) and layer scheduler.

- operations: fused_rms_qkvproj_gemv_rope, fused_decode_attention, fused_oproj_gemv_residual,
  fused_mlp_upgate, fused_mlp_down_residual, fused_mlp (= upgate + down_residual), fused_rms_lmhead_gemv.
- layer_scheduler: run_fused_decode_step(globs).

"""
from . import operations, layer_scheduler

__all__ = ["operations", "layer_scheduler"]
