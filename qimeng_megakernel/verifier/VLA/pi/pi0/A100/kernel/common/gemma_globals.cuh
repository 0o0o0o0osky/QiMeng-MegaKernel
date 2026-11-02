#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace pi0_a100 {

struct GemmaGlobals {
    nv_bfloat16* encoder_x;
    float* rms_factor;
    nv_bfloat16* qkv_w;
    nv_bfloat16* qkv_buf;
    nv_bfloat16* rope;
    nv_bfloat16* k_cache;
    nv_bfloat16* v_cache;
    // Materialized attention scratch, replacing the split kernel's
    // `attn_partial` / `attn_stats` pair.  `attn_scores` is [kHeads, seq_pad,
    // seq_pad] and carries S on the way into the softmax and P on the way out --
    // the softmax holds a whole row in registers before it writes, so the alias
    // is safe and halves the footprint.  `attn_vt` is [kHeadDim, seq_pad].
    nv_bfloat16* attn_scores;
    nv_bfloat16* attn_vt;
    nv_bfloat16* attn_out;
    nv_bfloat16* oproj_w;
    nv_bfloat16* gate_w;
    nv_bfloat16* up_w;
    nv_bfloat16* gate_buf;
    nv_bfloat16* up_buf;
    nv_bfloat16* down_w;
    int32_t* barriers;
    const int32_t* instructions;
    int depth;
    int seq;
};

}  // namespace pi0_a100
