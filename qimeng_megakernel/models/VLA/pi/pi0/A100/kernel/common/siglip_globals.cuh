#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace pi0_a100 {

struct SiglipGlobals {
    nv_bfloat16* vision_x;
    nv_bfloat16* siglip_x_normed;
    nv_bfloat16* ln_gamma;
    nv_bfloat16* ln_beta;
    nv_bfloat16* qkv_w;
    nv_bfloat16* qkv_b;
    nv_bfloat16* qkv_buf;
    nv_bfloat16* attn_out;
    nv_bfloat16* oproj_w;
    nv_bfloat16* oproj_b;
    nv_bfloat16* fc1_w;
    nv_bfloat16* fc1_b;
    nv_bfloat16* fc1_buf;
    nv_bfloat16* fc2_w;
    nv_bfloat16* fc2_b;
    nv_bfloat16* patch_in;
    nv_bfloat16* patch_w;
    nv_bfloat16* patch_b;
    nv_bfloat16* proj_w;
    nv_bfloat16* proj_b;
    nv_bfloat16* proj_out;
    float* ln_stats;
    nv_bfloat16* ln_fold_qkv;
    nv_bfloat16* ln_fold_fc1;
    int32_t* barriers;
    const int32_t* instructions;
    int depth;
    int seq;
};

}  // namespace pi0_a100
