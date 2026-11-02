#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace pi0_a100 {

struct AeGlobals {
    nv_bfloat16* x;
    float* x_acc;
    float* rms_factor;
    nv_bfloat16* qkv_w;
    nv_bfloat16* qkv_buf;
    nv_bfloat16* rope;
    nv_bfloat16* k_cache;
    nv_bfloat16* v_cache;
    nv_bfloat16* attn_partial;
    float* attn_stats;
    nv_bfloat16* attn_out;
    nv_bfloat16* oproj_w;
    nv_bfloat16* gate_w;
    nv_bfloat16* up_w;
    nv_bfloat16* gate_buf;
    nv_bfloat16* ffn_buf;
    nv_bfloat16* down_w;
    nv_bfloat16* noise;
    nv_bfloat16* act_h;
    nv_bfloat16* act_w_in;
    nv_bfloat16* act_w_mlp;
    nv_bfloat16* act_w_out;
    float* act_bias;
    int32_t* barriers;
    const int32_t* instructions;
    int depth;
    int prefix_len;
    int state_key;
    int valid_rows;
    int kv_rows;
    int num_spans;
};

}  // namespace pi0_a100
