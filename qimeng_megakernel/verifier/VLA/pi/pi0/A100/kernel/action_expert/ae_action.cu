#pragma once

#include <cuda_bf16.h>

#include "action_expert/ae_gemm.cuh"
#include "common/ae_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

__device__ __noinline__ void ae_act_in(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    AeActInEpilogue epi{
        g.act_bias + static_cast<int64_t>(inst[10]) * ae::kWidth};
    mk_gemm::gemm_unified_tile<AeGemm32>(
        g.noise,
        g.act_w_in + static_cast<int64_t>(inst[1]) * ae::kWidth * ae::kKSplit,
        g.act_h,
        ae::kRows,
        ae::kWidth,
        ae::kKSplit,
        inst[2],
        inst[3],
        inst[3] + 1,
        epi,
        smem);
}

__device__ __noinline__ void ae_act_mlp(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    const nv_bfloat16* weight =
        g.act_w_mlp + static_cast<int64_t>(inst[1]) * ae::kWidth * ae::kWidth;
    mk_gemm::gemm_splitk_partial<AeGemm32>(
        g.act_h,
        weight,
        g.x_acc,
        ae::kRows,
        ae::kWidth,
        ae::kWidth,
        inst[2],
        inst[3],
        0,
        ae::kWidth / AeGemm32::BK,
        smem);
    __syncthreads();
    const int col0 = inst[3] * AeGemm32::BN;
    const float* bias =
        g.act_bias + static_cast<int64_t>(inst[10]) * ae::kWidth;
    const float* state = g.act_bias + static_cast<int64_t>(12) * ae::kWidth;
    for (int item = threadIdx.x;
         item < ae::kRows * AeGemm32::BN;
         item += blockDim.x) {
        const int row = item / AeGemm32::BN;
        const int col = col0 + item % AeGemm32::BN;
        float value = g.x_acc[static_cast<int64_t>(row) * ae::kWidth + col] + bias[col];
        if (row == 0) value = state[col];
        if (row >= inst[12]) value = 0.0f;
        g.x_acc[static_cast<int64_t>(row) * ae::kWidth + col] = value;
    }
}

__device__ __noinline__ void ae_act_out(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    // Attention scratch is dead after the final decoder layer.  Reuse its
    // first 64x32 fp32 words so ACT_OUT can preserve the required two BF16
    // landings (delta first, then the noise store-add).
    float* raw = reinterpret_cast<float*>(g.attn_out);
    mk_gemm::gemm_splitk_partial<AeGemm32>(
        g.x,
        g.act_w_out + static_cast<int64_t>(inst[1]) * ae::kActionDim * ae::kWidth,
        raw,
        ae::kRows,
        ae::kActionDim,
        ae::kWidth,
        inst[2],
        inst[3],
        0,
        ae::kWidth / AeGemm32::BK,
        smem);
    __syncthreads();
    const float* bias =
        g.act_bias + static_cast<int64_t>(inst[10]) * ae::kWidth;
    const float* factor =
        g.rms_factor + static_cast<int64_t>(inst[11]) * ae::kRows;
    for (int item = threadIdx.x;
         item < ae::kRows * ae::kActionDim;
         item += blockDim.x) {
        const int row = item / ae::kActionDim;
        const int col = item % ae::kActionDim;
        const nv_bfloat16 delta = __float2bfloat16(raw[item] * factor[row] + bias[col]);
        nv_bfloat16* dst = g.noise + static_cast<int64_t>(row) * ae::kKSplit + col;
        *dst = __float2bfloat16(__bfloat162float(*dst) + __bfloat162float(delta));
    }
}

}  // namespace pi0_a100
