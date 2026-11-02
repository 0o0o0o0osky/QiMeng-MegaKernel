#pragma once

#include <cuda_bf16.h>

#include "common/gemma_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

__device__ __forceinline__ float gemma_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ void gemma_rms(
    const GemmaGlobals& g, const int32_t* inst, int norm_row) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = inst[2];
    const int count = inst[7];
    // Each lane reads eight BF16 values; 2048 columns require eight passes.
    constexpr int kPerLane = 8;
    constexpr int kPerPass = 32 * kPerLane;
    constexpr int kPasses = gemma::kWidth / kPerPass;
    static_assert(gemma::kWidth % kPerPass == 0);
    for (int local = warp; local < count; local += 4) {
        const int row = row0 + local;
        float sumsq = 0.0f;
        const nv_bfloat16* src =
            g.encoder_x + static_cast<int64_t>(row) * gemma::kWidth;
        #pragma unroll
        for (int pass = 0; pass < kPasses; ++pass) {
            const uint4 raw = *reinterpret_cast<const uint4*>(
                src + pass * kPerPass + lane * kPerLane);
            const nv_bfloat16* packed = reinterpret_cast<const nv_bfloat16*>(&raw);
            #pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                const float value = __bfloat162float(packed[j]);
                sumsq = fmaf(value, value, sumsq);
            }
        }
        sumsq = gemma_warp_sum(sumsq);
        if (lane == 0) {
            g.rms_factor[static_cast<int64_t>(norm_row) * gemma::kMaxSeq + row] =
                rsqrtf(sumsq / static_cast<float>(gemma::kWidth) + gemma::kRmsEps);
        }
    }
}

}  // namespace pi0_a100
