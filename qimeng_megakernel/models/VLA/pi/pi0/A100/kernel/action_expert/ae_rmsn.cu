#pragma once

#include <cuda_bf16.h>

#include "common/ae_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

__device__ __forceinline__ float ae_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Read each row once with float4 vectors and retain it through the reduction.
__device__ __noinline__ void ae_rmsn(
    const AeGlobals& g, const int32_t* inst, int norm_row, unsigned char* smem) {
    constexpr int kPerLane = 4;                     // one float4
    constexpr int kPerPass = kThreads * kPerLane;
    constexpr int kPasses = ae::kWidth / kPerPass;
    static_assert(ae::kWidth % kPerPass == 0);
    float* warp_sums = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = inst[2];
    const int count = inst[8];

    for (int local = 0; local < count; ++local) {
        const int row = row0 + local;
        if (row >= ae::kRows) break;
        const float* src = g.x_acc + static_cast<int64_t>(row) * ae::kWidth;
        float4 held[kPasses];
        #pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            held[p] = *reinterpret_cast<const float4*>(
                src + p * kPerPass + threadIdx.x * kPerLane);
        }
        float sumsq = 0.0f;
        #pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            sumsq = fmaf(held[p].x, held[p].x, sumsq);
            sumsq = fmaf(held[p].y, held[p].y, sumsq);
            sumsq = fmaf(held[p].z, held[p].z, sumsq);
            sumsq = fmaf(held[p].w, held[p].w, sumsq);
        }
        sumsq = ae_warp_sum(sumsq);
        if (lane == 0) warp_sums[warp] = sumsq;
        __syncthreads();
        if (warp == 0) {
            float total = lane < 4 ? warp_sums[lane] : 0.0f;
            total = ae_warp_sum(total);
            if (lane == 0) {
                g.rms_factor[static_cast<int64_t>(norm_row) * ae::kRows + row] =
                    rsqrtf(total / static_cast<float>(ae::kWidth) + ae::kRmsEps);
            }
        }
        __syncthreads();
        nv_bfloat16* dst = g.x + static_cast<int64_t>(row) * ae::kWidth;
        #pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            alignas(16) nv_bfloat16 out[kPerLane];
            out[0] = __float2bfloat16(held[p].x);
            out[1] = __float2bfloat16(held[p].y);
            out[2] = __float2bfloat16(held[p].z);
            out[3] = __float2bfloat16(held[p].w);
            *reinterpret_cast<uint2*>(dst + p * kPerPass + threadIdx.x * kPerLane) =
                *reinterpret_cast<const uint2*>(out);
        }
        __syncthreads();
    }
}

}  // namespace pi0_a100
