#pragma once

#include <cuda_bf16.h>

#include "common/pi0_config.cuh"
#include "common/siglip_globals.cuh"

namespace pi0_a100 {

__device__ __forceinline__ float siglip_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __noinline__ void siglip_stats(
    const SiglipGlobals& g, const int32_t* inst) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = inst[2];
    const int count = inst[7];
    // Store the finished mean and inverse standard deviation per row.
    const int mean_row = inst[3];
    const int inv_row = inst[8] ? inst[8] : mean_row + 1;
    // Each lane reads eight BF16 values; the final pass covers the 128-value tail.
    constexpr int kPerLane = 8;
    constexpr int kPerPass = 32 * kPerLane;
    constexpr int kFullPasses = siglip::kWidth / kPerPass;
    constexpr int kTailPerLane = (siglip::kWidth - kFullPasses * kPerPass) / 32;
    for (int local = warp; local < count; local += 4) {
        const int row = row0 + local;
        const nv_bfloat16* src =
            g.vision_x + static_cast<int64_t>(row) * siglip::kWidthPad;
        float sum = 0.0f;
        float sumsq = 0.0f;
        #pragma unroll
        for (int pass = 0; pass < kFullPasses; ++pass) {
            const uint4 raw = *reinterpret_cast<const uint4*>(
                src + pass * kPerPass + lane * kPerLane);
            const nv_bfloat16* packed = reinterpret_cast<const nv_bfloat16*>(&raw);
            #pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                const float value = __bfloat162float(packed[j]);
                sum += value;
                sumsq = fmaf(value, value, sumsq);
            }
        }
        {
            const uint2 raw = *reinterpret_cast<const uint2*>(
                src + kFullPasses * kPerPass + lane * kTailPerLane);
            const nv_bfloat16* packed = reinterpret_cast<const nv_bfloat16*>(&raw);
            #pragma unroll
            for (int j = 0; j < kTailPerLane; ++j) {
                const float value = __bfloat162float(packed[j]);
                sum += value;
                sumsq = fmaf(value, value, sumsq);
            }
        }
        sum = siglip_warp_sum(sum);
        sumsq = siglip_warp_sum(sumsq);
        if (lane == 0) {
            const float mean = sum / static_cast<float>(siglip::kWidth);
            const float variance = fmaxf(
                0.0f, sumsq / static_cast<float>(siglip::kWidth) - mean * mean);
            g.ln_stats[static_cast<int64_t>(mean_row) * g.seq + row] = mean;
            g.ln_stats[static_cast<int64_t>(inv_row) * g.seq + row] =
                rsqrtf(variance + siglip::kLnEps);
        }
    }
}

__device__ __noinline__ void siglip_final_norm(
    const SiglipGlobals& g, const int32_t* inst) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = inst[2];
    const int count = inst[7];
    const int weight_row = inst[3];
    const nv_bfloat16* gamma = g.ln_gamma +
        static_cast<int64_t>(weight_row) * siglip::kWidth;
    const nv_bfloat16* beta = g.ln_beta +
        static_cast<int64_t>(weight_row) * siglip::kWidth;
    // Keep each lane's row fragment in registers across the reduction.
    constexpr int kPerLane = 8;
    constexpr int kPerPass = 32 * kPerLane;
    constexpr int kFullPasses = siglip::kWidth / kPerPass;
    constexpr int kTailPerLane = (siglip::kWidth - kFullPasses * kPerPass) / 32;
    for (int local = warp; local < count; local += 4) {
        const int row = row0 + local;
        const nv_bfloat16* src =
            g.vision_x + static_cast<int64_t>(row) * siglip::kWidthPad;
        nv_bfloat16* dst =
            g.siglip_x_normed + static_cast<int64_t>(row) * siglip::kWidthPad;
        float held[kFullPasses * kPerLane + kTailPerLane];
        float sum = 0.0f;
        float sumsq = 0.0f;
        #pragma unroll
        for (int pass = 0; pass < kFullPasses; ++pass) {
            const uint4 raw = *reinterpret_cast<const uint4*>(
                src + pass * kPerPass + lane * kPerLane);
            const nv_bfloat16* packed = reinterpret_cast<const nv_bfloat16*>(&raw);
            #pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                const float value = __bfloat162float(packed[j]);
                held[pass * kPerLane + j] = value;
                sum += value;
                sumsq = fmaf(value, value, sumsq);
            }
        }
        {
            const uint2 raw = *reinterpret_cast<const uint2*>(
                src + kFullPasses * kPerPass + lane * kTailPerLane);
            const nv_bfloat16* packed = reinterpret_cast<const nv_bfloat16*>(&raw);
            #pragma unroll
            for (int j = 0; j < kTailPerLane; ++j) {
                const float value = __bfloat162float(packed[j]);
                held[kFullPasses * kPerLane + j] = value;
                sum += value;
                sumsq = fmaf(value, value, sumsq);
            }
        }
        sum = siglip_warp_sum(sum);
        sumsq = siglip_warp_sum(sumsq);
        const float mean = __shfl_sync(0xffffffffu, sum, 0) /
            static_cast<float>(siglip::kWidth);
        const float second = __shfl_sync(0xffffffffu, sumsq, 0) /
            static_cast<float>(siglip::kWidth);
        const float inv = rsqrtf(fmaxf(0.0f, second - mean * mean) + siglip::kLnEps);
        #pragma unroll
        for (int pass = 0; pass < kFullPasses; ++pass) {
            const int base = pass * kPerPass + lane * kPerLane;
            alignas(16) nv_bfloat16 out[kPerLane];
            #pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                const float value = (held[pass * kPerLane + j] - mean) * inv;
                out[j] = __float2bfloat16(
                    value * __bfloat162float(gamma[base + j]) +
                    __bfloat162float(beta[base + j]));
            }
            *reinterpret_cast<uint4*>(dst + base) = *reinterpret_cast<const uint4*>(out);
        }
        {
            const int base = kFullPasses * kPerPass + lane * kTailPerLane;
            alignas(8) nv_bfloat16 out[kTailPerLane];
            #pragma unroll
            for (int j = 0; j < kTailPerLane; ++j) {
                const float value = (held[kFullPasses * kPerLane + j] - mean) * inv;
                out[j] = __float2bfloat16(
                    value * __bfloat162float(gamma[base + j]) +
                    __bfloat162float(beta[base + j]));
            }
            *reinterpret_cast<uint2*>(dst + base) = *reinterpret_cast<const uint2*>(out);
        }
        for (int col = siglip::kWidth + lane; col < siglip::kWidthPad; col += 32) {
            dst[col] = __float2bfloat16(0.0f);
        }
    }
}

}  // namespace pi0_a100
