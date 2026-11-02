#pragma once

#include "action_expert/ae_gemm.cuh"
#include "common/ae_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

__device__ __noinline__ void ae_oproj(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    const nv_bfloat16* input =
        g.attn_out + static_cast<int64_t>(inst[2]) * ae::kRows * ae::kKSplit;
    const nv_bfloat16* weight = g.oproj_w +
        static_cast<int64_t>(inst[1]) * ae::kWidth * ae::kKSplit;
    constexpr int64_t kPlane = static_cast<int64_t>(ae::kRows) * ae::kWidth;
    const int group = inst[2] / 2;
    ae_segmented_partial<AeDecoderGemm, 2>(
        input,
        weight,
        reinterpret_cast<float*>(g.attn_partial) + group * kPlane,
        ae::kWidth,
        inst[3],
        smem,
        ae::kRows,
        true);
}

__device__ __noinline__ void ae_down(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    const nv_bfloat16* input =
        g.ffn_buf + static_cast<int64_t>(inst[2]) * ae::kRows * ae::kKSplit;
    const nv_bfloat16* weight = g.down_w +
        static_cast<int64_t>(inst[1]) * ae::kWidth * ae::kKSplit;
    constexpr int64_t kPlane = static_cast<int64_t>(ae::kRows) * ae::kWidth;
    const int group = inst[2] / 4;
    ae_segmented_partial<AeDecoderGemm, 4>(
        input,
        weight,
        reinterpret_cast<float*>(g.attn_partial) + group * kPlane,
        ae::kWidth,
        inst[3],
        smem,
        ae::kRows,
        true);
}

__device__ __noinline__ void ae_residual_reduce(
    const AeGlobals& g, const int32_t* inst, unsigned char*) {
    constexpr int BN = AeDecoderGemm::BN;
    constexpr int64_t kPlane = static_cast<int64_t>(ae::kRows) * ae::kWidth;
    const int tile_n = inst[3];
    const float* scratch = reinterpret_cast<const float*>(g.attn_partial);
    const int count = ae::kRows * BN;
    for (int linear = threadIdx.x; linear < count; linear += blockDim.x) {
        const int row = linear / BN;
        const int col = tile_n * BN + linear % BN;
        const int64_t index = static_cast<int64_t>(row) * ae::kWidth + col;
        const float left = scratch[index] + scratch[kPlane + index];
        const float right =
            scratch[2 * kPlane + index] + scratch[3 * kPlane + index];
        g.x_acc[index] += left + right;
    }
}

}  // namespace pi0_a100
