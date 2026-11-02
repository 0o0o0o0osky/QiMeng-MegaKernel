#pragma once

#include <cuda_bf16.h>

#include "action_expert/ae_gemm.cuh"
#include "common/ae_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

__device__ __noinline__ void ae_qkv_rope(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    const int layer = inst[1];
    const int tile_m = inst[2];
    const int logical_col0 = inst[3] * AeQkvGemm::BN;
    const int query_width = ae::kHeads * ae::kHeadDim;
    const bool query_section = logical_col0 < query_width;
    const int section_col0 = query_section ? logical_col0 : logical_col0 - query_width;
    const int section_width = query_section ? query_width : 2 * ae::kHeadDim;
    const nv_bfloat16* weight = g.qkv_w +
        static_cast<int64_t>(layer) * ae::kQkvWidth * ae::kWidth +
        (query_section ? 0 : static_cast<int64_t>(query_width) * ae::kWidth);

    const float* factor =
        g.rms_factor + static_cast<int64_t>(inst[10]) * ae::kRows;
    nv_bfloat16* section_out = g.qkv_buf + (query_section ? 0 : query_width);
    AeRowScale scale_epilogue{factor};
    mk_gemm::gemm_unified_tile<AeQkvGemm>(
        g.x,
        weight,
        section_out,
        ae::kRows,
        section_width,
        ae::kWidth,
        tile_m,
        section_col0 / AeQkvGemm::BN,
        section_col0 / AeQkvGemm::BN + 1,
        scale_epilogue,
        smem,
        false,
        ae::kWidth,
        ae::kWidth,
        ae::kQkvWidth);
    __syncthreads();

    const int state_key = inst[9] * 64;
    for (int item = threadIdx.x;
         item < AeQkvGemm::BM * (AeQkvGemm::BN / 2);
         item += blockDim.x) {
        const int row = tile_m * AeQkvGemm::BM + item / (AeQkvGemm::BN / 2);
        const int pair = item % (AeQkvGemm::BN / 2);
        const int col = logical_col0 + 2 * pair;
        nv_bfloat16* dst =
            g.qkv_buf + static_cast<int64_t>(row) * ae::kQkvWidth + col;
        float even = __bfloat162float(dst[0]);
        float odd = __bfloat162float(dst[1]);
        if (col < query_width + ae::kHeadDim) {
            const int rope_col = col % ae::kHeadDim;
            const nv_bfloat16* rope =
                g.rope + static_cast<int64_t>(row) * ae::kHeadDim + rope_col;
            const float cosine = __bfloat162float(rope[0]);
            const float sine = __bfloat162float(rope[1]);
            const float rotated_even = even * cosine - odd * sine;
            const float rotated_odd = even * sine + odd * cosine;
            even = rotated_even;
            odd = rotated_odd;
        }
        dst[0] = __float2bfloat16(even);
        dst[1] = __float2bfloat16(odd);
        if (!query_section) {
            const int section = (col - query_width) / ae::kHeadDim;
            const int cache_col = (col - query_width) % ae::kHeadDim;
            nv_bfloat16* cache = (section == 0 ? g.k_cache : g.v_cache) +
                (static_cast<int64_t>(layer) * g.kv_rows + state_key + row) *
                    ae::kHeadDim + cache_col;
            cache[0] = dst[0];
            cache[1] = dst[1];
        }
    }
}

}  // namespace pi0_a100
