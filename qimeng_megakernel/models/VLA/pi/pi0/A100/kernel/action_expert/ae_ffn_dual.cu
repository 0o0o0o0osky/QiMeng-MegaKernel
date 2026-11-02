#pragma once

#include "action_expert/ae_gemm.cuh"
#include "common/ae_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

__device__ __noinline__ void ae_ffn_dual(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    const int layer = inst[1];
    const int logical_col = inst[3];
    const int split = logical_col / (ae::kKSplit / AeDecoderGemm::BN);
    const int tile = logical_col % (ae::kKSplit / AeDecoderGemm::BN);
    const int64_t weight_offset =
        static_cast<int64_t>(layer) * ae::kMlpWidth * ae::kWidth +
        static_cast<int64_t>(split) * ae::kKSplit * ae::kWidth;
    nv_bfloat16* output =
        g.ffn_buf + static_cast<int64_t>(split) * ae::kRows * ae::kKSplit;
    AeFfnMerge merge{
        g.rms_factor + static_cast<int64_t>(inst[10]) * ae::kRows};
    mk_gemm::EpilogueIdentity epi;
    mk_gemm::gemm_unified_dual_tile<AeDecoderGemm>(
        g.x,
        g.gate_w + weight_offset,
        g.up_w + weight_offset,
        output,
        ae::kRows,
        ae::kKSplit,
        ae::kWidth,
        0,
        tile,
        tile + 1,
        merge,
        epi,
        smem);
}

}  // namespace pi0_a100
