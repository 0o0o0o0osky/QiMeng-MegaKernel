#pragma once

#include "backbone/gemma/gemma_gemm.cuh"
#include "common/gemm_epilogue.cuh"
#include "common/gemma_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

// Three views use a wide-N OProj tile; one and two views use 64 columns.
constexpr int kGemmaWideOprojMinSeq = 768;

__device__ __noinline__ void gemma_oproj(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    const nv_bfloat16* weight = g.oproj_w +
        static_cast<int64_t>(inst[1]) * gemma::kWidth * gemma::kWidth;
    mk_gemm::EpilogueResidual<nv_bfloat16> epi{g.encoder_x, gemma::kWidth};
    if (g.seq >= kGemmaWideOprojMinSeq) {
        gemma_gemm_64x128<GemmaOprojWideN>(
            g.attn_out, weight, g.encoder_x,
            g.seq, gemma::kWidth, gemma::kWidth,
            inst[2], inst[3], epi, smem);
    } else {
        gemma_gemm_64x128<GemmaGemm64>(
            g.attn_out, weight, g.encoder_x,
            g.seq, gemma::kWidth, gemma::kWidth,
            inst[2], inst[3], epi, smem);
    }
}

// Down uses a wide-N tile for two and three views and 64 columns for one view.
constexpr int kGemmaWideDownMinSeq = 512;

__device__ __noinline__ void gemma_down(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    const int layer = inst[1];
    const nv_bfloat16* weight = g.down_w +
        static_cast<int64_t>(layer) * gemma::kWidth * gemma::kMlpWidth;
    mk_gemm::EpilogueResidual<nv_bfloat16> epi{g.encoder_x, gemma::kWidth};
    if (g.seq >= kGemmaWideDownMinSeq) {
        gemma_gemm_64x128<GemmaGemmWideN>(
            g.up_buf, weight, g.encoder_x,
            g.seq, gemma::kWidth, gemma::kMlpWidth,
            inst[2], inst[3], epi, smem);
    } else {
        gemma_gemm_64x128<GemmaGemm64>(
            g.up_buf, weight, g.encoder_x,
            g.seq, gemma::kWidth, gemma::kMlpWidth,
            inst[2], inst[3], epi, smem);
    }
}

}  // namespace pi0_a100
