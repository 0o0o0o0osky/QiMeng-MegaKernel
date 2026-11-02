#pragma once

#include "backbone/gemma/gemma_gemm.cuh"
#include "common/gemma_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

__device__ __noinline__ void gemma_gate(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    const nv_bfloat16* gate_weight = g.gate_w +
        static_cast<int64_t>(inst[1]) * gemma::kMlpWidth * gemma::kWidth;
    const nv_bfloat16* up_weight = g.up_w +
        static_cast<int64_t>(inst[1]) * gemma::kMlpWidth * gemma::kWidth;
    GemmaFfnMerge merge{
        g.rms_factor + static_cast<int64_t>(inst[7]) * gemma::kMaxSeq};
    mk_gemm::EpilogueIdentity epi;
    using Cfg = GemmaGemmWideN;
    constexpr int kSubtiles = kGemmaInstructionCols / Cfg::BN;
    mk_gemm::gemm_unified_dual_tile<Cfg>(
        g.encoder_x,
        gate_weight,
        up_weight,
        g.up_buf,
        g.seq,
        gemma::kMlpWidth,
        gemma::kWidth,
        inst[2],
        kSubtiles * inst[3],
        kSubtiles * inst[3] + kSubtiles,
        merge,
        epi,
        smem);
}

__device__ __noinline__ void gemma_up_mul(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    // Compatibility publication for the existing opcode-8 schedule.  Opcode 7
    // now matches the H100 route and produces the fused gate/up activation.
    (void)g;
    (void)inst;
    (void)smem;
}

}  // namespace pi0_a100
