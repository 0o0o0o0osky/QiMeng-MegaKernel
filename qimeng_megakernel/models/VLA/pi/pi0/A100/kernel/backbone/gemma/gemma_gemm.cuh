#pragma once

#include <cuda_bf16.h>

#include "common/gemm_dual.cuh"
#include "common/matmul_pipeline.cuh"

namespace pi0_a100 {

using GemmaGemm64 = mk_gemm::GemmConfig<64, 64, 64, 3, 2, 2, 1>;

// Wide-N FFN tiles reduce activation rereads without changing instruction count.
using GemmaGemmWideN = mk_gemm::GemmConfig<64, 128, 32, 3, 2, 2, 2>;

// The three-view output projection benefits from one additional A/B pipeline
// stage.  Keep this residual-only config separate from GemmaGemmWideN, which is
// also used by the dual FFN and therefore has a different three-ring SMEM cost.
using GemmaOprojWideN = mk_gemm::GemmConfig<64, 128, 32, 4, 2, 2, 2>;

// Columns of C one instruction owns, independent of the tile that covers them.  This is the
// scheduler's `GEMMA_TILE_N` and the two have to agree.
constexpr int kGemmaInstructionCols = 128;

// Each instruction owns 64 output rows and 128 output columns.
template <typename Config, typename Epilogue>
__device__ __forceinline__ void gemma_gemm_64x128(
    const nv_bfloat16* a,
    const nv_bfloat16* b,
    nv_bfloat16* c,
    int m,
    int n,
    int k,
    int logical_m,
    int logical_n,
    Epilogue epi,
    unsigned char* smem,
    int lda = -1,
    int ldb = -1,
    int ldc = -1) {
    constexpr int kSubtiles = kGemmaInstructionCols / Config::BN;
    static_assert(kSubtiles * Config::BN == kGemmaInstructionCols,
        "an instruction's 128 columns must be a whole number of N tiles");
    mk_gemm::gemm_unified_tile<Config>(
        a, b, c, m, n, k,
        logical_m,
        kSubtiles * logical_n,
        kSubtiles * logical_n + kSubtiles,
        epi,
        smem,
        false,
        lda,
        ldb,
        ldc);
}

struct GemmaRowScale {
    const float* factor;
    bool gelu;
    __device__ __forceinline__ float transform(float acc, int row, int) const {
        const float value = acc * factor[row];
        return gelu ? mk_gemm::gelu_tanh(value) : value;
    }
};

struct GemmaRowScaleGate {
    const float* factor;
    const nv_bfloat16* gate;
    int ld;
    __device__ __forceinline__ float transform(float acc, int row, int col) const {
        return acc * factor[row] *
            __bfloat162float(gate[static_cast<int64_t>(row) * ld + col]);
    }
};

struct GemmaFfnMerge {
    const float* factor;

    __device__ __forceinline__ float transform(
        float gate, float up, int row) const {
        const float scale = factor[row];
        return mk_gemm::gelu_tanh(gate * scale) * (up * scale);
    }
};

}  // namespace pi0_a100
