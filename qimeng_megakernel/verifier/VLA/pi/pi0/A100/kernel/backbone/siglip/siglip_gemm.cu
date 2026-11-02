#pragma once

#include <cuda_bf16.h>

#include "common/gemm_epilogue.cuh"
#include "common/matmul_pipeline.cuh"
#include "common/pi0_config.cuh"
#include "common/siglip_globals.cuh"

namespace pi0_a100 {

// Per-layer GEMMs use a 64x64 accumulator to avoid the SM80 register cliff.
using SiglipGemm64 = mk_gemm::GemmConfig<64, 64, 64, 3, 2, 2, 1>;

// QKV has enough column work to amortize a 128-column tile.
using SiglipQkvGemm = mk_gemm::GemmConfig<64, 128, 64, 3, 2, 2, 2>;

template <typename Epilogue>
__device__ __forceinline__ void siglip_gemm_128x128(
    const nv_bfloat16* a,
    const nv_bfloat16* b,
    nv_bfloat16* c,
    int m,
    int n,
    int k,
    int logical_m,
    int logical_n,
    Epilogue epi,
    unsigned char* smem) {
    for (int m_sub = 0; m_sub < 2; ++m_sub) {
        mk_gemm::gemm_unified_tile<SiglipGemm64>(
            a, b, c, m, n, k,
            2 * logical_m + m_sub,
            2 * logical_n,
            2 * logical_n + 2,
            epi,
            smem);
    }
}

struct SiglipLnFoldEpilogue {
    const float* stats;
    const nv_bfloat16* folded;
    int seq;
    int folded_rows;
    int layer;
    int mean_row;
    int inv_row;
    int n_offset;
    bool gelu;

    // Statistics are pre-reduced to mean and inverse standard deviation per row.
    __device__ __forceinline__ float transform(float acc, int row, int n) const {
        const int global_n = n_offset + n;
        const float mean = stats[static_cast<int64_t>(mean_row) * seq + row];
        const float inv = stats[static_cast<int64_t>(inv_row) * seq + row];
        const int tile = global_n / 128;
        const int col = global_n % 128;
        const int64_t layer_base = static_cast<int64_t>(layer) * folded_rows * 128;
        const float c = __bfloat162float(folded[layer_base + (2 * tile) * 128 + col]);
        const float b = __bfloat162float(folded[layer_base + (2 * tile + 1) * 128 + col]);
        const float value = inv * (acc - mean * c) + b;
        return gelu ? mk_gemm::gelu_tanh(value) : value;
    }
};

// Fc1 stores each column's bf16 correction/bias pair adjacently.  Reading the pair as one
// bfloat162 halves the fold-table load instructions while preserving the exact same bf16
// values and fp32 arithmetic as SiglipLnFoldEpilogue.
struct SiglipLnFoldPackedEpilogue {
    const float* stats;
    const nv_bfloat16* folded;
    int seq;
    int columns;
    int layer;
    int mean_row;
    int inv_row;
    int n_offset;
    bool gelu;

    // OpenPI's approximate GELU is x * sigmoid(sqrt(8/pi) *
    // x * (1 + 0.044715*x*x)).  Writing it this way avoids the considerably
    // longer tanhf path in the generic epilogue while preserving the same
    // approximation (0.5 * (1 + tanh(z/2)) == sigmoid(z)).  Keep this local
    // to SigLIP: Gemma/AE have different numerical semantics and do not need
    // to inherit this implementation choice.
    __device__ __forceinline__ static float gelu_sigmoid(float x) {
        const float z = 1.5957691216057308f * x * fmaf(0.044715f, x * x, 1.0f);
        return x / (1.0f + __expf(-z));
    }

    __device__ __forceinline__ float transform(float acc, int row, int n) const {
        const int global_n = n_offset + n;
        const float mean = stats[static_cast<int64_t>(mean_row) * seq + row];
        const float inv = stats[static_cast<int64_t>(inv_row) * seq + row];
        const nv_bfloat162 cb = __ldg(
            reinterpret_cast<const nv_bfloat162*>(folded) +
            static_cast<int64_t>(layer) * columns + global_n);
        const float c = __bfloat162float(__low2bfloat16(cb));
        const float b = __bfloat162float(__high2bfloat16(cb));
        const float value = inv * (acc - mean * c) + b;
        return gelu ? gelu_sigmoid(value) : value;
    }
};

__device__ __noinline__ void siglip_qkv(
    const SiglipGlobals& g, const int32_t* inst, unsigned char* smem) {
    const int layer = inst[1];
    SiglipLnFoldPackedEpilogue epi{
        g.ln_stats, g.ln_fold_qkv, g.seq, siglip::kQkvWidth, layer,
        inst[7], inst[7] + 1, 0, false};
    const nv_bfloat16* weight = g.qkv_w +
        static_cast<int64_t>(layer) * siglip::kQkvWidth * siglip::kWidthPad;
    // K is the real width, not the padded one.  `vision_x` and the weight are both padded
    // 1152 -> 1280 on their *trailing* columns and `_fold_siglip_linear` zero-fills the
    // weight there, so the last two K tiles of every instruction multiplied 128 columns of
    // zeros.  Running K = kWidth with the row stride left at kWidthPad skips them; the
    // result is identical by construction.  (Oproj cannot do this: its K is head-padded
    // 72 -> 80 per head, so its zeros are interleaved rather than trailing.)
    mk_gemm::gemm_unified_tile<SiglipQkvGemm>(
        g.vision_x, weight, g.qkv_buf,
        g.seq, siglip::kQkvWidth, siglip::kWidth,
        inst[2], inst[3], inst[3] + 1, epi, smem,
        false, siglip::kWidthPad, siglip::kWidthPad, -1);
}

// Like Fc2, Oproj owns a bare 64x64 tile: at 1 view the 128-row instruction gave
// 40 tiles for 108 CTAs, and its epilogue is a plain bias+residual that does not
// care how the output is decomposed.
__device__ __noinline__ void siglip_oproj(
    const SiglipGlobals& g, const int32_t* inst, unsigned char* smem) {
    const int layer = inst[1];
    const nv_bfloat16* weight = g.oproj_w +
        static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kWidthPad;
    const nv_bfloat16* bias = g.oproj_b +
        static_cast<int64_t>(layer) * siglip::kWidthPad;
    mk_gemm::EpilogueBiasResidual<nv_bfloat16> epi{
        bias, g.vision_x, siglip::kWidthPad};
    mk_gemm::gemm_unified_tile<SiglipGemm64>(
        g.attn_out, weight, g.vision_x,
        g.seq, siglip::kWidthPad, siglip::kWidthPad,
        inst[2], inst[3], inst[3] + 1, epi, smem);
}

// Fc1 writes one [seq, kMlpWidth] buffer.  It used to scatter into six [seq, 768] planes so
// that Fc2 could walk them as K splits; Fc2 is now a single K=4608 GEMM and wants its A
// operand contiguous in K, so the planes are gone.  Nothing else changes: `tile_n` already
// runs 0..71 over the whole padded MLP width, and the old weight offset
// `split*kFc2Segment*kWidthPad + tile_in_split*64*kWidthPad` is identically
// `tile_n*64*kWidthPad`, which is what the GEMM's own `ix` term computes.  The epilogue's
// column offset falls out for the same reason.
__device__ __noinline__ void siglip_fc1(
    const SiglipGlobals& g, const int32_t* inst, unsigned char* smem) {
    const int layer = inst[1];
    const nv_bfloat16* weight = g.fc1_w +
        static_cast<int64_t>(layer) * siglip::kMlpWidth * siglip::kWidthPad;
    SiglipLnFoldPackedEpilogue epi{
        g.ln_stats, g.ln_fold_fc1, g.seq, siglip::kMlpWidth, layer,
        0, 10, 0, true};
    // K = kWidth rather than kWidthPad, for the reason given on Qkv.  Likewise,
    // produce only the logical 4304 MLP columns while retaining the 4608 physical
    // row stride consumed by Fc2.
    mk_gemm::gemm_unified_tile<SiglipGemm64>(
        g.vision_x, weight, g.fc1_buf,
        g.seq, siglip::kMlp, siglip::kWidth,
        inst[2], inst[3], inst[3] + 1, epi, smem,
        false, siglip::kWidthPad, siglip::kWidthPad, siglip::kMlpWidth);
}

// One K=4608 GEMM.  This used to be six sequential K=768 GEMMs accumulating into vision_x
// through a read-modify-write epilogue, which cost five extra bf16 round trips per output
// tile -- the tile was read and written six times to place one number, and the four
// intermediate roundings were pure error.  The splits were never a parallelism device: they
// could not run concurrently on one output tile, so one CTA walked all six anyway.  Fusing
// them keeps the CTA count at the number of output tiles and pays the epilogue once.
//
// The tile stays 64x64 rather than the logical 128x128 used elsewhere because 80 tiles keep
// 80 CTAs busy where 20 kept 20; that reasoning is unchanged.
__device__ __noinline__ void siglip_fc2(
    const SiglipGlobals& g, const int32_t* inst, unsigned char* smem) {
    const int layer = inst[1];
    const nv_bfloat16* bias = g.fc2_b +
        static_cast<int64_t>(layer) * siglip::kWidthPad;
    const nv_bfloat16* weight = g.fc2_w +
        static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kMlpWidth;
    mk_gemm::EpilogueBiasResidual<nv_bfloat16> epi{
        bias, g.vision_x, siglip::kWidthPad};
    // Only the leading 1152 columns are model values.  The trailing 128 columns are
    // storage padding and every consumer (LayerNorm, QKV and Fc1) already reads only
    // kWidth with a kWidthPad row stride.  Keep that stride for B/C, but do not spend two
    // extra N tiles producing padding that is never consumed.
    mk_gemm::gemm_unified_tile<SiglipGemm64>(
        g.fc1_buf, weight, g.vision_x,
        g.seq, siglip::kWidth, siglip::kMlp,
        inst[2], inst[3], inst[3] + 1, epi, smem,
        false, siglip::kMlpWidth, siglip::kMlpWidth, siglip::kWidthPad);
}

__device__ __noinline__ void siglip_patch(
    const SiglipGlobals& g, const int32_t* inst, unsigned char* smem) {
    mk_gemm::EpilogueBiasResidual<nv_bfloat16> epi{
        g.patch_b, g.vision_x, siglip::kWidthPad};
    siglip_gemm_128x128(
        g.patch_in, g.patch_w, g.vision_x,
        g.seq, siglip::kWidthPad, 768,
        inst[2], inst[3], epi, smem);
}

__device__ __noinline__ void siglip_projector(
    const SiglipGlobals& g, const int32_t* inst, unsigned char* smem) {
    mk_gemm::EpilogueBias<nv_bfloat16> epi{g.proj_b};
    siglip_gemm_128x128(
        g.siglip_x_normed, g.proj_w, g.proj_out,
        g.seq, gemma::kWidth, siglip::kWidthPad,
        inst[2], inst[3], epi, smem);
}

}  // namespace pi0_a100
