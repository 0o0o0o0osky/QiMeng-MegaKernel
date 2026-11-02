#pragma once

#include <cuda_bf16.h>

#include "backbone/gemma/gemma_gemm.cuh"
#include "common/gemma_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

// repN=2 makes the unified C epilogue's physical S2G vector eight BF16 values
// (one aligned uint128 transaction) for both qkv_buf and the optional cache
// mirror.  The MMA still accumulates every output through the same K sequence.
using GemmaQkvGemm = mk_gemm::GemmConfig<64, 64, 64, 3, 2, 2, 2>;
static_assert(GemmaQkvGemm::kPhysicalS2gVec == 8);

struct GemmaQkvFragmentEpilogue {
    static constexpr bool kUseFragmentTransform = true;
    static constexpr bool kUseFragmentMirror = true;

    const float* factor;
    const nv_bfloat16* rope;
    nv_bfloat16* k_cache;
    nv_bfloat16* v_cache;
    int layer;
    int output_col_offset;

    template <typename Accumulator, typename Coordinates>
    __device__ __forceinline__ void transform_fragment(
        Accumulator& accumulator,
        const Coordinates& coordinates,
        int row_base,
        int section_col_base,
        int rows,
        int section_width) const {
        // Adjacent even/odd columns share a lane. Keep both in FP32 through RoPE.
        // Each lane owns four fragment rows, repeated across N.  Cache each
        // row factor once in registers instead of reloading it for every pair.
        float row_factors[4];
        CUTE_UNROLL
        for (int slot = 0; slot < 4; ++slot) {
            const int row = row_base + cute::get<0>(coordinates(2 * slot));
            row_factors[slot] = row < rows ? factor[row] : 0.0f;
        }
        CUTE_UNROLL
        for (int e = 0; e < cute::size(accumulator); e += 2) {
            auto even_coord = coordinates(e);
            const int row = row_base + cute::get<0>(even_coord);
            const int section_col = section_col_base + cute::get<1>(even_coord);
            if (row >= rows || section_col + 1 >= section_width) continue;

            const float row_factor = row_factors[(e / 2) % 4];
            float even = float(accumulator(e)) * row_factor;
            float odd = float(accumulator(e + 1)) * row_factor;
            const int output_col = output_col_offset + section_col;
            if (output_col < gemma::kHeads * gemma::kHeadDim + gemma::kHeadDim) {
                const int rope_col = output_col % gemma::kHeadDim;
                const nv_bfloat16* row_rope =
                    rope + static_cast<int64_t>(row) * gemma::kHeadDim + rope_col;
                // row_rope is 4-byte aligned: rope rows contain 256 BF16
                // values and every pair begins at an even column.
                const nv_bfloat162 rope_pair =
                    *reinterpret_cast<const nv_bfloat162*>(row_rope);
                const float2 rope_f32 = __bfloat1622float2(rope_pair);
                const float cosine = rope_f32.x;
                const float sine = rope_f32.y;
                const float rotated_even = even * cosine - odd * sine;
                const float rotated_odd = even * sine + odd * cosine;
                even = rotated_even;
                odd = rotated_odd;
            }
            accumulator(e) = even;
            accumulator(e + 1) = odd;
        }
    }

    __device__ __forceinline__ nv_bfloat16* mirror_pointer(
        int row, int section_col) const {
        if (output_col_offset == 0) return nullptr;  // Q has no cache mirror.
        const int section = section_col / gemma::kHeadDim;
        // section_col is the base of an eight-BF16 S2G vector.  Cache rows are
        // 256 BF16 wide, so this pointer is always 16-byte aligned.
        return (section == 0 ? k_cache : v_cache) +
            (static_cast<int64_t>(layer) * gemma::kMaxSeq + row) * gemma::kHeadDim +
            section_col % gemma::kHeadDim;
    }
};

__device__ __noinline__ void gemma_qkv_rope(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    // One instruction owns one physical QKV N tile.
    constexpr int kColsPerInstruction = GemmaQkvGemm::BN;
    const int layer = inst[1];
    const int logical_col0 = inst[3] * kColsPerInstruction;
    const bool query_section = logical_col0 < gemma::kHeads * gemma::kHeadDim;
    const int section_col0 = query_section
        ? logical_col0
        : logical_col0 - gemma::kHeads * gemma::kHeadDim;
    const int section_width = query_section
        ? gemma::kHeads * gemma::kHeadDim
        : 2 * gemma::kHeadDim;
    const nv_bfloat16* weight = g.qkv_w +
        static_cast<int64_t>(layer) * gemma::kQkvWidth * gemma::kWidth +
        (query_section ? 0 : static_cast<int64_t>(gemma::kHeads * gemma::kHeadDim) * gemma::kWidth);
    const int row_tile = inst[2];
    const int col_tile = section_col0 / GemmaQkvGemm::BN;
    const float* factor =
        g.rms_factor + static_cast<int64_t>(inst[7]) * gemma::kMaxSeq;
    GemmaQkvFragmentEpilogue epilogue{
        factor,
        g.rope,
        g.k_cache,
        g.v_cache,
        layer,
        query_section ? 0 : gemma::kHeads * gemma::kHeadDim,
    };

    // The accumulator remains FP32 through row scale and Q/K RoPE.  The
    // unified epilogue then converts it to BF16 exactly once, stores qkv_buf,
    // and mirrors K/V from that same BF16 shared-memory tile.
    mk_gemm::gemm_unified_tile<GemmaQkvGemm>(
        g.encoder_x,
        weight,
        g.qkv_buf + epilogue.output_col_offset,
        g.seq,
        section_width,
        gemma::kWidth,
        row_tile,
        col_tile,
        col_tile + 1,
        epilogue,
        smem,
        false,
        gemma::kWidth,
        gemma::kWidth,
        gemma::kQkvWidth);
}

}  // namespace pi0_a100
