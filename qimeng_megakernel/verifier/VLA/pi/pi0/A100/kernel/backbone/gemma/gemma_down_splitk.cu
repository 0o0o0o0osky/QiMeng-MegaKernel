#pragma once

#include "backbone/gemma/gemma_gemm.cuh"
#include "common/gemm_splitk.cuh"
#include "common/gemma_globals.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {

// The two scheduler instructions for a 64x128 output tile own disjoint halves
// of K.  gate_buf is dead after UP_MUL and has enough storage for two fp32
// [seq_pad, width] planes, so it is the split-K workspace without extending the
// production ABI.  A per-tile counter lives in otherwise unused nonzero Bar
// slots; Bar is already cleared once at module launch.
//
// Retained for the frozen instruction ABI; supported production shapes use Down.
__device__ __noinline__ void gemma_down_splitk(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    constexpr int kSplits = 2;
    // The output tile is 64 rows by 128 columns.
    constexpr int kLogicalTileM = GemmaGemm64::BM;
    constexpr int kLogicalTileN = 128;
    constexpr int kNSubtiles = kLogicalTileN / GemmaGemm64::BN;
    constexpr int kMSubtiles = kLogicalTileM / GemmaGemm64::BM;
    constexpr int kNLogicalTiles = gemma::kWidth / kLogicalTileN;
    constexpr int kCounterSlotsPerRow = kBarrierSlots - 1;
    constexpr int kKTiles = gemma::kMlpWidth / GemmaGemm64::BK;
    constexpr int kKTilesPerSplit = kKTiles / kSplits;

    static_assert(kNSubtiles == 2 && kMSubtiles == 1);
    static_assert(kKTiles % kSplits == 0);

    const int layer = inst[1];
    const int logical_m = inst[2];
    const int logical_n = inst[3];
    const int split = inst[7];
    const int tile_index = logical_m * kNLogicalTiles + logical_n;
    const int64_t output_plane =
        static_cast<int64_t>(g.seq) * gemma::kWidth;
    float* workspace = reinterpret_cast<float*>(g.gate_buf);
    float* split_workspace = workspace + split * output_plane;
    const nv_bfloat16* weight = g.down_w +
        static_cast<int64_t>(layer) * gemma::kWidth * gemma::kMlpWidth;

    #pragma unroll
    for (int m_sub = 0; m_sub < kMSubtiles; ++m_sub) {
        #pragma unroll
        for (int n_sub = 0; n_sub < kNSubtiles; ++n_sub) {
            mk_gemm::gemm_splitk_partial<GemmaGemm64>(
                g.up_buf,
                weight,
                split_workspace,
                g.seq,
                gemma::kWidth,
                gemma::kMlpWidth,
                kMSubtiles * logical_m + m_sub,
                kNSubtiles * logical_n + n_sub,
                split * kKTilesPerSplit,
                kKTilesPerSplit,
                smem);
        }
    }

    // Publish every thread's fragment stores before one lane announces that
    // this split is complete.  The CTA observing the second arrival owns the
    // fp32 reduction and the residual epilogue for this tile.
    __threadfence();
    __syncthreads();
    int* last_split = reinterpret_cast<int*>(smem);
    if (threadIdx.x == 0) {
        const int counter_row = tile_index / kCounterSlotsPerRow;
        const int counter_slot = 1 + tile_index % kCounterSlotsPerRow;
        int32_t* counter = g.barriers +
            (static_cast<int64_t>(layer) * gemma::kBarrierRows + counter_row) *
                kBarrierSlots +
            counter_slot;
        *last_split = atomicAdd(counter, 1) == (kSplits - 1);
    }
    __syncthreads();
    if (!*last_split) return;
    __threadfence();

    constexpr int kValuesPerVector = 8;
    constexpr int kTileVectors =
        kLogicalTileM * kLogicalTileN / kValuesPerVector;
    for (int vector = threadIdx.x; vector < kTileVectors;
         vector += blockDim.x) {
        const int local_value = vector * kValuesPerVector;
        const int local_row = local_value / kLogicalTileN;
        const int local_col = local_value % kLogicalTileN;
        const int row = logical_m * kLogicalTileM + local_row;
        const int col = logical_n * kLogicalTileN + local_col;
        const int64_t offset =
            static_cast<int64_t>(row) * gemma::kWidth + col;

        const float4 split0_lo =
            *reinterpret_cast<const float4*>(workspace + offset);
        const float4 split0_hi =
            *reinterpret_cast<const float4*>(workspace + offset + 4);
        const float4 split1_lo = *reinterpret_cast<const float4*>(
            workspace + output_plane + offset);
        const float4 split1_hi = *reinterpret_cast<const float4*>(
            workspace + output_plane + offset + 4);
        const float partials[8] = {
            split0_lo.x + split1_lo.x,
            split0_lo.y + split1_lo.y,
            split0_lo.z + split1_lo.z,
            split0_lo.w + split1_lo.w,
            split0_hi.x + split1_hi.x,
            split0_hi.y + split1_hi.y,
            split0_hi.z + split1_hi.z,
            split0_hi.w + split1_hi.w,
        };
        alignas(16) nv_bfloat16 result[kValuesPerVector];
        #pragma unroll
        for (int item = 0; item < kValuesPerVector; ++item) {
            result[item] = __float2bfloat16(
                partials[item] +
                __bfloat162float(g.encoder_x[offset + item]));
        }
        *reinterpret_cast<uint4*>(g.encoder_x + offset) =
            *reinterpret_cast<const uint4*>(result);
    }
}

}  // namespace pi0_a100
