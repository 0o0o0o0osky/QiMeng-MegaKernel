#pragma once

#include <cuda_bf16.h>

#include "common/gemm_config.cuh"
#include "common/gemm_epilogue.cuh"
#include "common/gemma_globals.cuh"
#include "common/matmul_pipeline.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {
namespace gemma_attn_mat {

// Gemma attention is materialized in three phases:
//
//   QK       S = Q K^T for all eight query heads, scaled, with keys >= seq_len
//            written as -inf; the 32x32 V-transpose tiles ride in the same
//            instruction stream because Vt depends only on the KV cache, which
//            `gemma_qkv_rope` published two barriers earlier.
//   Softmax  a row softmax over the padded key extent, in place over S.
//   PV       O = P Vt, written straight into `attn_out` at column head*256.
//
// MQA lets all query heads share K/V and stack along the GEMM M dimension.
// QK writes -inf through the padded tail so softmax needs no key-bound branch.

constexpr int kHeads = gemma::kHeads;
constexpr int kHeadDim = gemma::kHeadDim;
constexpr float kScale = 0.0625f;  // 1/sqrt(256)

// QK is M = seq_pad, N = seq_pad, K = 256; PV is M = seq_pad, N = 256,
// K = seq_pad.  The two shapes want different tiles and nothing forces them to
// share one -- an opcode is its own device function and carries its own config
// as long as the module's one shared-memory request still covers it.
using QkConfig = mk_gemm::GemmConfig<128, 128, 32, 3, 2, 2, 2>;
using PvConfig = mk_gemm::GemmConfig<64, 128, 32, 4, 2, 2, 2>;

static_assert(QkConfig::kNumThreads == kThreads);
static_assert(PvConfig::kNumThreads == kThreads);

// One 32x32 staged tile transposes V.  33 columns break the bank conflict on
// the strided read.
constexpr int kVtTile = 32;
constexpr int kVtSmemElems = kVtTile * (kVtTile + 1);
constexpr int kVtTilesX = kHeadDim / kVtTile;

// Each softmax lane owns eight consecutive keys; the tail is predicated.
constexpr int kSoftmaxKeysPerLane = 8;
constexpr int kSoftmaxChunkKeys = 32 * kSoftmaxKeysPerLane;
constexpr int kSoftmaxChunks =
    (gemma::kMaxSeq + kSoftmaxChunkKeys - 1) / kSoftmaxChunkKeys;
constexpr int kSoftmaxWarps = kThreads / 32;

struct QkEpilogue {
    float scale;
    int key_limit;
    __device__ __forceinline__ float transform(float acc, int, int n) const {
        return (n < key_limit) ? acc * scale : -INFINITY;
    }
};

// V is [seq, head_dim] in the cache and PV needs the key axis contiguous in a
// V-shaped operand. Append the required transpose work to the QK phase.
__device__ __forceinline__ void transpose_v_tile(
    int tile,
    const nv_bfloat16* __restrict__ src,
    nv_bfloat16* __restrict__ dst,
    int keys_valid,
    int ld_dst,
    nv_bfloat16* tile_smem) {
    const int x0 = (tile % kVtTilesX) * kVtTile;
    const int y0 = (tile / kVtTilesX) * kVtTile;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int r = warp; r < kVtTile; r += kSoftmaxWarps) {
        const int key = y0 + r;
        tile_smem[r * (kVtTile + 1) + lane] =
            (key < keys_valid)
                ? src[static_cast<int64_t>(key) * kHeadDim + x0 + lane]
                : __float2bfloat16(0.0f);
    }
    __syncthreads();
    // Keys past `keys_valid` land as exact zeros, which is what makes PV safe to
    // run at K = seq_pad: the padded columns of Vt multiply the padded columns
    // of P, which the softmax also zeroed.
    for (int r = warp; r < kVtTile; r += kSoftmaxWarps) {
        dst[static_cast<int64_t>(x0 + r) * ld_dst + y0 + lane] =
            tile_smem[lane * (kVtTile + 1) + r];
    }
    __syncthreads();
}

}  // namespace gemma_attn_mat

// Opcode 4.  `inst[9] == 0` is a QK GEMM tile for head `inst[7]`; anything else
// is V-transpose tile `inst[2]`.  Both branches are CTA-uniform.
__device__ __noinline__ void gemma_attn_qk(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    using namespace gemma_attn_mat;

    const int layer = inst[1];
    const int seq_pad = g.seq;
    const int seq_len = inst[8];

    if (inst[9] != 0) {
        transpose_v_tile(
            inst[2],
            g.v_cache +
                static_cast<int64_t>(layer) * gemma::kMaxSeq * kHeadDim,
            g.attn_vt,
            seq_len,
            seq_pad,
            reinterpret_cast<nv_bfloat16*>(smem));
        return;
    }

    // `m = head * seq_pad + row` maps to `qkv_buf` offset
    // `row * kQkvWidth + head * kHeadDim`, which is not affine in m, so one call
    // cannot address all eight heads.  It does not have to: the head rides in
    // the instruction word and the base pointer moves, which costs nothing and
    // needs no repack of Q.
    const int head = inst[7];
    QkEpilogue epi{kScale, seq_len};
    mk_gemm::gemm_unified_tile<QkConfig>(
        g.qkv_buf + head * kHeadDim,
        g.k_cache + static_cast<int64_t>(layer) * gemma::kMaxSeq * kHeadDim,
        g.attn_scores + static_cast<int64_t>(head) * seq_pad * seq_pad,
        seq_pad, seq_pad, kHeadDim,
        inst[2], inst[3], inst[3] + 1,
        epi, smem, false,
        gemma::kQkvWidth, kHeadDim, seq_pad);
}

// Opcode 5.  One warp per row, the row held in registers, P written over S.
// Aliasing the two is free in time and halves the score-matrix footprint from
// 22.1 MB to 11.1.
__device__ __noinline__ void gemma_attn_softmax(
    const GemmaGlobals& g, const int32_t* inst) {
    using namespace gemma_attn_mat;

    const int ld = g.seq;
    const int total_rows = kHeads * ld;
    const int row0 = inst[2];
    const int rows = inst[7];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int r = warp; r < rows; r += kSoftmaxWarps) {
        const int row = row0 + r;
        if (row >= total_rows) break;
        nv_bfloat16* line = g.attn_scores + static_cast<int64_t>(row) * ld;

        float value[kSoftmaxChunks][kSoftmaxKeysPerLane];
        float row_max = -INFINITY;
        #pragma unroll
        for (int c = 0; c < kSoftmaxChunks; ++c) {
            const int first = c * kSoftmaxChunkKeys + lane * kSoftmaxKeysPerLane;
            if (first < ld) {
                const uint4 raw = *reinterpret_cast<const uint4*>(line + first);
                const nv_bfloat16* packed =
                    reinterpret_cast<const nv_bfloat16*>(&raw);
                #pragma unroll
                for (int j = 0; j < kSoftmaxKeysPerLane; ++j) {
                    value[c][j] = __bfloat162float(packed[j]);
                    row_max = fmaxf(row_max, value[c][j]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < kSoftmaxKeysPerLane; ++j) {
                    value[c][j] = -INFINITY;
                }
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, offset));
        }

        float row_sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < kSoftmaxChunks; ++c) {
            #pragma unroll
            for (int j = 0; j < kSoftmaxKeysPerLane; ++j) {
                value[c][j] = __expf(value[c][j] - row_max);
                row_sum += value[c][j];
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, offset);
        }

        const float inverse = 1.0f / row_sum;
        #pragma unroll
        for (int c = 0; c < kSoftmaxChunks; ++c) {
            const int first = c * kSoftmaxChunkKeys + lane * kSoftmaxKeysPerLane;
            if (first < ld) {
                uint4 raw;
                nv_bfloat16* packed = reinterpret_cast<nv_bfloat16*>(&raw);
                #pragma unroll
                for (int j = 0; j < kSoftmaxKeysPerLane; ++j) {
                    packed[j] = __float2bfloat16(value[c][j] * inverse);
                }
                *reinterpret_cast<uint4*>(line + first) = raw;
            }
        }
    }
}

// Opcode 11.  O = P Vt straight into `attn_out` at column head*256 with
// ldc = kWidth, which is where `gemma_oproj` already reads it.
__device__ __noinline__ void gemma_attn_pv(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    using namespace gemma_attn_mat;

    const int seq_pad = g.seq;
    const int head = inst[7];
    mk_gemm::EpilogueIdentity epi{};
    mk_gemm::gemm_unified_tile<PvConfig>(
        g.attn_scores + static_cast<int64_t>(head) * seq_pad * seq_pad,
        g.attn_vt,
        g.attn_out + head * kHeadDim,
        seq_pad, kHeadDim, seq_pad,
        inst[2], inst[3], inst[3] + 1,
        epi, smem, false,
        seq_pad, seq_pad, gemma::kWidth);
}

}  // namespace pi0_a100
