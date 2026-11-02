#pragma once

// Independent-kernel Action Expert attention for CUDA Graph execution.
//
// This deliberately stays outside mk_ae.cu: it is the standalone frontier used
// before MegaKernel fusion.  QK and PV are pure CUDA/CUTE GEMMs (no
// cuBLAS).  The score matrix is compacted around the physical KV-cache gap, and
// the QK launch also transposes V so the full path remains three launches.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "common/ae_globals.cuh"
#include "common/gemm_epilogue.cuh"
#include "common/matmul_pipeline.cuh"

namespace pi0_a100::ae_materialized {

using QConfig = mk_gemm::GemmConfig<32, 32, 64, 3, 2, 2, 1>;
using PvConfig = mk_gemm::GemmConfig<32, 32, 64, 4, 2, 2, 1>;

static constexpr int kHeads = 8;
static constexpr int kHeadDim = 256;
static constexpr int kPhysicalRows = 64;
static constexpr int kSoftmaxChunks = 4;

struct Fp32ScaleAll {
    static constexpr bool kUseFp32TerminalStore = true;
    float scale;
    float* output;
    int stride;
    __device__ __forceinline__ float transform(float acc, int, int) const {
        return acc * scale;
    }
    __device__ __forceinline__ void store(float value, int row, int col) const {
        output[static_cast<int64_t>(row) * stride + col] = value;
    }
};

struct Fp32ScaleSelfHeadMajor {
    static constexpr bool kUseFp32TerminalStore = true;
    float scale;
    float* output;
    int stride;
    __device__ __forceinline__ float transform(float acc, int row, int col) const {
        return (row != 0 || col == 0) ? acc * scale : -INFINITY;
    }
    __device__ __forceinline__ void store(float value, int row, int col) const {
        output[static_cast<int64_t>(row) * stride + col] = value;
    }
};

__device__ __forceinline__ void transpose_v_compact_tile(
    int tile, const nv_bfloat16* __restrict__ src,
    nv_bfloat16* __restrict__ dst, int prefix, int state_key,
    int self_slot, int self_rows, int stride, nv_bfloat16* tile_smem) {
    const int x0 = (tile & 7) * 32;
    const int y0 = (tile >> 3) * 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int row = warp; row < 32; row += 4) {
        const int dst_key = y0 + row;
        int src_key = -1;
        if (dst_key < prefix) src_key = dst_key;
        else if (dst_key >= self_slot && dst_key < self_slot + self_rows)
            src_key = state_key + dst_key - self_slot;
        tile_smem[row * 33 + lane] = src_key >= 0
            ? src[static_cast<int64_t>(src_key) * kHeadDim + x0 + lane]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();
    for (int row = warp; row < 32; row += 4) {
        const int key = y0 + lane;
        if (key < stride)
            dst[static_cast<int64_t>(x0 + row) * stride + key] =
                tile_smem[lane * 33 + row];
    }
    __syncthreads();
}

template <typename Config>
__global__ __launch_bounds__(Config::kNumThreads, 1) void qk_vt(
    const nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    nv_bfloat16* __restrict__ vt,
    float* __restrict__ scores,
    int query_rows, int prefix, int state_key,
    int self_slot, int stride, int query_tm,
    int prefix_tn, int self_tn) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int prefix_per_head = query_tm * prefix_tn;
    const int self_per_head = query_tm * self_tn;
    const int prefix_tiles = kHeads * prefix_per_head;
    const int self_tiles = kHeads * self_per_head;
    const int vt_tiles = (stride / 32) * 8;
    const int tasks = prefix_tiles + self_tiles + vt_tiles;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        if (task < prefix_tiles) {
            const int head = task / prefix_per_head;
            const int local = task - head * prefix_per_head;
            const int tm = local / prefix_tn;
            const int tn = local - tm * prefix_tn;
            mk_gemm::gemm_unified_tile<Config>(
                qkv + head * kHeadDim, k,
                reinterpret_cast<nv_bfloat16*>(scores),
                query_rows, prefix, kHeadDim, tm, tn, tn + 1,
                Fp32ScaleAll{
                    1.0f / 16.0f,
                    scores + static_cast<int64_t>(head) * query_rows * stride,
                    stride}, smem, false,
                kHeads * kHeadDim + 2 * kHeadDim, kHeadDim, stride);
        } else if (task < prefix_tiles + self_tiles) {
            const int relative = task - prefix_tiles;
            const int head = relative / self_per_head;
            const int local = relative - head * self_per_head;
            const int tm = local / self_tn;
            const int tn = local - tm * self_tn;
            mk_gemm::gemm_unified_tile<Config>(
                qkv + head * kHeadDim,
                k + static_cast<int64_t>(state_key) * kHeadDim,
                reinterpret_cast<nv_bfloat16*>(scores),
                query_rows, query_rows, kHeadDim, tm, tn, tn + 1,
                Fp32ScaleSelfHeadMajor{
                    1.0f / 16.0f,
                    scores + static_cast<int64_t>(head) * query_rows * stride + self_slot,
                    stride}, smem, false,
                kHeads * kHeadDim + 2 * kHeadDim, kHeadDim, stride);
        } else {
            const int vt_task = task - prefix_tiles - self_tiles;
            const int invalid_per_row = stride - prefix - query_rows;
            const int invalid_index = vt_task * blockDim.x + threadIdx.x;
            if (invalid_index < kHeads * query_rows * invalid_per_row) {
                const int row = invalid_index / invalid_per_row;
                const int offset = invalid_index - row * invalid_per_row;
                const int gap = self_slot - prefix;
                const int col = offset < gap
                    ? prefix + offset
                    : self_slot + query_rows + offset - gap;
                scores[static_cast<int64_t>(row) * stride + col] = -INFINITY;
            }
            __syncthreads();
            transpose_v_compact_tile(
                vt_task, v, vt, prefix, state_key, self_slot,
                query_rows, stride, reinterpret_cast<nv_bfloat16*>(smem));
        }
    }
}

__global__ __launch_bounds__(128) void softmax_in_place(
    float* __restrict__ scores, nv_bfloat16* __restrict__ probabilities,
    int rows, int stride) {
    const int lane = threadIdx.x & 31;
    const int warp = static_cast<int>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    const int warp_count = static_cast<int>(gridDim.x) * 4;
    for (int row = warp; row < rows; row += warp_count) {
        const int64_t base = static_cast<int64_t>(row) * stride;
        float value[kSoftmaxChunks][8];
        float row_max = -INFINITY;
        #pragma unroll
        for (int chunk = 0; chunk < kSoftmaxChunks; ++chunk) {
            const int first = chunk * 256 + lane * 8;
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                value[chunk][item] = first + item < stride
                    ? scores[base + first + item] : -INFINITY;
                row_max = fmaxf(row_max, value[chunk][item]);
            }
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            row_max = fmaxf(row_max,
                            __shfl_xor_sync(0xffffffffu, row_max, offset));
        float row_sum = 0.0f;
        #pragma unroll
        for (int chunk = 0; chunk < kSoftmaxChunks; ++chunk) {
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                value[chunk][item] = isfinite(value[chunk][item])
                    ? __expf(value[chunk][item] - row_max) : 0.0f;
                row_sum += value[chunk][item];
            }
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, offset);
        const float inverse = 1.0f / row_sum;
        #pragma unroll
        for (int chunk = 0; chunk < kSoftmaxChunks; ++chunk) {
            const int first = chunk * 256 + lane * 8;
            if (first >= stride) continue;
            uint4 raw{};
            nv_bfloat16* packed = reinterpret_cast<nv_bfloat16*>(&raw);
            #pragma unroll
            for (int item = 0; item < 8; ++item)
                packed[item] = __float2bfloat16(value[chunk][item] * inverse);
            *reinterpret_cast<uint4*>(
                probabilities + static_cast<int64_t>(row) * (2 * stride) + first) = raw;
        }
    }
}

template <typename Config>
__global__ __launch_bounds__(Config::kNumThreads, 1) void pv(
    const nv_bfloat16* __restrict__ probabilities,
    const nv_bfloat16* __restrict__ vt,
    nv_bfloat16* __restrict__ output,
    int query_rows, int stride, int probability_stride,
    int pv_tm, int pv_tn) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int per_head = pv_tm * pv_tn;
    const int tasks = kHeads * per_head;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int head = task / per_head;
        const int local = task - head * per_head;
        const int tm = local / pv_tn;
        const int tn = local - tm * pv_tn;
        mk_gemm::gemm_unified_tile<Config>(
            probabilities + static_cast<int64_t>(head) * query_rows * probability_stride,
            vt,
            output + static_cast<int64_t>(head) * kPhysicalRows * kHeadDim,
            query_rows, kHeadDim, stride, tm, tn, tn + 1,
            mk_gemm::EpilogueIdentity{}, smem, false,
            probability_stride, stride, kHeadDim);
    }
}

inline void launch(const AeGlobals& g, int layer, int query_rows,
                   cudaStream_t stream) {
    const int self_slot = ((g.prefix_len + 31) / 32) * 32;
    const int stride = ((self_slot + query_rows + 31) / 32) * 32;
    const int query_tm = (query_rows + QConfig::BM - 1) / QConfig::BM;
    const int prefix_tn = (g.prefix_len + QConfig::BN - 1) / QConfig::BN;
    const int self_tn = (query_rows + QConfig::BN - 1) / QConfig::BN;
    const int qk_tasks = kHeads * query_tm * (prefix_tn + self_tn);
    const int vt_tasks = (stride / 32) * 8;
    const int rows = kHeads * query_rows;
    const int pv_tm = (query_rows + PvConfig::BM - 1) / PvConfig::BM;
    const int pv_tn = (kHeadDim + PvConfig::BN - 1) / PvConfig::BN;
    float* scores = reinterpret_cast<float*>(g.attn_partial);
    nv_bfloat16* probabilities = g.attn_partial;
    const int probability_stride = 2 * stride;
    nv_bfloat16* vt = g.attn_partial + static_cast<int64_t>(2 * rows) * stride;
    const nv_bfloat16* k =
        g.k_cache + static_cast<int64_t>(layer) * g.kv_rows * kHeadDim;
    const nv_bfloat16* v =
        g.v_cache + static_cast<int64_t>(layer) * g.kv_rows * kHeadDim;
    qk_vt<QConfig><<<qk_tasks + vt_tasks, QConfig::kNumThreads,
                      QConfig::kShmSize, stream>>>(
        g.qkv_buf, k, v, vt, scores, query_rows, g.prefix_len,
        g.state_key, self_slot, stride, query_tm, prefix_tn, self_tn);
    softmax_in_place<<<(rows + 3) / 4, 128, 0, stream>>>(
        scores, probabilities, rows, stride);
    pv<PvConfig><<<kHeads * pv_tm * pv_tn, PvConfig::kNumThreads,
                   PvConfig::kShmSize, stream>>>(
        probabilities, vt, g.attn_out, query_rows, stride,
        probability_stride, pv_tm, pv_tn);
}

}  // namespace pi0_a100::ae_materialized
