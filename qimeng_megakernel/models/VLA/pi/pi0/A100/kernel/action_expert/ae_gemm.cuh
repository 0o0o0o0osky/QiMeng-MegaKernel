#pragma once

#include <cuda_bf16.h>

#include "common/gemm_dual.cuh"
#include "common/gemm_epilogue.cuh"
#include "common/gemm_splitk.cuh"
#include "common/matmul_pipeline.cuh"

namespace pi0_a100 {

using AeGemm32 = mk_gemm::GemmConfig<64, 32, 64, 3, 2, 2, 1>;

// QKV's 80 column tiles allow a 32-row split while remaining within one
// 216-CTA round. Other decoder GEMMs use 64 rows to avoid a second round.
using AeQkvGemm = mk_gemm::GemmConfig<32, 32, 64, 3, 2, 2, 1>;

struct AeRowScale {
    const float* factor;

    __device__ __forceinline__ float transform(float acc, int row, int) const {
        return acc * factor[row];
    }
};

// A 32-column tile exposes 128 CTAs without increasing weight traffic.
using AeDecoderGemm = AeGemm32;

// Accumulate adjacent physical K planes in registers before one FP32 global
// epilogue.  AE activations and residual weights are both K-plane-major, so
// each plane is an ordinary contiguous [M,Kseg] / [N,Kseg] GEMM.  Keeping the
// accumulator across planes reduces partial traffic.  The ordinary residual
// path uses atomic add; deterministic OProj/Down overwrite disjoint scratch.
template <typename Config, int Segments>
__device__ __forceinline__ void ae_segmented_partial(
    const typename Config::T* __restrict__ a_base,
    const typename Config::T* __restrict__ b_base,
    float* __restrict__ output,
    int n,
    int tile_n,
    void* smem,
    int runtime_m = ae::kRows,
    bool overwrite_output = false) {
    using T = typename Config::T;
    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;
    constexpr int kStage = Config::kStage;
    constexpr int M = ae::kRows;
    constexpr int K = ae::kKSplit;
    constexpr int kTiles = K / BK;
    static_assert(BM <= M && K % BK == 0 && Segments > 0);

    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using TiledMMA = typename Config::TiledMMA;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyA_ZFILL = typename Config::G2SCopyA_ZFILL;
    using G2SCopyB = typename Config::G2SCopyB;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;

    T* a_smem_ptr = reinterpret_cast<T*>(smem);
    T* b_smem_ptr = a_smem_ptr + cute::cosize(SmemLayoutA{});
    auto sA = cute::make_tensor(cute::make_smem_ptr(a_smem_ptr), SmemLayoutA{});
    auto sB = cute::make_tensor(cute::make_smem_ptr(b_smem_ptr), SmemLayoutB{});

    const int tid = threadIdx.x;
    TiledMMA tiled_mma;
    auto mma_thread = tiled_mma.get_slice(tid);

    G2SCopyA g2s_a;
    auto g2s_a_thread = g2s_a.get_slice(tid);
    G2SCopyA_ZFILL g2s_a_zfill;
    G2SCopyB g2s_b;
    auto g2s_b_thread = g2s_b.get_slice(tid);

    auto s2r_a = cute::make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_a_thread = s2r_a.get_slice(tid);
    auto tAsA = s2r_a_thread.partition_S(sA);
    auto s2r_b = cute::make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_b_thread = s2r_b.get_slice(tid);
    auto tBsB = s2r_b_thread.partition_S(sB);

    auto a0 = cute::make_tensor(
        cute::make_gmem_ptr(a_base),
        cute::make_shape(runtime_m, K),
        cute::make_stride(K, cute::Int<1>{}));
    auto b0 = cute::make_tensor(
        cute::make_gmem_ptr(b_base),
        cute::make_shape(n, K),
        cute::make_stride(K, cute::Int<1>{}));
    auto gA0 = cute::local_tile(
        a0, cute::make_tile(cute::Int<BM>{}, cute::Int<BK>{}),
        cute::make_coord(0, cute::_));
    auto gB0 = cute::local_tile(
        b0, cute::make_tile(cute::Int<BN>{}, cute::Int<BK>{}),
        cute::make_coord(tile_n, cute::_));
    auto fragment_a = mma_thread.partition_fragment_A(gA0(_, _, 0));
    auto fragment_b = mma_thread.partition_fragment_B(gB0(_, _, 0));
    T* c_dummy = a_smem_ptr;
    auto c_fake = cute::make_tensor(
        cute::make_gmem_ptr(c_dummy),
        cute::make_shape(cute::Int<BM>{}, cute::Int<BN>{}),
        cute::make_stride(cute::Int<BN>{}, cute::Int<1>{}));
    auto accumulator = mma_thread.partition_fragment_C(c_fake);
    cute::clear(accumulator);
    auto fragment_a_view = s2r_a_thread.retile_D(fragment_a);
    auto fragment_b_view = s2r_b_thread.retile_D(fragment_b);

    #pragma unroll
    for (int segment = 0; segment < Segments; ++segment) {
        auto a = cute::make_tensor(
            cute::make_gmem_ptr(a_base + static_cast<int64_t>(segment) * M * K),
            cute::make_shape(runtime_m, K),
            cute::make_stride(K, cute::Int<1>{}));
        auto b = cute::make_tensor(
            cute::make_gmem_ptr(b_base + static_cast<int64_t>(segment) * n * K),
            cute::make_shape(n, K),
            cute::make_stride(K, cute::Int<1>{}));
        auto gA = cute::local_tile(
            a, cute::make_tile(cute::Int<BM>{}, cute::Int<BK>{}),
            cute::make_coord(0, cute::_));
        auto gB = cute::local_tile(
            b, cute::make_tile(cute::Int<BN>{}, cute::Int<BK>{}),
            cute::make_coord(tile_n, cute::_));
        auto tAgA = g2s_a_thread.partition_S(gA);
        auto tAsA_copy = g2s_a_thread.partition_D(sA);
        auto tBgB = g2s_b_thread.partition_S(gB);
        auto tBsB_copy = g2s_b_thread.partition_D(sB);
        auto cA = cute::make_identity_tensor(
            cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}));
        auto tAcA = g2s_a_thread.partition_S(cA);
        auto tApA = cute::make_tensor<bool>(cute::make_shape(
            cute::size<1>(tAsA_copy), cute::size<2>(tAsA_copy)));
        auto load_a = [&](int kt, int stage) {
            if (runtime_m == BM) {
                cute::copy(g2s_a, tAgA(_, _, _, kt),
                           tAsA_copy(_, _, _, stage));
            } else {
                CUTE_UNROLL
                for (int m = 0; m < cute::size<0>(tApA); ++m)
                    CUTE_UNROLL
                    for (int k = 0; k < cute::size<1>(tApA); ++k)
                        tApA(m, k) = cute::get<0>(tAcA(0, m, k)) < runtime_m;
                cute::copy_if(g2s_a_zfill, tApA, tAgA(_, _, _, kt),
                              tAsA_copy(_, _, _, stage));
            }
        };

        int next_tile = 0;
        int read_stage = 0;
        int write_stage = 0;
        #pragma unroll
        for (int stage = 0; stage < kStage - 1; ++stage) {
            load_a(next_tile, write_stage);
            cute::copy(g2s_b, tBgB(_, _, _, next_tile),
                       tBsB_copy(_, _, _, write_stage));
            cute::cp_async_fence();
            ++next_tile;
            ++write_stage;
        }
        cute::cp_async_wait<kStage - 2>();
        __syncthreads();
        cute::copy(
            s2r_a, tAsA(_, _, 0, read_stage), fragment_a_view(_, _, 0));
        cute::copy(
            s2r_b, tBsB(_, _, 0, read_stage), fragment_b_view(_, _, 0));

        #pragma unroll 1
        for (int kt = 0; kt < kTiles; ++kt) {
            const int fragments_k = cute::size<2>(fragment_a);
            #pragma unroll
            for (int ik = 0; ik < fragments_k; ++ik) {
                const int next_fragment = (ik + 1) % fragments_k;
                if (ik == fragments_k - 1) {
                    if (kt + kStage >= kTiles + 1) {
                        cute::cp_async_wait<0>();
                    } else {
                        cute::cp_async_wait<kStage - 2>();
                    }
                    __syncthreads();
                    read_stage = (read_stage + 1) % kStage;
                }
                cute::copy(
                    s2r_a,
                    tAsA(_, _, next_fragment, read_stage),
                    fragment_a_view(_, _, next_fragment));
                cute::copy(
                    s2r_b,
                    tBsB(_, _, next_fragment, read_stage),
                    fragment_b_view(_, _, next_fragment));
                if (ik == 0 && next_tile < kTiles) {
                    load_a(next_tile, write_stage);
                    cute::copy(g2s_b, tBgB(_, _, _, next_tile),
                               tBsB_copy(_, _, _, write_stage));
                    cute::cp_async_fence();
                    ++next_tile;
                    write_stage = (write_stage + 1) % kStage;
                }
                cute::gemm(
                    tiled_mma,
                    accumulator,
                    fragment_a(_, _, ik),
                    fragment_b(_, _, ik),
                    accumulator);
            }
        }
        cute::cp_async_wait<0>();
        __syncthreads();
    }

    auto coordinates = cute::make_identity_tensor(
        cute::make_shape(cute::Int<BM>{}, cute::Int<BN>{}));
    auto thread_coordinates = mma_thread.partition_C(coordinates);
    CUTE_UNROLL
    for (int value = 0; value < cute::size(accumulator); ++value) {
        auto coordinate = thread_coordinates(value);
        const int row = cute::get<0>(coordinate);
        const int col = tile_n * BN + cute::get<1>(coordinate);
        if (row < runtime_m && col < n) {
            float* destination = output + static_cast<int64_t>(row) * n + col;
            if (overwrite_output) {
                *destination = float(accumulator(value));
            } else {
                atomicAdd(destination, float(accumulator(value)));
            }
        }
    }
}

struct AeFfnMerge {
    const float* factor;

    __device__ __forceinline__ float transform(
        float gate, float up, int row) const {
        const float scale = factor[row];
        return mk_gemm::gelu_tanh(gate * scale) * (up * scale);
    }
};

struct AeActInEpilogue {
    const float* bias;

    __device__ __forceinline__ float transform(float acc, int, int col) const {
        const float value = acc + bias[col];
        return value / (1.0f + expf(-value));
    }
};

}  // namespace pi0_a100
