#pragma once

#include "kernel_config.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cutlass/numeric_types.h>
#include <type_traits>

namespace megakernel {

using namespace cute;
using bfloat16_t = cutlass::bfloat16_t;

// ========================================================================
// 1. Warp Reduce
// ========================================================================
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    static_assert(sizeof(T) == 4, "warp_reduce_sum currently only supports 32-bit types.");
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ========================================================================
// 2. Block Reduce (RMS Norm)
// ========================================================================
template <int NumWarps, typename T, typename TensorSmem>
__device__ __forceinline__ T block_reduce_sum(T val, TensorSmem& t_smem) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    T warp_sum = warp_reduce_sum(val);

    if (lane_id == 0) t_smem(warp_id) = warp_sum;
    __syncthreads();

    T block_sum = T(0);
    if (warp_id == 0) {
        if (lane_id < NumWarps) block_sum = t_smem(lane_id);
        block_sum = warp_reduce_sum(block_sum);
    }
    if (tid == 0) t_smem(0) = block_sum;
    __syncthreads();

    return t_smem(0);
}

// ========================================================================
// 3. Warp-group reduce
// ========================================================================
template <typename T, typename TensorSmem>
__device__ __forceinline__ T warp_group_reduce(
    T partial_sum,
    TensorSmem& t_smem,
    int warps_per_row,
    int warp_id,
    int lane_id
) {
    if (lane_id == 0) t_smem(warp_id) = partial_sum;
    __syncthreads();

    T total_sum = T(0);
    int rank_in_row = warp_id % warps_per_row;
    int row_base_warp = warp_id - rank_in_row;

    if (rank_in_row == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < warps_per_row; ++i) {
            total_sum += t_smem(row_base_warp + i);
        }
    }
    return total_sum;
}

// ========================================================================
// 3.5 RoPE interleaved rotation (for Q/K in decode)
// ========================================================================
template <int TileM>
__device__ __forceinline__ float apply_rope_interleaved(
    float val, float cos_val, float sin_val, int lane_id
) {
    static_assert(TileM > 0 && TileM <= 32, "TileM must be <= 32 and > 0");
    static_assert(TileM % 2 == 0, "TileM must be even for RoPE");
    constexpr unsigned mask =
        (TileM == 32) ? 0xffffffffu : (1u << TileM) - 1;
    float neighbor = __shfl_xor_sync(mask, val, 1);
    float sign = (lane_id % 2 == 0) ? -1.0f : 1.0f;
    return val * cos_val + sign * neighbor * sin_val;
}

// ========================================================================
// 4. Tile copy (Global -> Shared)
// ========================================================================
template <int SizeBytes>
__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

    if constexpr (SizeBytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem_ptr));
    } else {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
            :: "r"(smem), "l"(gmem_ptr), "n"(SizeBytes)
        );
    }
}

template <int N>
__device__ __forceinline__ void g2s_cp_async_16b_n(const uint128_t* gmem, uint128_t* smem) {
    static_assert(N > 0 && N <= KernelConfig::NUM_THREADS,
                  "g2s_cp_async_16b_n: N must be in (0, NUM_THREADS].");
    auto t_g_128 = make_tensor(
        make_gmem_ptr(gmem),
        make_shape(Int<N>{}, Int<1>{}),
        LayoutRight{}
    );
    auto t_s_128 = make_tensor(
        make_smem_ptr(smem),
        make_shape(Int<N>{}, Int<1>{}),
        LayoutRight{}
    );
    using CopyOp = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using Atom = Copy_Atom<CopyOp, uint128_t>;
    auto tiled_copy = make_tiled_copy(
        Atom{},
        make_layout(Int<N>{}, LayoutRight{}),
        make_layout(Int<1>{})
    );
    int tid = threadIdx.x;
    if (tid < N) {
        auto thr_copy = tiled_copy.get_slice(tid);
        auto t_src = thr_copy.partition_S(t_g_128);
        auto t_dst = thr_copy.partition_D(t_s_128);
        copy(static_cast<Atom const&>(tiled_copy), t_src, t_dst);
    }
}

template <int TotalChunks>
__device__ __forceinline__ void g2s_cp_async_16b_loop(const uint128_t* gmem, uint128_t* smem) {
    constexpr int NUM_T = KernelConfig::NUM_THREADS;
    constexpr int NumFull = TotalChunks / NUM_T;
    constexpr int Last = TotalChunks % NUM_T;
    #pragma unroll
    for (int i = 0; i < NumFull; ++i) {
        g2s_cp_async_16b_n<NUM_T>(gmem + i * NUM_T, smem + i * NUM_T);
    }
    if constexpr (Last > 0) {
        g2s_cp_async_16b_n<Last>(gmem + NumFull * NUM_T, smem + NumFull * NUM_T);
    }
}

template <typename TensorG, typename TensorS>
__device__ __forceinline__ void load_tile_async(TensorG const& t_g, TensorS& t_s) {
    using G = std::remove_cv_t<std::remove_reference_t<TensorG>>;
    using LayoutG = typename G::layout_type;
    constexpr int TotalElems = cute::size(LayoutG{});
    constexpr int BytesPerAsync = 16;
    constexpr int N = (TotalElems * sizeof(typename G::value_type)) / BytesPerAsync;
    static_assert((TotalElems * sizeof(typename G::value_type)) % BytesPerAsync == 0,
                  "load_tile_async: tensor size must be a multiple of 16 bytes.");
    static_assert(N > 0, "load_tile_async: N must be positive.");
    const auto* gmem = reinterpret_cast<const uint128_t*>(t_g.data().get());
    uint128_t* smem = reinterpret_cast<uint128_t*>(t_s.data().get());
    if constexpr (N <= KernelConfig::NUM_THREADS) {
        g2s_cp_async_16b_n<N>(gmem, smem);
    } else {
        g2s_cp_async_16b_loop<N>(gmem, smem);
    }
}

// ========================================================================
// 5. RMS Norm
// ========================================================================
template <
    int NumWarps,
    typename TensorInput,
    typename TensorWeight,
    typename TensorSmemOut,
    typename TensorReduceWorkspace
>
__device__ __forceinline__ void rms_norm(
    const TensorInput& t_in_g,
    const TensorWeight& t_w_g,
    TensorSmemOut& t_out_s,
    float eps,
    TensorReduceWorkspace& t_smem_workspace
) {
    using Element = typename TensorInput::value_type;
    constexpr int HiddenDim = size(TensorInput{});
    constexpr int NumThreads = NumWarps * 32;
    constexpr int ElemsPerThread = HiddenDim / NumThreads;

    constexpr int BytesPerThread = ElemsPerThread * sizeof(Element);
    using VecType = typename std::conditional<BytesPerThread == 16, uint128_t,
                    typename std::conditional<BytesPerThread == 8,  uint64_t,
                                                                    uint32_t
                    >::type>::type;

    auto layout_vec = make_layout(make_shape(Int<NumThreads>{}, Int<ElemsPerThread>{}), LayoutRight{});

    auto t_g_in_vec = make_tensor(t_in_g.data(), layout_vec);
    auto t_g_w_vec  = make_tensor(t_w_g.data(),  layout_vec);
    auto t_s_out_vec= make_tensor(t_out_s.data(), layout_vec);

    auto t_thr_in = t_g_in_vec(threadIdx.x, _);
    auto t_thr_w  = t_g_w_vec(threadIdx.x, _);
    auto t_thr_out= t_s_out_vec(threadIdx.x, _);

    auto t_r_in = make_tensor_like(t_thr_in);
    auto t_r_w  = make_tensor_like(t_thr_w);

    copy(Copy_Atom<UniversalCopy<VecType>, Element>{}, t_thr_in, t_r_in);
    copy(Copy_Atom<UniversalCopy<VecType>, Element>{}, t_thr_w,  t_r_w);

    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < size(t_r_in); ++i) {
        float x = static_cast<float>(t_r_in(i));
        sum_sq += x * x;
    }

    float total_sum_sq = block_reduce_sum<NumWarps>(sum_sq, t_smem_workspace);
    float rms_scale = rsqrtf((total_sum_sq / float(HiddenDim)) + eps);

    #pragma unroll
    for (int i = 0; i < size(t_r_in); ++i) {
        float x = static_cast<float>(t_r_in(i));
        float w = static_cast<float>(t_r_w(i));
        // Match the fused/PyVM RMS boundary: normalize in fp32, round that
        // value to BF16, then apply the BF16 scale weight and round again.
        Element normalized = static_cast<Element>(x * rms_scale);
        t_r_in(i) = static_cast<Element>(static_cast<float>(normalized) * w);
    }
    copy(Copy_Atom<UniversalCopy<VecType>, Element>{}, t_r_in, t_thr_out);
}

// ========================================================================
// 6. Cross-CTA barrier (SM80 release/publish and acquire/wait over a GMEM counter)
// ========================================================================
__device__ __forceinline__ uint32_t barrier_load_relaxed(
    const int32_t* barrier_ptr
) {
    uint32_t value;
    asm volatile(
        "ld.relaxed.gpu.global.u32 %0, [%1];"
        : "=r"(value)
        : "l"(barrier_ptr)
        : "memory"
    );
    return value;
}

__device__ __forceinline__ uint32_t barrier_load_acquire(
    const int32_t* barrier_ptr
) {
    uint32_t value;
    asm volatile(
        "ld.acquire.gpu.global.u32 %0, [%1];"
        : "=r"(value)
        : "l"(barrier_ptr)
        : "memory"
    );
    return value;
}

}  // namespace megakernel

#ifndef NDEBUG
#include "include/debug/barrier_watchdog.cuh"
#endif

namespace megakernel {

__device__ __forceinline__ void barrier_wait(
    const int32_t* barrier_ptr,
    int expected_count
) {
#ifndef NDEBUG
    debug::barrier_wait_with_timeout(barrier_ptr, expected_count);
#else
    const uint32_t expected = static_cast<uint32_t>(expected_count);
    while (true) {
        if (barrier_load_relaxed(barrier_ptr) >= expected &&
            barrier_load_acquire(barrier_ptr) >= expected) {
            return;
        }
        __nanosleep(KernelConfig::GMEM_SPIN_LOOP_SLEEP_NANOS);
    }
#endif
}

__device__ __forceinline__ void barrier_publish(
    int32_t* barrier_ptr,
    int increment
) {
    asm volatile(
        "red.release.gpu.global.add.u32 [%0], %1;"
        :
        : "l"(barrier_ptr), "r"(static_cast<uint32_t>(increment))
        : "memory"
    );
}

// ========================================================================
// 7. GEMV Tile
// ========================================================================
template <
    int NumWarps,
    int TileM,
    int TileK,
    int SlicedK = 1,
    typename TensorInput,
    typename TensorWeight,
    typename TensorReduceWorkspace,
    typename TensorOutput
>
__device__ __forceinline__ void compute_gemv_tile(
    const TensorInput& t_input,
    const TensorWeight& t_weight,
    TensorReduceWorkspace& t_reduce,
    TensorOutput& t_output
) {
    constexpr int WarpsPerRow = NumWarps / TileM;
    static_assert(NumWarps % TileM == 0, "NumWarps must be divisible by TileM.");
    if constexpr (SlicedK > 1) {
        static_assert(TileK % SlicedK == 0, "TileK must be divisible by SlicedK when SlicedK > 1.");
    }

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int my_row_idx = warp_id / WarpsPerRow;
    int warp_in_row = warp_id % WarpsPerRow;

    using VecType = uint64_t;
    constexpr int ElemsPerVec = 4;
    constexpr int ThreadsPerWarp = 32;

    auto t_r_in_bf16 = make_tensor<bfloat16_t>(Shape<Int<ElemsPerVec>>{});
    auto t_r_w_bf16  = make_tensor<bfloat16_t>(Shape<Int<ElemsPerVec>>{});
    auto t_r_in_vec  = recast<VecType>(t_r_in_bf16);
    auto t_r_w_vec   = recast<VecType>(t_r_w_bf16);

    float acc = 0.0f;

    if constexpr (SlicedK == 1) {
        constexpr int K_Chunk = TileK / WarpsPerRow;
        auto t_in_tile = local_tile(t_input, Shape<Int<K_Chunk>>{}, make_coord(warp_in_row));
        auto t_w_row   = t_weight(my_row_idx, _);
        auto t_w_tile  = local_tile(t_w_row, Shape<Int<K_Chunk>>{}, make_coord(warp_in_row));

        auto t_in_vec_view = recast<VecType>(t_in_tile);
        auto t_w_vec_view  = recast<VecType>(t_w_tile);
        int K_Chunk_Vec = size(t_in_vec_view);

        for (int k = lane_id; k < K_Chunk_Vec; k += ThreadsPerWarp) {
            Tensor t_src_in = make_tensor(&t_in_vec_view(k), Shape<_1>{});
            Tensor t_src_w  = make_tensor(&t_w_vec_view(k),  Shape<_1>{});
            copy(Copy_Atom<UniversalCopy<VecType>, VecType>{}, t_src_in, t_r_in_vec);
            copy(Copy_Atom<UniversalCopy<VecType>, VecType>{}, t_src_w,  t_r_w_vec);
            #pragma unroll
            for (int e = 0; e < ElemsPerVec; ++e) {
                acc = fmaf(float(t_r_in_bf16(e)), float(t_r_w_bf16(e)), acc);
            }
        }
    } else {
        if (lane_id == 0) t_reduce(warp_id) = 0.0f;
        constexpr int K_slice = TileK / SlicedK;
        for (int sk = 0; sk < SlicedK; ++sk) {
            auto t_in_slice = local_tile(t_input, Shape<Int<K_slice>>{}, make_coord(sk));
            auto t_w_row    = t_weight(my_row_idx, _);
            auto t_w_slice  = local_tile(t_w_row, Shape<Int<K_slice>>{}, make_coord(sk));

            auto t_in_vec_view = recast<VecType>(t_in_slice);
            auto t_w_vec_view  = recast<VecType>(t_w_slice);
            int K_slice_Vec = size(t_in_vec_view);

            float partial = 0.0f;
            for (int k = lane_id; k < K_slice_Vec; k += ThreadsPerWarp) {
                Tensor t_src_in = make_tensor(&t_in_vec_view(k), Shape<_1>{});
                Tensor t_src_w  = make_tensor(&t_w_vec_view(k),  Shape<_1>{});
                copy(Copy_Atom<UniversalCopy<VecType>, VecType>{}, t_src_in, t_r_in_vec);
                copy(Copy_Atom<UniversalCopy<VecType>, VecType>{}, t_src_w,  t_r_w_vec);
                #pragma unroll
                for (int e = 0; e < ElemsPerVec; ++e) {
                    partial = fmaf(float(t_r_in_bf16(e)), float(t_r_w_bf16(e)), partial);
                }
            }
            partial = warp_reduce_sum(partial);
            if (lane_id == 0) {
                float t = t_reduce(warp_id);
                t_reduce(warp_id) = t + partial;
            }
        }
        __syncthreads();
        float final_row_sum = (lane_id == 0) ? t_reduce(warp_id) : 0.0f;
        float final_val = warp_group_reduce(final_row_sum, t_reduce, WarpsPerRow, warp_id, lane_id);
        if (warp_in_row == 0 && lane_id == 0) t_output(my_row_idx) = final_val;
        return;
    }

    float row_sum = warp_reduce_sum(acc);
    float final_val = warp_group_reduce(row_sum, t_reduce, WarpsPerRow, warp_id, lane_id);

    if (warp_in_row == 0 && lane_id == 0) {
        t_output(my_row_idx) = final_val;
    }
}
}  // namespace megakernel
