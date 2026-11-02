#pragma once

#include <cmath>
#include <cuda_bf16.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_conversion.h>

#include "common/pi0_config.cuh"
#include "common/siglip_globals.cuh"

namespace pi0_a100 {
namespace siglip_attn_tc {

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accum = float;

// A 64-row tile exposes enough attention CTAs without spilling the accumulator.
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kHeadDim = 96;
constexpr int kValidHeadDim = siglip::kHeadDim;
constexpr int kStoredHeadDim = siglip::kHeadDimPad;
constexpr int kWarps = 4;
constexpr int kThreadsTc = kWarps * 32;
constexpr int kKvBlocks = 256 / kBlockN;
constexpr float kScaleLog2E =
    0.11785113019775793f * 1.44269504088896340736f;

static_assert(kThreadsTc == kThreads);
static_assert(kHeadDim % 32 == 0);
static_assert(kValidHeadDim % 8 == 0);
static_assert(kStoredHeadDim % 8 == 0);

using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using TiledMma = TiledMMA<
    MmaAtom,
    Layout<Shape<Int<kWarps>, _1, _1>>,
    Tile<Int<16 * kWarps>, _16, _16>>;

constexpr int kBlockKSmem = 32;
constexpr int kSwizzle = 2;
using SmemLayoutAtom = decltype(composition(
    Swizzle<kSwizzle, 3, 3>{},
    Layout<Shape<_8, Int<kBlockKSmem>>,
           Stride<Int<kBlockKSmem>, _1>>{}));
using SmemLayoutQ = decltype(tile_to_shape(
    SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
using SmemLayoutKV = decltype(tile_to_shape(
    SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
using SmemLayoutVTransposed = decltype(composition(
    SmemLayoutKV{},
    make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
using SmemLayoutVTransposedNoSwizzle =
    decltype(get_nonswizzle_portion(SmemLayoutVTransposed{}));
using SmemLayoutO = SmemLayoutQ;

using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
using SmemCopyAtomO =
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

constexpr int kGmemValuesPerCopy = sizeof(cute::uint128_t) / sizeof(Element);
constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemValuesPerCopy;
using GmemThreadLayout = Layout<
    Shape<Int<kThreadsTc / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
    Stride<Int<kGmemThreadsPerRow>, _1>>;
using GmemCopyAtom = Copy_Atom<
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
using GmemTiledCopy = decltype(make_tiled_copy(
    GmemCopyAtom{}, GmemThreadLayout{}, Layout<Shape<_1, _8>>{}));
using GmemStoreAtom =
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
using GmemTiledStore = decltype(make_tiled_copy(
    GmemStoreAtom{}, GmemThreadLayout{}, Layout<Shape<_1, _8>>{}));

struct alignas(128) AttentionSmem {
    Element q[cute::cosize(SmemLayoutQ{})];
    Element k[cute::cosize(SmemLayoutKV{})];
    Element v[cute::cosize(SmemLayoutKV{})];
};

static_assert(sizeof(AttentionSmem) == 36 * 1024);

template <int Threads>
struct AllReduce {
    template <typename T, typename Op>
    __device__ __forceinline__ static T run(T value, Op op) {
        value = op(value, __shfl_xor_sync(0xffffffffu, value, Threads / 2));
        return AllReduce<Threads / 2>::run(value, op);
    }
};

template <>
struct AllReduce<2> {
    template <typename T, typename Op>
    __device__ __forceinline__ static T run(T value, Op op) {
        return op(value, __shfl_xor_sync(0xffffffffu, value, 1));
    }
};

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a + b;
    }
};

template <typename LayoutT>
__device__ __forceinline__ auto accumulator_row_col_layout(LayoutT layout) {
    static_assert(decltype(size<0>(layout))::value == 4);
    static_assert(decltype(rank(layout))::value == 3);
    auto divided = logical_divide(layout, Shape<_2>{});
    return make_layout(
        make_layout(get<1>(get<0>(divided)), get<1>(divided)),
        make_layout(get<0>(get<0>(divided)), get<2>(divided)));
}

template <typename Mma, typename LayoutT>
__device__ __forceinline__ auto accumulator_as_a_layout(LayoutT layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(layout))::value == 4);
    static_assert(decltype(rank(layout))::value == 3);
    constexpr int mma_k = get<2>(typename Mma::AtomShape_MNK{});
    static_assert(mma_k == 16);
    auto divided = logical_divide(layout, Shape<X, X, _2>{});
    return make_layout(
        make_layout(get<0>(divided), get<0>(get<2>(divided))),
        get<1>(divided),
        get<1>(get<2>(divided)));
}

template <typename To, typename Engine, typename LayoutT>
__device__ __forceinline__ auto convert_fragment(
    Tensor<Engine, LayoutT> const& source) {
    using From = typename Engine::value_type;
    constexpr int count = decltype(size(source))::value;
    cutlass::NumericArrayConverter<To, From, count> convert;
    auto converted = convert(
        *reinterpret_cast<const cutlass::Array<From, count>*>(source.data()));
    return make_tensor(make_rmem_ptr<To>(&converted), source.layout());
}

template <typename TensorIn, typename TensorOut, typename CoordTensor,
          typename TiledCopy>
__device__ __forceinline__ void copy_gmem_to_smem_padded(
    TiledCopy copy_op,
    TensorIn const& source,
    TensorOut& destination,
    CoordTensor const& coordinates) {
    CUTE_STATIC_ASSERT_V(rank(source) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(destination) == Int<3>{});
    #pragma unroll
    for (int row = 0; row < size<1>(source); ++row) {
        #pragma unroll
        for (int col = 0; col < size<2>(source); ++col) {
            if (get<1>(coordinates(0, row, col)) < kValidHeadDim) {
                cute::copy(
                    copy_op,
                    source(_, row, col),
                    destination(_, row, col));
            } else {
                cute::clear(destination(_, row, col));
            }
        }
    }
}

template <typename TensorIn, typename TensorOut, typename CoordTensor,
          typename TiledCopy>
__device__ __forceinline__ void copy_smem_to_gmem_valid(
    TiledCopy copy_op,
    TensorIn const& source,
    TensorOut& destination,
    CoordTensor const& coordinates) {
    CUTE_STATIC_ASSERT_V(rank(source) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(destination) == Int<3>{});
    #pragma unroll
    for (int row = 0; row < size<1>(source); ++row) {
        #pragma unroll
        for (int col = 0; col < size<2>(source); ++col) {
            if (get<1>(coordinates(0, row, col)) < kValidHeadDim) {
                cute::copy(
                    copy_op,
                    source(_, row, col),
                    destination(_, row, col));
            }
        }
    }
}

template <bool AInRegisters = false, typename TensorC, typename TensorA,
          typename TensorB, typename TensorSmemA, typename TensorSmemB,
          typename Mma, typename CopyA, typename CopyB,
          typename ThreadCopyA, typename ThreadCopyB>
__device__ __forceinline__ void tiled_gemm(
    TensorC& accumulator,
    TensorA& fragment_a,
    TensorB& fragment_b,
    TensorSmemA const& smem_a,
    TensorSmemB const& smem_b,
    Mma mma,
    CopyA copy_a,
    CopyB copy_b,
    ThreadCopyA thread_copy_a,
    ThreadCopyB thread_copy_b) {
    auto fragment_a_copy = thread_copy_a.retile_D(fragment_a);
    auto fragment_b_copy = thread_copy_b.retile_D(fragment_b);
    if (!AInRegisters) {
        cute::copy(copy_a, smem_a(_, _, _0{}), fragment_a_copy(_, _, _0{}));
    }
    cute::copy(copy_b, smem_b(_, _, _0{}), fragment_b_copy(_, _, _0{}));
    #pragma unroll
    for (int k = 0; k < size<2>(fragment_a); ++k) {
        if (k + 1 < size<2>(fragment_a)) {
            if (!AInRegisters) {
                cute::copy(
                    copy_a, smem_a(_, _, k + 1), fragment_a_copy(_, _, k + 1));
            }
            cute::copy(
                copy_b, smem_b(_, _, k + 1), fragment_b_copy(_, _, k + 1));
        }
        cute::gemm(
            mma,
            fragment_a(_, _, k),
            fragment_b(_, _, k),
            accumulator);
    }
}

template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSmemB, typename Mma, typename CopyB,
          typename ThreadCopyB>
__device__ __forceinline__ void tiled_gemm_register_smem(
    TensorC& accumulator,
    TensorA& fragment_a,
    TensorB& fragment_b,
    TensorSmemB const& smem_b,
    Mma mma,
    CopyB copy_b,
    ThreadCopyB thread_copy_b) {
    auto fragment_b_copy = thread_copy_b.retile_D(fragment_b);
    cute::copy(copy_b, smem_b(_, _, _0{}), fragment_b_copy(_, _, _0{}));
    #pragma unroll
    for (int k = 0; k < size<2>(fragment_a); ++k) {
        if (k + 1 < size<2>(fragment_a)) {
            cute::copy(
                copy_b, smem_b(_, _, k + 1), fragment_b_copy(_, _, k + 1));
        }
        cute::gemm(
            mma,
            fragment_a(_, _, k),
            fragment_b(_, _, k),
            accumulator);
    }
}

template <int RowsPerThread>
struct OnlineSoftmax {
    using RowTensor = decltype(make_tensor<float>(Shape<Int<RowsPerThread>>{}));
    RowTensor row_max;
    RowTensor row_sum;

    template <bool First, typename ScoreTensor, typename OutputTensor>
    __device__ __forceinline__ void update(
        ScoreTensor& score_accumulator,
        OutputTensor& output_accumulator) {
        auto scores = make_tensor(
            score_accumulator.data(),
            accumulator_row_col_layout(score_accumulator.layout()));

        #pragma unroll
        for (int row = 0; row < size<0>(scores); ++row) {
            float maximum = scores(row, 0);
            #pragma unroll
            for (int col = 1; col < size<1>(scores); ++col) {
                maximum = fmaxf(maximum, scores(row, col));
            }
            maximum = AllReduce<4>::run(maximum, MaxOp{});
            if constexpr (!First) {
                maximum = fmaxf(row_max(row), maximum);
            }

            if constexpr (!First) {
                const float correction =
                    exp2f((row_max(row) - maximum) * kScaleLog2E);
                row_sum(row) *= correction;
                auto output_rows = make_tensor(
                    output_accumulator.data(),
                    accumulator_row_col_layout(output_accumulator.layout()));
                #pragma unroll
                for (int col = 0; col < size<1>(output_rows); ++col) {
                    output_rows(row, col) *= correction;
                }
            }
            row_max(row) = maximum;

            float sum = First ? 0.0f : row_sum(row);
            const float maximum_scaled = maximum * kScaleLog2E;
            #pragma unroll
            for (int col = 0; col < size<1>(scores); ++col) {
                const float probability =
                    exp2f(scores(row, col) * kScaleLog2E - maximum_scaled);
                scores(row, col) = probability;
                sum += probability;
            }
            row_sum(row) = sum;
        }
    }

    template <typename OutputTensor>
    __device__ __forceinline__ void normalize(OutputTensor& output_accumulator) {
        auto output_rows = make_tensor(
            output_accumulator.data(),
            accumulator_row_col_layout(output_accumulator.layout()));
        #pragma unroll
        for (int row = 0; row < size<0>(output_rows); ++row) {
            const float sum = AllReduce<4>::run(row_sum(row), SumOp{});
            const float inverse = 1.0f / sum;
            #pragma unroll
            for (int col = 0; col < size<1>(output_rows); ++col) {
                output_rows(row, col) *= inverse;
            }
        }
    }
};

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

}  // namespace siglip_attn_tc

// Non-causal, per-view attention.  One instruction per (view, 64-query tile, head).
__device__ __noinline__ void siglip_attention(
    const SiglipGlobals& g, const int32_t* inst, unsigned char* smem) {
    using namespace siglip_attn_tc;

    AttentionSmem& storage = *reinterpret_cast<AttentionSmem*>(smem);
    const int tid = threadIdx.x;
    const int view = inst[2];
    const int query_tile = inst[3];
    const int head = inst[4];
    const int view_row0 = view * 256;
    const int query_row0 = view_row0 + query_tile * kBlockM;
    const int head_col = head * siglip::kHeadDimPad;
    const int q_col = head_col;
    const int k_col = siglip::kHeads * siglip::kHeadDimPad + head_col;
    const int v_col = 2 * siglip::kHeads * siglip::kHeadDimPad + head_col;

    const Element* q_ptr = reinterpret_cast<const Element*>(
        g.qkv_buf + static_cast<int64_t>(query_row0) * siglip::kQkvWidth + q_col);
    const Element* k_ptr = reinterpret_cast<const Element*>(
        g.qkv_buf + static_cast<int64_t>(view_row0) * siglip::kQkvWidth + k_col);
    const Element* v_ptr = reinterpret_cast<const Element*>(
        g.qkv_buf + static_cast<int64_t>(view_row0) * siglip::kQkvWidth + v_col);

    auto gQ = make_tensor(
        make_gmem_ptr(q_ptr),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(Int<siglip::kQkvWidth>{}, _1{}));
    auto mK = make_tensor(
        make_gmem_ptr(k_ptr),
        Shape<Int<256>, Int<kHeadDim>>{},
        make_stride(Int<siglip::kQkvWidth>{}, _1{}));
    auto mV = make_tensor(
        make_gmem_ptr(v_ptr),
        Shape<Int<256>, Int<kHeadDim>>{},
        make_stride(Int<siglip::kQkvWidth>{}, _1{}));
    auto gK = local_tile(
        mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    auto gV = local_tile(
        mV, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    auto sQ = make_tensor(make_smem_ptr(storage.q), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(storage.k), SmemLayoutKV{});
    auto sV = make_tensor(make_smem_ptr(storage.v), SmemLayoutKV{});
    auto sVt = make_tensor(make_smem_ptr(storage.v), SmemLayoutVTransposed{});
    auto sVtNoSwizzle =
        make_tensor(make_smem_ptr(storage.v), SmemLayoutVTransposedNoSwizzle{});

    GmemTiledCopy gmem_copy;
    auto gmem_thread_copy = gmem_copy.get_thread_slice(tid);
    auto tQgQ = gmem_thread_copy.partition_S(gQ);
    auto tQsQ = gmem_thread_copy.partition_D(sQ);
    auto tKgK = gmem_thread_copy.partition_S(gK);
    auto tKsK = gmem_thread_copy.partition_D(sK);
    auto tVgV = gmem_thread_copy.partition_S(gV);
    auto tVsV = gmem_thread_copy.partition_D(sV);

    auto cQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto cKV = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    auto tQcQ = gmem_thread_copy.partition_S(cQ);
    auto tKVcKV = gmem_thread_copy.partition_S(cKV);

    TiledMma tiled_mma;
    auto mma_thread = tiled_mma.get_thread_slice(tid);
    auto fragment_q = mma_thread.partition_fragment_A(sQ);
    auto fragment_k = mma_thread.partition_fragment_B(sK);
    auto fragment_v = mma_thread.partition_fragment_B(sVtNoSwizzle);
    auto output_accumulator =
        partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(output_accumulator);

    auto copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto thread_copy_q = copy_q.get_thread_slice(tid);
    auto thread_copy_k = copy_k.get_thread_slice(tid);
    auto thread_copy_v = copy_v.get_thread_slice(tid);
    auto tSsQ = thread_copy_q.partition_S(sQ);
    auto tSsK = thread_copy_k.partition_S(sK);
    auto tOsVt = thread_copy_v.partition_S(sVt);

    copy_gmem_to_smem_padded(gmem_copy, tQgQ, tQsQ, tQcQ);
    cute::cp_async_fence();
    copy_gmem_to_smem_padded(gmem_copy, tKgK(_, _, _, 0), tKsK, tKVcKV);
    cute::cp_async_fence();

    OnlineSoftmax<2 * size<1>(output_accumulator)> softmax;

    #pragma unroll
    for (int kv_block = 0; kv_block < kKvBlocks; ++kv_block) {
        cp_async_wait_all();
        __syncthreads();

        copy_gmem_to_smem_padded(
            gmem_copy, tVgV(_, _, _, kv_block), tVsV, tKVcKV);
        cute::cp_async_fence();

        auto score_accumulator =
            partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(score_accumulator);
        tiled_gemm(
            score_accumulator,
            fragment_q,
            fragment_k,
            tSsQ,
            tSsK,
            tiled_mma,
            copy_q,
            copy_k,
            thread_copy_q,
            thread_copy_k);

        cp_async_wait_all();
        __syncthreads();
        if (kv_block + 1 < kKvBlocks) {
            copy_gmem_to_smem_padded(
                gmem_copy,
                tKgK(_, _, _, kv_block + 1),
                tKsK,
                tKVcKV);
            cute::cp_async_fence();
        }

        if (kv_block == 0) {
            softmax.template update<true>(score_accumulator, output_accumulator);
        } else {
            softmax.template update<false>(score_accumulator, output_accumulator);
        }

        auto probability = convert_fragment<Element>(score_accumulator);
        auto probability_as_a = make_tensor(
            probability.data(),
            accumulator_as_a_layout<TiledMma>(probability.layout()));
        tiled_gemm_register_smem(
            output_accumulator,
            probability_as_a,
            fragment_v,
            tOsVt,
            tiled_mma,
            copy_v,
            thread_copy_v);
    }

    softmax.normalize(output_accumulator);
    auto output_bf16 = convert_fragment<Element>(output_accumulator);

    __syncthreads();
    auto sO = make_tensor(make_smem_ptr(storage.q), SmemLayoutO{});
    auto copy_output_to_smem = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto thread_output_to_smem = copy_output_to_smem.get_thread_slice(tid);
    auto tOrO = thread_output_to_smem.retile_S(output_bf16);
    auto tOsO = thread_output_to_smem.partition_D(sO);
    cute::copy(copy_output_to_smem, tOrO, tOsO);
    __syncthreads();

    Element* output_ptr = reinterpret_cast<Element*>(
        g.attn_out + static_cast<int64_t>(query_row0) * siglip::kWidthPad + head_col);
    auto gO = make_tensor(
        make_gmem_ptr(output_ptr),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(Int<siglip::kWidthPad>{}, _1{}));
    GmemTiledStore gmem_store;
    auto gmem_store_thread = gmem_store.get_thread_slice(tid);
    auto tOsOGlobal = gmem_store_thread.partition_S(sO);
    auto tOgO = gmem_store_thread.partition_D(gO);
    auto tOrOGlobal = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_store, tOsOGlobal, tOrOGlobal);
    auto cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto tOcO = gmem_store_thread.partition_D(cO);
    copy_smem_to_gmem_valid(
        gmem_store, tOrOGlobal, tOgO, tOcO);

    // The physical head tail starts at a 16-byte-aligned address.  Keep it out
    // of the CuTe register fragment (which raises the persistent root's stack
    // allocation) and replace the old eight scalar BF16 stores with one vector
    // zero transaction per row.
    if (tid < kBlockM) {
        nv_bfloat16* padding = g.attn_out +
            static_cast<int64_t>(query_row0 + tid) * siglip::kWidthPad +
            head_col + siglip::kHeadDim;
        *reinterpret_cast<uint4*>(padding) = make_uint4(0, 0, 0, 0);
    }
}

}  // namespace pi0_a100
