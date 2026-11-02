#pragma once

#include <cfloat>
#include <cuda_bf16.h>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>

#include "common/ae_globals.cuh"
#include "common/attention_tc.cuh"
#include "common/pi0_config.cuh"

namespace pi0_a100 {
namespace ae_attn_tc {

using namespace cute;

using Element = cutlass::bfloat16_t;

constexpr int kBlockM = ae::kRows;
// One instruction owns 64 keys and walks two 32-key shared-memory tiles.
constexpr int kKeysPerBlock = ae::kKSplit / 4;
constexpr int kBlockN = kKeysPerBlock / 2;
constexpr int kTilesPerBlock = kKeysPerBlock / kBlockN;
constexpr int kHeadDim = ae::kHeadDim;
constexpr int kWarps = 4;
constexpr int kThreadsTc = kWarps * 32;
constexpr float kScaleLog2E = 0.0625f * 1.44269504088896340736f;

static_assert(kBlockM == 64 && kKeysPerBlock == 64 && kBlockN == 32 && kHeadDim == 256);
static_assert(kThreadsTc == kThreads);

using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using TiledMma = TiledMMA<
    MmaAtom,
    Layout<Shape<Int<kWarps>, _1, _1>>,
    Tile<Int<16 * kWarps>, _16, _16>>;

constexpr int kBlockKSmem = 64;
constexpr int kSwizzle = 3;
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
using GmemCopyAtomZfill = Copy_Atom<
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;
using GmemTiledCopy = decltype(make_tiled_copy(
    GmemCopyAtom{}, GmemThreadLayout{}, Layout<Shape<_1, _8>>{}));
using GmemTiledCopyZfill = decltype(make_tiled_copy(
    GmemCopyAtomZfill{}, GmemThreadLayout{}, Layout<Shape<_1, _8>>{}));
using GmemStoreAtom =
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
using GmemTiledStore = decltype(make_tiled_copy(
    GmemStoreAtom{}, GmemThreadLayout{}, Layout<Shape<_1, _8>>{}));

struct alignas(128) AttentionSmem {
    Element q[cute::cosize(SmemLayoutQ{})];
    Element k[cute::cosize(SmemLayoutKV{})];
    Element v[cute::cosize(SmemLayoutKV{})];
};

// 32 KiB of Q plus two 16 KiB KV tiles.
static_assert(sizeof(AttentionSmem) == 64 * 1024);

}  // namespace ae_attn_tc

__device__ __noinline__ void ae_attention_partial(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    using namespace ae_attn_tc;
    using namespace mk_attention;

    AttentionSmem& storage = *reinterpret_cast<AttentionSmem*>(smem);
    const int tid = threadIdx.x;
    const int layer = inst[1];
    const int head = inst[2];
    const int partial = inst[3];
    const int first_key = inst[7] * kKeysPerBlock;
    const int num_tiles = inst[8] * kTilesPerBlock;
    const int key_end = min(first_key + inst[8] * kKeysPerBlock, inst[9]);
    const int prefix_len = inst[10];
    const int state_key = inst[11];
    const int head_col = head * kHeadDim;

    const Element* q_ptr = reinterpret_cast<const Element*>(
        g.qkv_buf + head_col);
    auto gQ = make_tensor(
        make_gmem_ptr(q_ptr),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(Int<ae::kQkvWidth>{}, _1{}));

    auto sQ = make_tensor(make_smem_ptr(storage.q), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(storage.k), SmemLayoutKV{});
    auto sV = make_tensor(make_smem_ptr(storage.v), SmemLayoutKV{});
    auto sVt = make_tensor(
        make_smem_ptr(storage.v), SmemLayoutVTransposed{});
    auto sVtNoSwizzle = make_tensor(
        make_smem_ptr(storage.v), SmemLayoutVTransposedNoSwizzle{});

    GmemTiledCopy gmem_copy;
    GmemTiledCopyZfill gmem_copy_zfill;
    auto gmem_thread_copy = gmem_copy.get_thread_slice(tid);
    auto tQgQ = gmem_thread_copy.partition_S(gQ);
    auto tQsQ = gmem_thread_copy.partition_D(sQ);
    auto tKsK = gmem_thread_copy.partition_D(sK);
    auto tVsV = gmem_thread_copy.partition_D(sV);

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

    auto load_k_async = [&](int key_row0) {
        const Element* pointer = reinterpret_cast<const Element*>(
            g.k_cache +
            (static_cast<int64_t>(layer) * g.kv_rows + key_row0) * kHeadDim);
        auto tile = make_tensor(
            make_gmem_ptr(pointer),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            LayoutRight{});
        auto source = gmem_thread_copy.partition_S(tile);
        cute::copy(gmem_copy, source, tKsK);
        cute::cp_async_fence();
    };
    auto load_v_async = [&](int key_row0) {
        const Element* pointer = reinterpret_cast<const Element*>(
            g.v_cache +
            (static_cast<int64_t>(layer) * g.kv_rows + key_row0) * kHeadDim);
        auto tile = make_tensor(
            make_gmem_ptr(pointer),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            LayoutRight{});
        auto source = gmem_thread_copy.partition_S(tile);
        if (key_row0 + kBlockN <= key_end) {
            cute::copy(gmem_copy, source, tVsV);
        } else {
            auto coordinates = make_identity_tensor(
                Shape<Int<kBlockN>, Int<kHeadDim>>{});
            auto thread_coordinates =
                gmem_thread_copy.partition_S(coordinates);
            auto predicate = make_tensor<bool>(make_shape(
                size<1>(tVsV), size<2>(tVsV)));
            CUTE_UNROLL
            for (int row = 0; row < size<0>(predicate); ++row) {
                CUTE_UNROLL
                for (int col = 0; col < size<1>(predicate); ++col) {
                    predicate(row, col) =
                        key_row0 + get<0>(thread_coordinates(0, row, col)) <
                        key_end;
                }
            }
            cute::copy_if(gmem_copy_zfill, predicate, source, tVsV);
        }
        cute::cp_async_fence();
    };

    cute::copy(gmem_copy, tQgQ, tQsQ);
    cute::cp_async_fence();
    load_k_async(first_key);

    OnlineSoftmax<2 * size<1>(output_accumulator)> softmax;
    auto score_coordinates =
        make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto thread_score_coordinates = mma_thread.partition_C(score_coordinates);

    for (int block = 0; block < num_tiles; ++block) {
        const int key_row0 = first_key + block * kBlockN;
        cp_async_wait_all();
        __syncthreads();

        load_v_async(key_row0);

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
        if (block + 1 < num_tiles) {
            load_k_async(key_row0 + kBlockN);
        }

        #pragma unroll
        for (int value = 0; value < size(score_accumulator); ++value) {
            const int row = get<0>(thread_score_coordinates(value));
            const int key = key_row0 + get<1>(thread_score_coordinates(value));
            const bool visible = key < key_end &&
                (key < prefix_len ||
                 (key >= state_key && (row != 0 || key == state_key)));
            if (!visible) score_accumulator(value) = -INFINITY;
        }

        if (block == 0) {
            softmax.template update<true>(
                score_accumulator, output_accumulator, kScaleLog2E);
        } else {
            softmax.template update<false>(
                score_accumulator, output_accumulator, kScaleLog2E);
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

    Element* partial_ptr = reinterpret_cast<Element*>(
        g.attn_partial +
        (static_cast<int64_t>(partial) * ae::kRows) *
            (ae::kHeads * ae::kHeadDim) + head_col);
    auto gO = make_tensor(
        make_gmem_ptr(partial_ptr),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(Int<ae::kHeads * ae::kHeadDim>{}, _1{}));
    GmemTiledStore gmem_store;
    auto gmem_store_thread = gmem_store.get_thread_slice(tid);
    auto tOsOGlobal = gmem_store_thread.partition_S(sO);
    auto tOgO = gmem_store_thread.partition_D(gO);
    auto tOrOGlobal = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_store, tOsOGlobal, tOrOGlobal);
    cute::copy(gmem_store, tOrOGlobal, tOgO);

    auto output_coordinates =
        make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto thread_output_coordinates = mma_thread.partition_C(output_coordinates);
    auto row_coordinates = logical_divide(
        thread_output_coordinates, Shape<_2>{})(make_coord(0, _), _, 0);
    if (get<1>(row_coordinates(0)) == 0) {
        const int64_t stats_base =
            static_cast<int64_t>(partial) * 2 * ae::kHeads * ae::kRows;
        #pragma unroll
        for (int row_item = 0; row_item < size(softmax.row_max); ++row_item) {
            const int row = get<0>(row_coordinates(row_item));
            g.attn_stats[
                stats_base + static_cast<int64_t>(head) * ae::kRows + row] =
                softmax.row_max(row_item) * 0.0625f;
            g.attn_stats[
                stats_base +
                static_cast<int64_t>(ae::kHeads + head) * ae::kRows + row] =
                softmax.row_sum(row_item);
        }
    }
}

__device__ __noinline__ void ae_attention_reduce(
    const AeGlobals& g, const int32_t* inst, unsigned char* smem) {
    (void)smem;
    const int head = inst[2];
    constexpr int kRowsPerInstruction = 8;
    constexpr int kMaxSplits = 8;
    constexpr int kVec = 8;  // one uint4 of bf16
    static_assert(ae::kHeadDim % (32 * kVec) == 0);
    const int row0 = inst[3] * kRowsPerInstruction;
    const int splits = inst[7];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kColPasses = ae::kHeadDim / (32 * kVec);
    const int col0 = lane * kVec;

    // One warp covers a 256-wide row with eight BF16 values per lane.
    for (int local = warp; local < kRowsPerInstruction; local += 4) {
        const int row = row0 + local;
        float local_max[kMaxSplits] = {};
        float local_sum[kMaxSplits] = {};
        #pragma unroll
        for (int split = 0; split < kMaxSplits; ++split) {
            if (split < splits) {
                const int64_t stats_base =
                    static_cast<int64_t>(split) * 2 * ae::kHeads * ae::kRows;
                local_max[split] = g.attn_stats[
                    stats_base + static_cast<int64_t>(head) * ae::kRows + row];
                local_sum[split] = g.attn_stats[
                    stats_base +
                    static_cast<int64_t>(ae::kHeads + head) * ae::kRows + row];
            }
        }
        float maximum = -FLT_MAX;
        #pragma unroll
        for (int split = 0; split < kMaxSplits; ++split) {
            if (split < splits) maximum = fmaxf(maximum, local_max[split]);
        }
        float weights[kMaxSplits] = {};
        float denominator = 0.0f;
        #pragma unroll
        for (int split = 0; split < kMaxSplits; ++split) {
            if (split < splits) {
                weights[split] =
                    local_sum[split] * __expf(local_max[split] - maximum);
                denominator += weights[split];
            }
        }
        const float inv = 1.0f / denominator;

        #pragma unroll
        for (int pass = 0; pass < kColPasses; ++pass) {
            const int col = col0 + pass * 32 * kVec;
            uint4 raw[kMaxSplits];
            #pragma unroll
            for (int split = 0; split < kMaxSplits; ++split) {
                if (split < splits) {
                    const nv_bfloat16* src = g.attn_partial +
                        (static_cast<int64_t>(split) * ae::kRows + row) *
                            (ae::kHeads * ae::kHeadDim) +
                        head * ae::kHeadDim + col;
                    raw[split] = *reinterpret_cast<const uint4*>(src);
                }
            }
            float value[kVec] = {};
            #pragma unroll
            for (int split = 0; split < kMaxSplits; ++split) {
                if (split < splits) {
                    const nv_bfloat16* packed =
                        reinterpret_cast<const nv_bfloat16*>(&raw[split]);
                    #pragma unroll
                    for (int j = 0; j < kVec; ++j) {
                        value[j] = fmaf(
                            weights[split],
                            __bfloat162float(packed[j]),
                            value[j]);
                    }
                }
            }
            uint4 out;
            nv_bfloat16* packed_out = reinterpret_cast<nv_bfloat16*>(&out);
            #pragma unroll
            for (int j = 0; j < kVec; ++j) {
                packed_out[j] = __float2bfloat16(value[j] * inv);
            }
            *reinterpret_cast<uint4*>(
                g.attn_out +
                (static_cast<int64_t>(head) * ae::kRows + row) * ae::kHeadDim +
                col) = out;
        }
    }
}

}  // namespace pi0_a100
