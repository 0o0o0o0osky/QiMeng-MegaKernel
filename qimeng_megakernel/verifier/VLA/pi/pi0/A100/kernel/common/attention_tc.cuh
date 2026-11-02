#pragma once

#include <cmath>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

namespace pi0_a100::mk_attention {

using namespace cute;

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
        OutputTensor& output_accumulator,
        float scale_log2e) {
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
                    exp2f((row_max(row) - maximum) * scale_log2e);
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
            const float maximum_scaled = maximum * scale_log2e;
            #pragma unroll
            for (int col = 0; col < size<1>(scores); ++col) {
                const float probability =
                    exp2f(scores(row, col) * scale_log2e - maximum_scaled);
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
            row_sum(row) = sum;
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

}  // namespace pi0_a100::mk_attention
