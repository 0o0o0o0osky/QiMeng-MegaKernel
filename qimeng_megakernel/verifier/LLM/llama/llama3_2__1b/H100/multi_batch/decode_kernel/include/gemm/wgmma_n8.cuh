#pragma once

#include "kittens.cuh"

namespace qimeng_megakernel::hopper::gemm {

struct wgmma_n8_accumulator {
    float value[4];
};

template <bool Fence, bool Accumulate, typename WeightTile,
          typename ActivationTile>
__device__ __forceinline__ void
wgmma_n8_ABt(wgmma_n8_accumulator &accum, const WeightTile &weight,
             const ActivationTile &activation) {
    static_assert(WeightTile::rows == 64 && WeightTile::cols == 64);
    static_assert(ActivationTile::rows >= 8 && ActivationTile::cols == 64);
    kittens::st_descriptor<WeightTile, 0> weight_desc(weight);
    kittens::st_descriptor<ActivationTile, 0> activation_desc(activation);
    asm volatile("" : "+f"(accum.value[0]), "+f"(accum.value[1]),
                 "+f"(accum.value[2]), "+f"(accum.value[3])
                 :
                 : "memory");
    if constexpr (Fence)
        kittens::warpgroup::mma_fence();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        constexpr int scale_d_base = Accumulate ? 1 : 0;
        const int scale_d = scale_d_base || k != 0;
        const uint64_t a_desc = weight_desc.chunk_descriptor(k);
        const uint64_t b_desc = activation_desc.chunk_descriptor(k);
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %6, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n"
            "}\n"
            : "+f"(accum.value[0]), "+f"(accum.value[1]),
              "+f"(accum.value[2]), "+f"(accum.value[3])
            : "l"(a_desc), "l"(b_desc), "r"(scale_d));
    }
    kittens::warpgroup::mma_commit_group();
}

template <typename OutputTile>
__device__ __forceinline__ void
store_wgmma_n8(OutputTile &output, const wgmma_n8_accumulator &accum) {
    static_assert(OutputTile::rows == 64 && OutputTile::cols >= 8);
    const int tid = threadIdx.x % kittens::WARPGROUP_THREADS;
    const int row = (tid / kittens::WARP_THREADS) * 16 +
                    (tid % kittens::WARP_THREADS) / 4;
    const int col = (tid % 4) * 2;
    output[make_int2(row, col)] = accum.value[0];
    output[make_int2(row, col + 1)] = accum.value[1];
    output[make_int2(row + 8, col)] = accum.value[2];
    output[make_int2(row + 8, col + 1)] = accum.value[3];
}

} // namespace qimeng_megakernel::hopper::gemm
