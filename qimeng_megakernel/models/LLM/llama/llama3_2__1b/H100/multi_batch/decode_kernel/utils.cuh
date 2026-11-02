#pragma once

#include "instruction_list.cuh"
#include "include/gemm/matmul_pipeline.cuh"
#include "include/gemm/wgmma_n8.cuh"

namespace llama_mb {

namespace hopper_gemm = qimeng_megakernel::hopper::gemm;

template <typename Globals>
__device__ __forceinline__ void wait_business_dep(
    const Globals &g, int layer, int row, int expected, int slot) {
    if (expected <= 0)
        return;
    megakernel::mk_barrier_wait(
        reinterpret_cast<volatile int32_t *>(
            &g.Bar[kittens::coord<>{layer, row, slot}]),
        expected);
}

template <typename Globals>
__device__ __forceinline__ void publish_business_dep(
    const Globals &g, int layer, int row, int slot, int count) {
    if (count <= 0)
        return;
    __threadfence();
    atomicAdd(reinterpret_cast<uint32_t *>(
                  &g.Bar[kittens::coord<>{layer, row, slot}]),
              static_cast<uint32_t>(count));
}

template <typename Config>
__device__ __forceinline__ int range_count(
    const range_instruction<Config> &inst) {
    if (inst.stride <= 0 || inst.begin >= inst.end)
        return 0;
    return (inst.end - inst.begin + inst.stride - 1) /
           inst.stride;
}

template <typename Config>
__device__ __forceinline__ void return_unused_pages(
    megakernel::state<Config> &s) {
    if (kittens::laneid() < Config::NUM_PAGES) {
        const int pid = s.pid(kittens::laneid());
        s.wait_page_ready(pid);
        s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
    }
}

template <typename Config, typename Globals, typename ParsedInstruction,
          auto InputPtr, auto OutputPtr, auto WeightPtr, bool LayeredWeight>
__device__ __forceinline__ void run_rms_norm_row(
    const Globals &g, megakernel::state<Config> &s,
    const ParsedInstruction &inst, int row, int dep_layer, int dep_row,
    int dep_expected, int dep_slot, int publish_row, int publish_slot) {
    if (row < 0 || row >= inst.batch_size)
        return;
    if (kittens::warpid() == 0 && kittens::laneid() == 0)
        wait_business_dep(g, dep_layer, dep_row, dep_expected, dep_slot);
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(12);

    const auto &input = g.*InputPtr;
    const auto &output = g.*OutputPtr;
    const auto &weight = g.*WeightPtr;
    float *scratch = reinterpret_cast<float *>(s.scratch());
    const int tid = kittens::warpid() * kittens::WARP_THREADS +
                    kittens::laneid();
    const bool active = g.row_active(row);
    float sum = 0.0f;
    if (active) {
        for (int d = tid; d < HIDDEN_DIM;
             d += Config::NUM_CONSUMER_WARPS * kittens::WARP_THREADS) {
            const float x = static_cast<float>(
                input.raw_ptr[(long)row * HIDDEN_DIM + d]);
            sum = fmaf(x, x, sum);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (kittens::laneid() == 0)
        scratch[kittens::warpid()] = sum;
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(12);
    if (tid == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < Config::NUM_CONSUMER_WARPS; ++w)
            total += scratch[w];
        scratch[Config::NUM_CONSUMER_WARPS] =
            active ? rsqrtf(total / static_cast<float>(HIDDEN_DIM) +
                            g.rms_norm_eps)
                   : 0.0f;
    }
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(12);
    const float scale = scratch[Config::NUM_CONSUMER_WARPS];
    for (int d = tid; d < HIDDEN_DIM;
         d += Config::NUM_CONSUMER_WARPS * kittens::WARP_THREADS) {
        float y = 0.0f;
        if (active) {
            const float x = static_cast<float>(
                input.raw_ptr[(long)row * HIDDEN_DIM + d]);
            const kittens::bf16 normalized =
                __float2bfloat16_rn(x * scale);
            const long weight_idx =
                LayeredWeight ? (long)inst.layer * HIDDEN_DIM + d : d;
            y = static_cast<float>(normalized) *
                static_cast<float>(weight.raw_ptr[weight_idx]);
        }
        output.raw_ptr[(long)row * HIDDEN_DIM + d] =
            __float2bfloat16_rn(y);
    }
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(12);
    if (kittens::warpid() == 0 && kittens::laneid() == 0)
        publish_business_dep(g, inst.layer, publish_row, publish_slot, 1);
}

} // namespace llama_mb
