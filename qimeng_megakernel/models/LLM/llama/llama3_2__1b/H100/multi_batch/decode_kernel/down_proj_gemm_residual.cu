#pragma once

#include "utils.cuh"

namespace llama_mb {

template <typename Config, typename Globals>
struct down_proj_gemm {
    static constexpr int opcode = OPCODE_DOWN;
    using parsed_instruction = range_instruction<Config>;
    static constexpr int STAGES = 15;
    static constexpr int K_ITERS =
        INTERMEDIATE_DIM / DOWN_SPLIT_K / K_TILE;
    static constexpr int OUTPUT_SLOTS = 1;
    static constexpr int IN_FLIGHT_GROUPS = 2;
    static constexpr int ACTIVE_CONSUMER_WARPGROUPS = 1;
    using weight_tile = kittens::st_bf<64, K_TILE>;
    using activation_tile = kittens::st_bf<16, K_TILE>;
    using output_tile = kittens::st_fl<64, 16>;
    using accumulator_t = kittens::rt_fl<16, 16>;

    struct alignas(128) static_stage_t { weight_tile weight; };
    struct alignas(128) dependent_stage_t { activation_tile activation; };
    struct alignas(128) output_stage_t { output_tile output; };

    static_assert(sizeof(static_stage_t) == 8192);
    static_assert(sizeof(dependent_stage_t) == 2048);
    static_assert(sizeof(output_stage_t) == 4096);

    __device__ static int tma_bytes(const parsed_instruction &, int) {
        return sizeof(static_stage_t) + sizeof(dependent_stage_t);
    }
    __device__ static void load_static(
        const Globals &g, const parsed_instruction &inst, int item, int k_iter,
        static_stage_t &stage, kittens::semaphore &arrived) {
        const int out_tile = item / DOWN_SPLIT_K;
        const int split = item % DOWN_SPLIT_K;
        const int global_k = split * K_ITERS + k_iter;
        kittens::tma::load_async(stage.weight, g.down_proj_weights,
                                 {inst.layer, out_tile, global_k}, arrived);
    }
    template <typename State>
    __device__ static void wait_dependency(const Globals &g, State &,
                                           const parsed_instruction &inst) {
        const int split = inst.begin % DOWN_SPLIT_K;
        wait_business_dep(g, inst.dep_layer, inst.dep_row, inst.dep_expected,
                          split);
    }
    __device__ static void load_dependent(
        const Globals &g, const parsed_instruction &, int item, int k_iter,
        dependent_stage_t &stage, kittens::semaphore &arrived) {
        const int split = item % DOWN_SPLIT_K;
        const int global_k = split * K_ITERS + k_iter;
        kittens::tma::load_async(stage.activation, g.silu_out,
                                 {0, global_k}, arrived);
    }
    __device__ static void zero_accumulator(accumulator_t &) {}
    __device__ static bool wg_active(const parsed_instruction &, int, int wg) {
        return wg == 0;
    }
    __device__ static void issue(
        accumulator_t &accum, const static_stage_t &stat,
        const dependent_stage_t &dep, const parsed_instruction &, int, int,
        int, bool accumulate) {
        if (accumulate)
            kittens::warpgroup::mma_ABt<accumulator_t, weight_tile,
                                        activation_tile, 0, 1>(
                accum, stat.weight, dep.activation);
        else
            kittens::warpgroup::mm_ABt(accum, stat.weight, dep.activation);
    }
    __device__ static void write_output(
        output_stage_t &out, const accumulator_t &accum,
        const parsed_instruction &, int, int wg, bool active) {
        if (active && wg == 0)
            kittens::warpgroup::store(out.output, accum);
    }
    template <typename State>
    __device__ static void store(const Globals &g, State &,
                                 const parsed_instruction &inst, int item,
                                 const output_stage_t &out, int worker) {
        const int lane = kittens::laneid();
        const int out_tile = item / DOWN_SPLIT_K;
        const int split = item % DOWN_SPLIT_K;
        constexpr int ELEMENTS = BATCH_CAPACITY * 64;
        for (int idx = worker * kittens::WARP_THREADS + lane;
             idx < ELEMENTS; idx += 2 * kittens::WARP_THREADS) {
            const int batch = idx / 64;
            const int row = idx % 64;
            const long dst = ((long)split * BATCH_CAPACITY + batch) *
                                 HIDDEN_DIM +
                             out_tile * 64 + row;
            g.down_partials.raw_ptr[dst] =
                batch < inst.batch_size && g.row_active(batch)
                    ? out.output[make_int2(row, batch)]
                    : 0.0f;
        }
    }
    __device__ static void publish(const Globals &g,
                                   const parsed_instruction &inst,
                                   int item) {
        publish_business_dep(g, inst.layer, inst.publish_row,
                             item / DOWN_SPLIT_K, 1);
    }

    using pipeline = hopper_gemm::matmul_pipeline<
        Config, Globals, parsed_instruction, down_proj_gemm>;
    struct controller {
        static __device__ int release_lid(
            const Globals &g, typename Config::instruction_t &instruction,
            int &query) {
            return pipeline::template release_lid<0>(g, instruction, query);
        }
        static __device__ int init_semaphores(
            const Globals &, megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };
    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::loader_loop(s, g);
        }
    };
    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };
    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::consumer_loop(s, g);
        }
    };
    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
        }
    };
};

template <typename Config, typename Globals>
struct down_proj_reduction {
    static constexpr int opcode = OPCODE_DOWN_REDUCE;
    using parsed_instruction = range_instruction<Config>;

    struct controller {
        static __device__ int release_lid(
            const Globals &, typename Config::instruction_t &, int &query) {
            return query < 0 || query >= Config::NUM_PAGES ? 0 : query;
        }
        static __device__ int init_semaphores(
            const Globals &, megakernel::state<Config> &) {
            return 0;
        }
    };
    struct loader {
        static __device__ void run(const Globals &,
                                   megakernel::state<Config> &s) {
            return_unused_pages(s);
        }
    };
    struct launcher {
        static __device__ void run(const Globals &,
                                   megakernel::state<Config> &) {}
    };
    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (kittens::warpid() == 0 && kittens::laneid() == 0)
                wait_business_dep(g, inst.dep_layer, inst.dep_row,
                                  inst.dep_expected, inst.begin);
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(14);
            const int tid = kittens::warpid() * kittens::WARP_THREADS +
                            kittens::laneid();
            constexpr int ELEMENTS = BATCH_CAPACITY * 64;
            for (int item = inst.begin; item < inst.end;
                 item += inst.stride) {
                for (int idx = tid; idx < ELEMENTS;
                     idx += Config::NUM_CONSUMER_WARPS *
                            kittens::WARP_THREADS) {
                    const int batch = idx / 64;
                    const int row = idx % 64;
                    if (batch >= inst.batch_size || !g.row_active(batch))
                        continue;
                    const int d = item * 64 + row;
                    const long stride = (long)BATCH_CAPACITY * HIDDEN_DIM;
                    const long base = (long)batch * HIDDEN_DIM + d;
                    float sum = g.down_partials.raw_ptr[base];
#pragma unroll
                    for (int split = 1; split < DOWN_SPLIT_K; ++split)
                        sum += g.down_partials.raw_ptr[base + split * stride];
                    const float residual =
                        static_cast<float>(g.hidden_states.raw_ptr[base]);
                    g.hidden_states.raw_ptr[base] =
                        __float2bfloat16_rn(residual + sum);
                }
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(14);
            if (kittens::warpid() == 0 && kittens::laneid() == 0)
                publish_business_dep(g, inst.layer, inst.publish_row,
                                     inst.publish_slot, range_count(inst));
        }
    };
    struct storer {
        static __device__ void run(const Globals &,
                                   megakernel::state<Config> &) {}
    };
};

} // namespace llama_mb
