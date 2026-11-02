#pragma once

#include "utils.cuh"

namespace llama_mb {

template <typename Config, typename Globals>
struct oproj_gemm {
    static constexpr int opcode = OPCODE_OPROJ;
    using parsed_instruction = range_instruction<Config>;
    static constexpr int STAGES = 8;
    static constexpr int K_ITERS = HIDDEN_DIM / K_TILE / 2;
    static constexpr int OUTPUT_SLOTS = 1;
    static constexpr int IN_FLIGHT_GROUPS = 2;
    static constexpr int ACTIVE_CONSUMER_WARPGROUPS = 2;
    using weight_tile = kittens::st_bf<64, K_TILE>;
    using activation_tile = kittens::st_bf<16, K_TILE>;
    using output_tile = kittens::st_fl<64, 16>;
    using accumulator_t = hopper_gemm::wgmma_n8_accumulator;

    struct alignas(128) static_stage_t { weight_tile weight[2]; };
    struct alignas(128) dependent_stage_t {
        activation_tile activation[2];
    };
    struct alignas(128) output_stage_t { output_tile partial[2]; };

    static_assert(sizeof(static_stage_t) == 16384);
    static_assert(sizeof(dependent_stage_t) == 4096);
    static_assert(sizeof(output_stage_t) == 8192);

    __device__ static int tma_bytes(const parsed_instruction &, int) {
        return sizeof(static_stage_t) + sizeof(dependent_stage_t);
    }
    __device__ static void load_static(
        const Globals &g, const parsed_instruction &inst, int item, int k_iter,
        static_stage_t &stage, kittens::semaphore &arrived) {
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int out_tile = item / batch_tiles;
#pragma unroll
        for (int wg = 0; wg < 2; ++wg) {
            const int global_k = wg * K_ITERS + k_iter;
            kittens::tma::load_async(stage.weight[wg], g.o_proj_weights,
                                     {inst.layer, out_tile, global_k}, arrived);
        }
    }
    template <typename State>
    __device__ static void wait_dependency(const Globals &g, State &,
                                           const parsed_instruction &inst) {
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int batch_tile = inst.begin % batch_tiles;
        const int rows = inst.batch_size - batch_tile * 8;
        const int active_rows = rows < 8 ? rows : 8;
        wait_business_dep(g, inst.dep_layer, inst.dep_row,
                          active_rows * NUM_KV_HEADS, batch_tile);
    }
    __device__ static void load_dependent(
        const Globals &g, const parsed_instruction &inst, int item, int k_iter,
        dependent_stage_t &stage, kittens::semaphore &arrived) {
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int batch_tile = item % batch_tiles;
        const auto &activation =
            batch_tile == 0 ? g.attn_out_low : g.attn_out_high;
#pragma unroll
        for (int wg = 0; wg < 2; ++wg) {
            const int global_k = wg * K_ITERS + k_iter;
            kittens::tma::load_async(stage.activation[wg], activation,
                                     {0, global_k}, arrived);
        }
    }
    __device__ static void zero_accumulator(accumulator_t &accum) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            accum.value[i] = 0.0f;
    }
    __device__ static bool wg_active(const parsed_instruction &, int, int wg) {
        return wg < 2;
    }
    __device__ static void issue(
        accumulator_t &accum, const static_stage_t &stat,
        const dependent_stage_t &dep, const parsed_instruction &, int, int,
        int wg, bool accumulate) {
        if (accumulate)
            hopper_gemm::wgmma_n8_ABt<false, true>(
                accum, stat.weight[wg], dep.activation[wg]);
        else
            hopper_gemm::wgmma_n8_ABt<true, false>(
                accum, stat.weight[wg], dep.activation[wg]);
    }
    __device__ static void write_output(
        output_stage_t &out, const accumulator_t &accum,
        const parsed_instruction &, int, int wg, bool active) {
        if (active)
            hopper_gemm::store_wgmma_n8(out.partial[wg], accum);
    }
    template <typename State>
    __device__ static void store(const Globals &g, State &,
                                 const parsed_instruction &inst, int item,
                                 const output_stage_t &out, int worker) {
        const int lane = kittens::laneid();
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int out_tile = item / batch_tiles;
        const int batch_tile = item % batch_tiles;
        constexpr int PAIRS = 8 * 64 / 2;
        for (int pair = worker * kittens::WARP_THREADS + lane;
             pair < PAIRS; pair += 2 * kittens::WARP_THREADS) {
            const int local_batch = pair / 32;
            const int row0 = (pair % 32) * 2;
            const int batch = batch_tile * 8 + local_batch;
            if (batch >= inst.batch_size || !g.row_active(batch))
                continue;
            const float value0 =
                out.partial[0][make_int2(row0, local_batch)] +
                out.partial[1][make_int2(row0, local_batch)];
            const float value1 =
                out.partial[0][make_int2(row0 + 1, local_batch)] +
                out.partial[1][make_int2(row0 + 1, local_batch)];
            const long base = (long)batch * HIDDEN_DIM +
                              out_tile * 64 + row0;
            const float residual0 =
                static_cast<float>(g.hidden_states.raw_ptr[base]);
            const float residual1 =
                static_cast<float>(g.hidden_states.raw_ptr[base + 1]);
            g.hidden_states.raw_ptr[base] =
                __float2bfloat16_rn(residual0 + value0);
            g.hidden_states.raw_ptr[base + 1] =
                __float2bfloat16_rn(residual1 + value1);
        }
    }
    __device__ static void publish(const Globals &g,
                                   const parsed_instruction &inst,
                                   int item) {
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int batch_tile = item % batch_tiles;
        publish_business_dep(g, inst.layer, inst.publish_row, batch_tile, 1);
    }

    using pipeline = hopper_gemm::matmul_pipeline<
        Config, Globals, parsed_instruction, oproj_gemm>;
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

} // namespace llama_mb
