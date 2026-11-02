#pragma once

#include "utils.cuh"

namespace llama_mb {

template <typename Config, typename Globals>
struct upgate_gemm {
    static constexpr int opcode = OPCODE_UPGATE;
    using parsed_instruction = range_instruction<Config>;
    static constexpr int STAGES = 10;
    static constexpr int K_ITERS = HIDDEN_DIM / K_TILE;
    static constexpr int OUTPUT_SLOTS = 1;
    static constexpr int IN_FLIGHT_GROUPS = 2;
    static constexpr int ACTIVE_CONSUMER_WARPGROUPS = 2;
    using weight_tile = kittens::st_bf<64, K_TILE>;
    using activation_tile = kittens::st_bf<16, K_TILE>;
    using output_tile = kittens::st_bf<64, 16>;
    using accumulator_t = kittens::rt_fl<16, 16>;

    struct alignas(128) static_stage_t { weight_tile weight[2]; };
    struct alignas(128) dependent_stage_t { activation_tile activation; };
    struct alignas(128) output_stage_t { output_tile output[2]; };

    static_assert(sizeof(static_stage_t) == 16384);
    static_assert(sizeof(dependent_stage_t) == 2048);
    static_assert(sizeof(output_stage_t) == 4096);

    __device__ static int tma_bytes(const parsed_instruction &, int) {
        return sizeof(static_stage_t) + sizeof(dependent_stage_t);
    }
    __device__ static void load_static(
        const Globals &g, const parsed_instruction &inst, int item, int k_iter,
        static_stage_t &stage, kittens::semaphore &arrived) {
        kittens::tma::load_async(stage.weight[0], g.gate_proj_weights,
                                 {inst.layer, item, k_iter}, arrived);
        kittens::tma::load_async(stage.weight[1], g.up_proj_weights,
                                 {inst.layer, item, k_iter}, arrived);
    }
    template <typename State>
    __device__ static void wait_dependency(const Globals &g, State &,
                                           const parsed_instruction &inst) {
        wait_business_dep(g, inst.layer, BAR_RMS_MLP, inst.batch_size, 0);
    }
    __device__ static void load_dependent(
        const Globals &g, const parsed_instruction &, int, int k_iter,
        dependent_stage_t &stage, kittens::semaphore &arrived) {
        kittens::tma::load_async(stage.activation, g.mlp_rms_out,
                                 {0, k_iter}, arrived);
    }
    __device__ static void zero_accumulator(accumulator_t &) {}
    __device__ static bool wg_active(const parsed_instruction &, int, int wg) {
        return wg < 2;
    }
    __device__ static void issue(
        accumulator_t &accum, const static_stage_t &stat,
        const dependent_stage_t &dep, const parsed_instruction &, int, int,
        int wg, bool accumulate) {
        if (accumulate)
            kittens::warpgroup::mma_ABt<accumulator_t, weight_tile,
                                        activation_tile, 0, 1>(
                accum, stat.weight[wg], dep.activation);
        else
            kittens::warpgroup::mm_ABt(accum, stat.weight[wg],
                                       dep.activation);
    }
    __device__ static void write_output(
        output_stage_t &out, const accumulator_t &accum,
        const parsed_instruction &, int, int wg, bool active) {
        if (active)
            kittens::warpgroup::store(out.output[wg], accum);
    }
    template <typename State>
    __device__ static void store(const Globals &g, State &,
                                 const parsed_instruction &inst, int item,
                                 const output_stage_t &out, int worker) {
        const int lane = kittens::laneid();
        constexpr int ELEMENTS = BATCH_CAPACITY * 64;
        for (int idx = worker * kittens::WARP_THREADS + lane;
             idx < ELEMENTS; idx += 2 * kittens::WARP_THREADS) {
            const int batch = idx / 64;
            const int row = idx % 64;
            if (batch >= inst.batch_size)
                continue;
            const float gate = out.output[0][make_int2(row, batch)];
            const float up = out.output[1][make_int2(row, batch)];
            const kittens::bf16 silu =
                __float2bfloat16_rn(gate / (1.0f + __expf(-gate)));
            const float value = static_cast<float>(silu) * up;
            g.silu_out.raw_ptr[(long)batch * INTERMEDIATE_DIM + item * 64 +
                               row] = __float2bfloat16_rn(
                g.row_active(batch) ? value : 0.0f);
        }
    }
    __device__ static void publish(const Globals &g,
                                   const parsed_instruction &inst,
                                   int item) {
        publish_business_dep(g, inst.layer, BAR_UPGATE,
                             item / (INTERMEDIATE_DIM / 64 / DOWN_SPLIT_K),
                             1);
    }

    using pipeline = hopper_gemm::matmul_pipeline<
        Config, Globals, parsed_instruction, upgate_gemm>;
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
            parsed_instruction inst{s};
            const int row = inst.begin < inst.batch_size ? inst.begin : -1;
            run_rms_norm_row<
                Config, Globals, parsed_instruction,
                &Globals::hidden_states, &Globals::mlp_rms_out,
                &Globals::mlp_norm_weights, true>(
                g, s, inst, row, inst.layer, BAR_OPROJ, HIDDEN_DIM / 64,
                row >= 0 ? row / 8 : 0, BAR_RMS_MLP, 0);
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
