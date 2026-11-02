#pragma once

#include "utils.cuh"

namespace llama_mb {

template <typename Config, typename Globals>
struct lmhead_gemm {
    static constexpr int opcode = OPCODE_LMHEAD;
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

    __device__ static int tma_bytes(const parsed_instruction &, int item) {
        const int second =
            item * 2 + 1 < (VOCAB_SIZE + 63) / 64 ? 1 : 0;
        return (1 + second) * sizeof(weight_tile) +
               sizeof(dependent_stage_t);
    }
    __device__ static void load_static(
        const Globals &g, const parsed_instruction &, int item, int k_iter,
        static_stage_t &stage, kittens::semaphore &arrived) {
        kittens::tma::load_async(stage.weight[0], g.lm_head_weights,
                                 {item * 2, k_iter}, arrived);
        if (item * 2 + 1 < (VOCAB_SIZE + 63) / 64)
            kittens::tma::load_async(stage.weight[1], g.lm_head_weights,
                                     {item * 2 + 1, k_iter}, arrived);
    }
    template <typename State>
    __device__ static void wait_dependency(const Globals &g, State &,
                                           const parsed_instruction &inst) {
        wait_business_dep(g, inst.layer, BAR_RMS_LMHEAD, inst.batch_size, 0);
    }
    __device__ static void load_dependent(
        const Globals &g, const parsed_instruction &, int, int k_iter,
        dependent_stage_t &stage, kittens::semaphore &arrived) {
        kittens::tma::load_async(stage.activation, g.lm_head_rms_out,
                                 {0, k_iter}, arrived);
    }
    __device__ static void zero_accumulator(accumulator_t &) {}
    __device__ static bool wg_active(const parsed_instruction &, int item,
                                     int wg) {
        return item * 2 + wg < (VOCAB_SIZE + 63) / 64;
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
        for (int wg = 0; wg < 2; ++wg) {
            const int tile = item * 2 + wg;
            if (tile >= (VOCAB_SIZE + 63) / 64)
                continue;
            constexpr int ELEMENTS = BATCH_CAPACITY * 64;
            for (int idx = worker * kittens::WARP_THREADS + lane;
                 idx < ELEMENTS; idx += 2 * kittens::WARP_THREADS) {
                const int batch = idx / 64;
                const int row = idx % 64;
                const int vocab = tile * 64 + row;
                if (batch >= inst.batch_size || vocab >= VOCAB_SIZE)
                    continue;
                const float value = out.output[wg][make_int2(row, batch)];
                g.logits.raw_ptr[(long)batch * VOCAB_SIZE + vocab] =
                    __float2bfloat16_rn(g.row_active(batch) ? value
                                                           : -1.0e9f);
            }
        }
    }
    __device__ static void publish(const Globals &g,
                                   const parsed_instruction &inst,
                                   int) {
        publish_business_dep(g, inst.layer, inst.publish_row,
                             inst.publish_slot, 1);
    }
    using pipeline = hopper_gemm::matmul_pipeline<
        Config, Globals, parsed_instruction, lmhead_gemm>;
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
                &Globals::hidden_states, &Globals::lm_head_rms_out,
                &Globals::lm_head_norm_weights, false>(
                g, s, inst, row, inst.layer, BAR_LAYER_READY,
                HIDDEN_DIM / 64, 0, BAR_RMS_LMHEAD, 0);
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
