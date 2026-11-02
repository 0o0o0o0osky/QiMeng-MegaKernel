#pragma once

#include "utils.cuh"

namespace llama_mb {

template <typename Config, typename Globals>
struct qkv_gemm {
    static constexpr int opcode = OPCODE_QKV;
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
            kittens::tma::load_async(stage.weight[wg], g.qkv_weights,
                                     {inst.layer, out_tile, global_k}, arrived);
        }
    }
    template <typename State>
    __device__ static void wait_dependency(const Globals &g, State &,
                                           const parsed_instruction &inst) {
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int batch_tile = inst.begin % batch_tiles;
        const int rows = inst.batch_size - batch_tile * 8;
        wait_business_dep(g, inst.layer, BAR_RMS_ATTN,
                          rows < 8 ? rows : 8, batch_tile);
    }
    __device__ static void load_dependent(
        const Globals &g, const parsed_instruction &inst, int item, int k_iter,
        dependent_stage_t &stage, kittens::semaphore &arrived) {
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int batch_tile = item % batch_tiles;
        const auto &activation = batch_tile == 0 ? g.attn_rms_out_low
                                                 : g.attn_rms_out_high;
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
            const int global_m0 = out_tile * 64 + row0;
            float x0 = out.partial[0][make_int2(row0, local_batch)] +
                       out.partial[1][make_int2(row0, local_batch)];
            float x1 = out.partial[0][make_int2(row0 + 1, local_batch)] +
                       out.partial[1][make_int2(row0 + 1, local_batch)];
            const bool in_batch = batch < inst.batch_size;
            const bool active = in_batch && g.row_active(batch);
            const bool is_q = global_m0 < HIDDEN_DIM;
            const bool is_k = global_m0 >= HIDDEN_DIM &&
                              global_m0 <
                                  HIDDEN_DIM + NUM_KV_HEADS * HEAD_DIM;
            if ((is_q || is_k) && active) {
                const int d0 = global_m0 % HEAD_DIM;
                const int pos = g.position(batch);
                const float co =
                    g.rope_cos.raw_ptr[(long)pos * HEAD_DIM + d0];
                const float si =
                    g.rope_sin.raw_ptr[(long)pos * HEAD_DIM + d0];
                const float y0 = fmaf(-x1, si, x0 * co);
                const float y1 = fmaf(x0, si, x1 * co);
                x0 = y0;
                x1 = y1;
            }
            if (in_batch && is_q) {
                const long base = (long)batch * HIDDEN_DIM + global_m0;
                g.q_post_rope.raw_ptr[base] =
                    __float2bfloat16_rn(active ? x0 : 0.0f);
                g.q_post_rope.raw_ptr[base + 1] =
                    __float2bfloat16_rn(active ? x1 : 0.0f);
            } else if (active) {
                const int segment =
                    global_m0 -
                    (is_k ? HIDDEN_DIM
                          : HIDDEN_DIM + NUM_KV_HEADS * HEAD_DIM);
                const int kv_head = segment / HEAD_DIM;
                const int d0 = segment % HEAD_DIM;
                const int pos = g.position(batch);
                const long native_base =
                    g.kv_index(inst.layer, batch, pos, kv_head, d0);
                auto *native_dst =
                    is_k ? g.k_cache.raw_ptr : g.v_cache.raw_ptr;
                native_dst[native_base] = __float2bfloat16_rn(x0);
                native_dst[native_base + 1] = __float2bfloat16_rn(x1);
                const long decode_base =
                    ((((long)inst.layer * BATCH_CAPACITY + batch) *
                           NUM_KV_HEADS +
                       kv_head) *
                          g.max_seq_len +
                      pos) *
                         HEAD_DIM +
                     d0;
                auto *decode_dst = is_k ? g.k_cache_decode.raw_ptr
                                        : g.v_cache_decode.raw_ptr;
                decode_dst[decode_base] = __float2bfloat16_rn(x0);
                decode_dst[decode_base + 1] = __float2bfloat16_rn(x1);
            }
        }
    }
    __device__ static void publish(const Globals &g,
                                   const parsed_instruction &inst,
                                   int item) {
        const int batch_tiles = inst.batch_size > 8 ? 2 : 1;
        const int out_tile = item / batch_tiles;
        const int batch_tile = item % batch_tiles;
        const int kv_head =
            out_tile < NUM_Q_HEADS
                ? out_tile / GQA_RATIO
                : (out_tile - NUM_Q_HEADS) % NUM_KV_HEADS;
        publish_business_dep(g, inst.layer, BAR_QKV,
                             batch_tile * NUM_KV_HEADS + kv_head, 1);
    }

    using pipeline = hopper_gemm::matmul_pipeline<
        Config, Globals, parsed_instruction, qkv_gemm>;
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
                &Globals::hidden_states, &Globals::attn_rms_out,
                &Globals::attn_norm_weights, true>(
                g, s, inst, row, inst.layer > 0 ? inst.layer - 1 : 0,
                BAR_LAYER_READY, inst.layer > 0 ? HIDDEN_DIM / 64 : 0, 0,
                BAR_RMS_ATTN, row >= 0 ? row / 8 : 0);
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
