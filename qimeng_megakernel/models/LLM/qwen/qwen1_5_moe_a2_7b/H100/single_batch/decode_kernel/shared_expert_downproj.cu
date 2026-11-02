#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"

// ============================================================================
// opcode 11 — SharedExpert_DownProj
//
// 132 instructions per layer = 4 split-K x split(128 hidden tiles, 33).
//
// Static instruction layout (9 words)
//   [0] opcode = 11
//   [1] layer_idx            0..23
//   [2] tile_M_idx           == 0
//   [3] tile_N_start_idx     0..127   (output hidden tile)
//   [4] tile_N_end_idx       exclusive
//   [5] split_K_idx          0..3
//   [6] tile_M_granularity   == 1
//   [7] tile_N_granularity   == 16
//   [8] split_K_granularity  == 1408
//
// reads   shared_expert_silu_out[split_K_idx * 1408 .. +1408],
//         shared_expert_down_proj_weights[L]   (K = 5632, cols)
// writes  shared_out[tile] +=                  (4 split-K segments accumulate)
//
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_SHARED_EXPERT_UPGATE_SILU), 0] >= 352
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_SHARED_EXPERT_DOWNPROJ), t] += 1
//                for each (split_K_idx, hidden tile t) — so each slot t reaches 4
//
// PIPELINE — this is the K = 1408 instantiation
//   Geometry: STAGE_PAGES = 3 (512 + 512 + 384), ACTIVE_CONSUMER_WARPS = 11,
//   TOTAL_PAGES = 10, SEM_COUNT = 12.
//
// K=1408 pipeline invariant
//   1. `load_iter` is a TEMPLATE ON THE ST TYPE — the tail page is
//      st_bf<16,384>, not st_bf<16,512>.  Column offset stays
//      `inst.start_reduction_col + 512 * col_idx` (page geometry, not tile
//      width).
//   2. In `store`, matvec_reduce's 5th template argument must be
//      `pipeline::ACTIVE_CONSUMER_WARPS` (11), which bounds the valid partials
//      in the mixed instruction stream.
//   3. In the consumer, guard the activation load with
//      `warpid() < pipeline::ACTIVE_CONSUMER_WARPS` and `zero()` the
//      activations_vec of the inactive warps — but `warp_finish_page` must be
//      executed by ALL 16 warps, and only after `kittens::warp::sync()`.
//
// Numerics: four split-K results accumulate through the bf16 `shared_out`
// buffer, matching the Python VM (PyVM) stage boundary and rounding order.
// ============================================================================

template <typename Config, typename Globals> struct shared_expert_downproj {
    static constexpr int opcode = OPCODE_SHARED_EXPERT_DOWNPROJ;
    static constexpr int prev_opcode = OPCODE_SHARED_EXPERT_UPGATE_SILU;

    // Opcode 10 signals once per 16-element shared-intermediate tile.
    static constexpr int EXPECTED_SHARED_UPGATE = 352;
    static_assert(EXPECTED_SHARED_UPGATE == Globals::shared_intermediate_dim /
                                                Globals::matvec_block_size,
                  "op10 signals one tile block per 16 shared-intermediate rows");

    static constexpr int SPLIT_K_GRANULARITY = Globals::split_k_shared; // 1408

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, split_K_idx;
        int start_reduction_col, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            start_block_idx = instruction[3];
            end_block_idx = instruction[4];
            split_K_idx = instruction[5];
            start_reduction_col = split_K_idx * SPLIT_K_GRANULARITY;
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        // The third stage page is st_bf<16,384>.
        template <typename ST>
        __device__ static inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  ST &weight_chunk, kittens::semaphore &sem) {
            kittens::tma::load_async<kittens::dim::ROW,
                                     kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.shared_expert_down_proj_weights,
                kittens::coord<>{inst.layer_idx,
                                 (inst.start_block_idx + iter) *
                                     Globals::matvec_block_size,
                                 inst.start_reduction_col +
                                     pipeline::COLS_PER_PAGE * col_idx},
                sem);
        }

        __device__ static inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            const int block_idx = inst.start_block_idx + output_idx;
            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<Globals::matvec_block_size> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<Globals::matvec_block_size> *>(
                    output_scratch_start);

            kittens::rv_fl<Globals::matvec_block_size> output_rv;
            // RULE 2: ACTIVE_CONSUMER_WARPS (11), never NUM_CONSUMER_WARPS.
            matvec_reduce<Config, kittens::sv_fl<Globals::matvec_block_size>,
                          kittens::rv_fl<Globals::matvec_block_size>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(output_scratch_start,
                                                           output_rv);
            kittens::warp::sync();
            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                kittens::tma::store_add_async<kittens::cache_policy::EVICT_LAST>(
                    g.shared_out, output_smem_bf, {block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline =
        matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics,
                        SPLIT_K_GRANULARITY>;
    static_assert(pipeline::STAGE_PAGES == 3);
    static_assert(pipeline::LAST_PAGE_COLS == 384);
    static_assert(pipeline::ACTIVE_CONSUMER_WARPS == 11);
    static_assert(pipeline::TOTAL_PAGES == 10);
    static_assert(pipeline::SEM_COUNT == 12);

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
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
            if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            parsed_instruction inst{s};
            const int warpid = kittens::warpid();
            const int activation_page = pipeline::get_activation_page(s);

            if (kittens::laneid() == 0 && warpid == 0) {
                s.wait_page_ready(activation_page);
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                                BARRIER_ROW(prev_opcode), 0}],
                    EXPECTED_SHARED_UPGATE);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            // RULE 3: warps 11..15 contribute nothing but must still hold a
            // zeroed vector (they walk the same pipeline) and must still
            // release the activation page.
            rv_t activations_vec;
            kittens::warp::zero(activations_vec);
            if (warpid < pipeline::ACTIVE_CONSUMER_WARPS) {
                sv_t &activations_smem =
                    reinterpret_cast<sv_t *>(
                        &pipeline::get_activations(s))[warpid];
                kittens::warp::load(
                    activations_smem, g.shared_expert_silu_out,
                    kittens::coord<>{inst.start_reduction_col +
                                     warpid * pipeline::REDUCTION_DIM_PER_WARP});
                kittens::warp::sync();
                kittens::warp::load(activations_vec, activations_smem);
            }
            kittens::warp::sync();
            s.warp_finish_page(activation_page, 1);

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                kittens::tma::store_async_wait();
                __threadfence();
                // per-tile slot: each of the 128 slots reaches 4 (one per
                // split-K segment) — that is what op14 spins on.
                for (int t = inst.start_block_idx; t < inst.end_block_idx; t++)
                    atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), t}],
                              1);
            }
        }
    };
};
