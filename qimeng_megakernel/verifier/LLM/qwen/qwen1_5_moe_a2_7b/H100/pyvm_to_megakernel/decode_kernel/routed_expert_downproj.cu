#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"

// opcode 13 — RoutedExpert_DownProj
// 132 instructions per layer = 4 top-k slots x split(128 hidden tiles, 33).
// Opcode 13 accumulates routed GEMV output; opcode 14 owns the combine.
// Static instruction layout (8 words)
//   [0] opcode = 13
//   [1] layer_idx           0..23
//   [2] expert_idx          top-k slot 0..3; resolved to a physical id on device
//   [3] tile_M_idx          == 0
//   [4] tile_N_start_idx    0..127  (output hidden tile)
//   [5] tile_N_end_idx      exclusive
//   [6] tile_M_granularity  == 1
//   [7] tile_N_granularity  == 16
// device-side expert dereference:
//   physical_expert = expert_indices[expert_idx]
//   weights = expert_down_proj_weights_flat[layer_idx * 60 + physical_expert]
//             // [2048, 1408]
//   scale   = expert_weights[expert_idx]   // fp32 60-way softmax probability
// reads   routed_silu_buffer[expert_idx] (K = 1408),
//         expert_down_proj_weights_flat, expert_weights[expert_idx]
// writes  routed_out_f32[tile] += scale * dot   (fp32 accumulator)
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_ROUTED_EXPERT_UPGATE_SILU), 0] >= 352
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_ROUTED_EXPERT_DOWNPROJ), t] += 1
//                for each (slot, hidden tile t) — each slot t reaches 4
// IMPLEMENTATION NOTES
// K = 1408: STAGE_PAGES = 3 (512 + 512 + 384), ACTIVE_CONSUMER_WARPS = 11,
// TOTAL_PAGES = 10 (three pages spare, freed by the loader's spare-lane
// branch), SEM_COUNT = 12.  `start_reduction_col` is 0 — there is no split-K
// here; the member is kept so op11 and op13 share a pipeline_specifics shape.
// K=1408 pipeline invariant:
//   1. `load_iter` is a template on ST (the tail page is st_bf<16,384>).
//   2. `matvec_reduce`'s 5th argument is `pipeline::ACTIVE_CONSUMER_WARPS`.
//      This bounds the valid partials in a mixed instruction stream.
//   3. Warps >= ACTIVE_CONSUMER_WARPS zero their activations_vec and skip the
//      activation load, but ALL 16 warps run `warp_finish_page`, after
//      `warp::sync()`.
// `loader_gmem_wait` waits on opcode 9 so expert-index publication can start
// weight prefetch. The consumer waits on opcode 12 before reading
// `routed_silu_buffer`. Both waits use `megakernel::mk_barrier_wait`.
// The storer repeats the op12 wait before its loop: it reads
// `expert_weights[slot]` (written by op9) and that read must sit behind an
// acquire fence in its own warp.

template <typename Config, typename Globals> struct routed_expert_downproj {
    static constexpr int opcode = OPCODE_ROUTED_EXPERT_DOWNPROJ;

    //: the consumer / storer gate: all 4 x 88 routed gate/up tile blocks.
    static constexpr int WAIT_ROW =
        BARRIER_ROW(OPCODE_ROUTED_EXPERT_UPGATE_SILU);
    static constexpr int EXPECTED_ARRIVAL_COUNT = 352;
    //: the loader gate: router top-k has published expert_indices.
    static constexpr int TOPK_WAIT_ROW = BARRIER_ROW(OPCODE_ROUTER_SOFTMAX_TOPK);
    static constexpr int TOPK_EXPECTED = 1;
    static constexpr int SIGNAL_ROW = BARRIER_ROW(OPCODE_ROUTED_EXPERT_DOWNPROJ);

    static_assert(EXPECTED_ARRIVAL_COUNT ==
                      Globals::num_experts_per_tok *
                          (Globals::moe_intermediate_dim /
                           Globals::matvec_block_size),
                  "op13 expected count drifted from routed upgate tile count");

    static constexpr int REDUCTION_DIM = Globals::moe_intermediate_dim; // 1408
    static constexpr int COLS_PER_PAGE = 512;

    struct parsed_instruction {
        int layer_idx, expert_slot, start_block_idx, end_block_idx;
        int start_reduction_col, iters;
        //: physical expert id — INVALID until `loader_gmem_wait` fills it in.
        int expert_id;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            expert_slot = instruction[2];
            // instruction[3] == tile_M_idx, always 0.
            start_block_idx = instruction[4];
            end_block_idx = instruction[5];
            start_reduction_col = 0; // no split-K on the routed down-proj
            iters = end_block_idx - start_block_idx;
            expert_id = -1; // poison; an unrepaired id TMA-reads out of bounds
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        // ---- the routed-expert hook (utils.cuh `loader_loop`) --------------
        __device__ static inline void
        loader_gmem_wait(megakernel::state<Config> &s, const Globals &g,
                         parsed_instruction &inst) {
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[kittens::coord<>{inst.layer_idx,
                                                            TOPK_WAIT_ROW, 0}],
                TOPK_EXPECTED);

            inst.expert_id =
                (int)g.expert_indices[kittens::coord<>{inst.expert_slot}];
        }

        // MUST be a template on ST: the third stage page is st_bf<16,384>.
        template <typename ST>
        __device__ static inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  ST &weight_chunk, kittens::semaphore &sem) {
            kittens::tma::load_async<kittens::dim::ROW,
                                     kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.expert_down_proj_weights_flat,
                kittens::coord<>{inst.layer_idx * Globals::num_experts +
                                     inst.expert_id,
                                 (inst.start_block_idx + iter) *
                                     Globals::matvec_block_size,
                                 inst.start_reduction_col +
                                     COLS_PER_PAGE * col_idx},
                sem);
        }

        __device__ static inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            const int block_idx = inst.start_block_idx + output_idx;
            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);

            kittens::sv_fl<Globals::matvec_block_size> &output_smem_fl =
                *reinterpret_cast<kittens::sv_fl<Globals::matvec_block_size> *>(
                    output_scratch_start);

            kittens::rv_fl<Globals::matvec_block_size> output_rv;
            // ACTIVE_CONSUMER_WARPS (11), NOT NUM_CONSUMER_WARPS (16).
            matvec_reduce<Config, kittens::sv_fl<Globals::matvec_block_size>,
                          kittens::rv_fl<Globals::matvec_block_size>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(output_scratch_start,
                                                           output_rv);

            // routing weight for this top-k slot (op9, fp32, NOT renormalised)
            const float scale =
                g.expert_weights[kittens::coord<>{inst.expert_slot}];
            kittens::warp::mul(output_rv, output_rv, scale);

            // output_smem_fl aliases the scratch that was just reduced.
            kittens::warp::sync();
            kittens::warp::store(output_smem_fl, output_rv);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                kittens::tma::store_add_async<kittens::cache_policy::EVICT_LAST>(
                    g.routed_out_f32, output_smem_fl,
                    kittens::coord<>{block_idx * Globals::matvec_block_size});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics, REDUCTION_DIM>;
    static_assert(pipeline::STAGE_PAGES == 3);
    static_assert(pipeline::LAST_PAGE_COLS == 384);
    static_assert(pipeline::ACTIVE_CONSUMER_WARPS == 11);
    static_assert(pipeline::TOTAL_PAGES == 10);

    struct controller {
        static __device__ int
        release_lid(const Globals &g, typename Config::instruction_t &instruction,
                    int &query) {
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

            const int activation_page = pipeline::get_activation_page(s);
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                s.wait_page_ready(activation_page);
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[kittens::coord<>{inst.layer_idx,
                                                                WAIT_ROW, 0}],
                    EXPECTED_ARRIVAL_COUNT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            rv_t activations_vec;
            kittens::warp::zero(activations_vec);
            if (kittens::warpid() < pipeline::ACTIVE_CONSUMER_WARPS) {
                sv_t &activations_smem = reinterpret_cast<sv_t *>(
                    &pipeline::get_activations(s))[kittens::warpid()];
                // routed_silu_buffer is [4, 1408]; the row index is the SLOT.
                kittens::warp::load(
                    activations_smem, g.routed_silu_buffer,
                    kittens::coord<>{0, 0, inst.expert_slot,
                                     inst.start_reduction_col +
                                         kittens::warpid() *
                                             pipeline::REDUCTION_DIM_PER_WARP});
                kittens::warp::sync();
                kittens::warp::load(activations_vec, activations_smem);
            }
            // finish_page strictly after warp::sync(), and by ALL 16 warps.
            kittens::warp::sync();
            s.warp_finish_page(activation_page, 1);

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (kittens::laneid() == 0) {
                // Same gate the consumer takes.  It is already (or about to be)
                // satisfied, and it gives this warp its own acquire fence before
                // `store()` reads expert_weights[slot].
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[kittens::coord<>{inst.layer_idx,
                                                                WAIT_ROW, 0}],
                    EXPECTED_ARRIVAL_COUNT);
            }
            kittens::warp::sync();

            pipeline::storer_loop(s, g);

            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                kittens::tma::store_async_wait();
                // Device-scope release fence before signalling: op14 spins on
                // these per-tile slots and then reads routed_out_f32.
                __threadfence();
                // One slot per hidden tile this instruction produced; each slot
                // reaches 4 (one per top-k slot).
                for (int t = 0; t < inst.iters; t++) {
                    atomicAdd(&g.Bar[kittens::coord<>{
                                  inst.layer_idx, SIGNAL_ROW,
                                  inst.start_block_idx + t}],
                              1);
                }
            }
        }
    };
};
