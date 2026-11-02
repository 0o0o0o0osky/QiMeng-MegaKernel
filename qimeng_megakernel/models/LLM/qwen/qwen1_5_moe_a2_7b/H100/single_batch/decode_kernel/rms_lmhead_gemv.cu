#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"

// ============================================================================
// opcode 7 — RMS_LMHead_GEMV
//
// 132 instructions, ONCE per decode step (appended after the last layer).
// 151936 / 16 = 9496 output blocks split over 132 SMs.
//
// Static instruction layout (5 words; layer is fixed to the terminal layer)
//   [0] opcode = 7
//   [1] start_output_block_idx  0..9495
//   [2] end_output_block_idx    exclusive
//   [3] tile_M_granularity      == 1
//   [4] tile_N_granularity      == 16
//
// reads   hidden_states, lm_head_norm_weights, lm_head_weights
// writes  logits  (bf16, [151936])
//
// barrier WAIT   Bar[NUM_LAYERS - 1, BARRIER_ROW(OPCODE_MOE_COMBINE_COMMIT), 0]
//                  >= 128            // layer index is hard-coded to 23
// barrier SIGNAL none — op7 is terminal (row 6 of the barrier tensor stays 0).
//
// pipeline
//   rms_matvec_pipeline<..., &Globals::hidden_states,
//                            &Globals::lm_head_norm_weights>
//   lm_head_norm_weights is 1-D [2048]; it arrives as {1,1,1,2048}, so the
//   TMA row index for the RMS scale is 0, not a layer index.
// ============================================================================

using namespace kittens;
using namespace megakernel;

template <typename Config, typename Globals> struct rms_lmhead_gemv {
    static constexpr int opcode = OPCODE_RMS_LMHEAD_GEMV;
    static constexpr int prev_opcode = OPCODE_MOE_COMBINE_COMMIT;
    // The LM head starts after all final-layer combine tiles commit.
    static constexpr int EXPECTED_ARRIVAL_COUNT = 128;
    //: op7 carries no layer_idx; it always waits on the LAST layer.
    static constexpr int WAIT_LAYER_IDX = Globals::num_layers - 1;

    static constexpr int BLK = Globals::matvec_block_size; // 16

    struct parsed_instruction {
        int start_block_idx, end_block_idx, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            start_block_idx = instruction[1];
            end_block_idx = instruction[2];
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{WAIT_LAYER_IDX,
                                            BARRIER_ROW(prev_opcode), 0}],
                EXPECTED_ARRIVAL_COUNT);
        }

        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            // lm_head_weights is [151936, 2048] -> {1, 1, 151936, 2048}:
            // depth index 0, row index in elements.
            kittens::tma::load_async<kittens::dim::ROW,
                                     kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.lm_head_weights,
                kittens::coord<>{0, (inst.start_block_idx + iter) * BLK,
                                 512 * col_idx},
                sem);
        }

        static __device__ inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            const int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<BLK> &logits_smem_bf =
                *reinterpret_cast<kittens::sv_bf<BLK> *>(output_scratch_start);

            kittens::rv_fl<BLK> logits_rv;
            matvec_reduce<Config, kittens::sv_fl<BLK>, kittens::rv_fl<BLK>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(
                output_scratch_start, logits_rv);

            kittens::warp::sync();
            kittens::warp::store(logits_smem_bf, logits_rv);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                    g.logits, logits_smem_bf, {0, 0, 0, block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline =
        rms_matvec_pipeline<Config, Globals, parsed_instruction,
                            pipeline_specifics, &Globals::hidden_states,
                            &Globals::lm_head_norm_weights>;

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
            // lm_head_norm_weights is [2048] -> {1,1,1,2048}: TMA row 0.
            pipeline::loader_loop(s, g, 0);
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
            // Terminal op: nothing signals afterwards, so this warp is the last
            // guarantee that `logits` actually reached global memory before the
            // kernel retires.  `store_async_read_wait` inside store() only
            // covers the smem source.
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                kittens::tma::store_async_wait();
            }
        }
    };
};
