#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"

// ============================================================================
// opcode 4 — OProj_GEMV_Residual
//
// 128 instructions per layer (2048 / 16 = 128 output blocks).
//
// Static instruction layout (8 words)
//   [0] opcode = 4
//   [1] layer_idx            0..23
//   [2] tile_M_idx           == 0
//   [3] tile_N_start_idx     0..127
//   [4] tile_N_end_idx       exclusive
//   [5] reduction_block_idx  == 0
//   [6] tile_M_granularity   == 1
//   [7] tile_N_granularity   == 16
//
// reads   attn_out, o_proj_weights[L]
// writes  hidden_states[tile] += (residual add, bf16)
//
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_DECODE_ATTENTION_REDUCTION), 0] >= 16
//                (16 == num_attention_heads; op3 signals +4 four times)
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_OPROJ_GEMV_RESIDUAL), 0] += 1
//                per tile block
//
// pipeline
//   matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>
//   (no RMSNorm here; activations are attn_out.  K = hidden_dim = 2048.)
// ============================================================================

using namespace kittens;
using namespace megakernel;

template <typename Config, typename Globals> struct oproj_gemv_residual {
    static constexpr int opcode = OPCODE_OPROJ_GEMV_RESIDUAL;
    static constexpr int prev_opcode = OPCODE_DECODE_ATTENTION_REDUCTION;
    // One arrival per attention head.
    static constexpr int EXPECTED_ARRIVAL_COUNT = 16;

    static constexpr int BLK = Globals::matvec_block_size; // 16

    static_assert(EXPECTED_ARRIVAL_COUNT == Globals::num_attention_heads,
                  "attention reduction must signal once per head");

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, reduction_block_idx,
            start_reduction_col, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            // instruction[2] == tile_M_idx, always 0
            start_block_idx = instruction[3];
            end_block_idx = instruction[4];
            reduction_block_idx = instruction[5]; // always 0 for this model
            start_reduction_col = reduction_block_idx * Globals::hidden_dim;
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            kittens::tma::load_async<kittens::dim::ROW,
                                     kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.o_proj_weights,
                kittens::coord<>{inst.layer_idx,
                                 (inst.start_block_idx + iter) * BLK,
                                 inst.start_reduction_col + 512 * col_idx},
                sem);
        }

        static __device__ inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            const int block_idx = inst.start_block_idx + output_idx;
            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<BLK> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<BLK> *>(output_scratch_start);

            kittens::rv_fl<BLK> output_rv;
            matvec_reduce<Config, kittens::sv_fl<BLK>, kittens::rv_fl<BLK>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(
                output_scratch_start, output_rv);
            kittens::warp::sync();
            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                // Residual add.  Every hidden tile is written by exactly one
                // instruction, so this TMA reduction has no contention and the
                // result is bit-reproducible.
                kittens::tma::store_add_async<kittens::cache_policy::EVICT_LAST>(
                    g.hidden_states, output_smem_bf, {block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics>;

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

            const int activation_page = pipeline::get_activation_page(s);
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                s.wait_page_ready(activation_page);
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                                BARRIER_ROW(prev_opcode), 0}],
                    EXPECTED_ARRIVAL_COUNT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            sv_t &activations_smem =
                reinterpret_cast<sv_t *>(
                    &pipeline::get_activations(s))[kittens::warpid()];
            kittens::warp::load(
                activations_smem, g.attn_out,
                kittens::coord<>{inst.start_reduction_col +
                                 kittens::warpid() *
                                     pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();
            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            // finish_page strictly after warp::sync(): the releasing lane must
            // not outrun its own warp's smem reads.
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
                // Device-scope release fence before signalling the cross-CTA
                // barrier: hidden_states must be visible to the router CTAs.
                __threadfence();
                atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), 0}],
                          inst.iters);
            }
        }
    };
};
