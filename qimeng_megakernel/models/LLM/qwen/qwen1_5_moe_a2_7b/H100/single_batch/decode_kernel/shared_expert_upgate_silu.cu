#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"

// ============================================================================
// opcode 10 — SharedExpert_UpGate_SiLU
//
// 132 instructions per layer; 5632 / 16 = 352 output blocks.
//
// Static instruction layout (8 words)
//   [0] opcode = 10
//   [1] layer_idx           0..23
//   [2] tile_M_idx          == 0
//   [3] tile_N_start_idx    0..351
//   [4] tile_N_end_idx      exclusive
//   [5] init_shared_out     0/1 — exactly one per layer (tile_N_start_idx == 0)
//   [6] tile_M_granularity  == 1
//   [7] tile_N_granularity  == 16
//
// reads   normed_hidden_states published by opcode 8,
//         shared_expert_gate_proj_weights[L], shared_expert_up_proj_weights[L]
// writes  shared_expert_silu_out[tile]  = silu(gate) * up
//         if init_shared_out: zero the whole shared_out[2048] FIRST
//
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_ROUTER_GEMV), 1] >= 1
//                (slot 1 publishes normed_hidden_states after o_proj completes)
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_SHARED_EXPERT_UPGATE_SILU), 0] += 1
//                per tile block  (352 per layer)
//
// notes
//   * Uses a plain K=2048 matvec pipeline over the published normalized input.
//   * Two weight matrices per output tile (gate and up).  Either run two
//     pipeline passes or widen the stage. In either case, `iters` must equal
//     tile_N_end_idx - tile_N_start_idx so release_lid remains correct.
//   * The shared_out zeroing is safe because op11 waits for ALL 352 of op10.
// ============================================================================
//
// IMPLEMENTATION NOTES
// --------------------
// 1. `iters` is 2 x tiles: even iterations fetch `up`, odd iterations fetch `gate`, and
//    `storer_loop<2>` retires the pair.  `release_lid` is therefore driven by
//    the doubled `iters`, which is exactly what `mvp_release_lid_impl` expects
//    (it only needs "how many pipeline iterations does this instruction run").
//
// 2. `init_shared_out` zeroes `shared_out` in the consumer after the global
//    dependency wait. This establishes both required edges:
//      (a) the previous layer's op14 must already have consumed `shared_out`.
//          The consumer's wait on Bar[L, 7, 1] chains op8 -> op4 -> op3 -> op2
//          -> op1 -> Bar[L-1, 13, 0] >= 128.
//      (b) the zeroing must be visible before THIS instruction's tile-block
//          signal.  The consumer fences, then arrives on `outputs_arrived`;
//          the storer only reaches its `atomicAdd` after waiting on that
//          semaphore, so the fence is ordered before the signal.
// ============================================================================

template <typename Config, typename Globals> struct shared_expert_upgate_silu {
    static constexpr int opcode = OPCODE_SHARED_EXPERT_UPGATE_SILU;
    static constexpr int prev_opcode = OPCODE_ROUTER_GEMV;

    // Opcode 8 publishes normalized hidden state once per layer.
    static constexpr int EXPECTED_NORMED_PUBLISHED = 1;
    static constexpr int NORMED_HIDDEN_SLOT = 1;

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, init_shared_out;
        int start_reduction_col, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            start_block_idx = instruction[3];
            end_block_idx = instruction[4];
            init_shared_out = instruction[5];
            start_reduction_col = 0; // full K = hidden_dim
            iters = 2 * (end_block_idx - start_block_idx); // up + gate per tile
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        // K == hidden_dim, so every stage page is a full st_bf<16,512>; the
        // signature is still a template on ST so that it keeps compiling if
        // the page geometry ever changes.
        template <typename ST>
        __device__ static inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  ST &weight_chunk, kittens::semaphore &sem) {
            const int block_idx = inst.start_block_idx + iter / 2;
            const int row = block_idx * Globals::matvec_block_size;
            const int col = inst.start_reduction_col +
                            pipeline::COLS_PER_PAGE * col_idx;
            if (iter % 2 == 0) {
                kittens::tma::load_async<kittens::dim::ROW,
                                         kittens::cache_policy::EVICT_FIRST>(
                    weight_chunk, g.shared_expert_up_proj_weights,
                    kittens::coord<>{inst.layer_idx, row, col}, sem);
            } else {
                kittens::tma::load_async<kittens::dim::ROW,
                                         kittens::cache_policy::EVICT_FIRST>(
                    weight_chunk, g.shared_expert_gate_proj_weights,
                    kittens::coord<>{inst.layer_idx, row, col}, sem);
            }
        }

        __device__ static inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            if (output_idx % 2 == 0) return; // wait for the gate half

            const int block_idx = inst.start_block_idx + output_idx / 2;
            const int prev_output_stage =
                (output_idx - 1) % pipeline::OUTPUT_PIPELINE_STAGES;

            uint8_t *gate_scratch = pipeline::get_output_start(s, output_stage);
            uint8_t *up_scratch =
                pipeline::get_output_start(s, prev_output_stage);

            kittens::sv_bf<Globals::matvec_block_size> &out_smem =
                *reinterpret_cast<kittens::sv_bf<Globals::matvec_block_size> *>(
                    gate_scratch);

            kittens::rv_fl<Globals::matvec_block_size> up_out, gate_out, tmp;
            matvec_reduce<Config, kittens::sv_fl<Globals::matvec_block_size>,
                          kittens::rv_fl<Globals::matvec_block_size>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(up_scratch, up_out);
            matvec_reduce<Config, kittens::sv_fl<Globals::matvec_block_size>,
                          kittens::rv_fl<Globals::matvec_block_size>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(gate_scratch,
                                                           gate_out);

            // silu(gate) * up in FP32, matching the Python VM (PyVM).
            kittens::warp::mul(tmp, gate_out, -1.f);
            kittens::warp::exp(tmp, tmp);
            kittens::warp::add(tmp, tmp, 1.f);
            kittens::warp::div(gate_out, gate_out, tmp);
            kittens::warp::mul(gate_out, gate_out, up_out);

            kittens::warp::sync();
            kittens::warp::store(out_smem, gate_out);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                    g.shared_expert_silu_out, out_smem, {block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics, Globals::hidden_dim>;
    static_assert(pipeline::OUTPUT_PIPELINE_STAGES == 3,
                  "store() hard-codes the prev-stage arithmetic for 3 stages");
    static_assert(pipeline::ACTIVE_CONSUMER_WARPS == Config::NUM_CONSUMER_WARPS,
                  "K == hidden_dim must keep every consumer warp active");

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
                                                BARRIER_ROW(prev_opcode),
                                                NORMED_HIDDEN_SLOT}],
                    EXPECTED_NORMED_PUBLISHED);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            // See note 2: safe here, and ordered before the storer's signal.
            if (inst.init_shared_out) {
                constexpr int F4_PER_WARP =
                    (pipeline::REDUCTION_DIM_PER_WARP *
                     (int)sizeof(kittens::bf16)) /
                    (int)sizeof(float4); // 128 * 2 / 16 = 16
                float4 *dst = (float4 *)g.shared_out.raw_ptr + warpid * F4_PER_WARP;
                if (kittens::laneid() < F4_PER_WARP)
                    dst[kittens::laneid()] = make_float4(0.f, 0.f, 0.f, 0.f);
                __threadfence();
            }

            sv_t &activations_smem =
                reinterpret_cast<sv_t *>(&pipeline::get_activations(s))[warpid];
            kittens::warp::load(
                activations_smem, g.normed_hidden_states,
                kittens::coord<>{warpid * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();
            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();
            s.warp_finish_page(activation_page, 1);

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::template storer_loop<2>(s, g);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                kittens::tma::store_async_wait();
                __threadfence();
                // one signal per tile block (iters counts up+gate pairs)
                atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), 0}],
                          inst.end_block_idx - inst.start_block_idx);
            }
        }
    };
};
