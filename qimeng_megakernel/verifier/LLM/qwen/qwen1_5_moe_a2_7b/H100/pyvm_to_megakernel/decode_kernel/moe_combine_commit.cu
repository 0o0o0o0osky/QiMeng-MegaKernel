#pragma once
#include "instruction_list.cuh"

// ============================================================================
// opcode 14 — MoE_Combine_Commit
//
// 128 instructions per layer, one 16-element hidden tile each. Opcode 14 owns
// the shared/routed combine and commits the next hidden state.
//
// Static instruction layout (5 words)
//   [0] opcode = 14
//   [1] layer_idx           0..23
//   [2] tile_N_start_idx    0..127
//   [3] tile_N_end_idx      == tile_N_start_idx + 1  (1 tile per instruction)
//   [4] tile_N_granularity  == 16
//
// consumer sequence
//   1. gate = sigmoid( dot( shared_expert_gate_weights[L], normed_hidden_states ) )
//      2048-wide dot over the native [24, 2048] gate-weight view.
//   2. spin until BOTH
//        Bar[L, BARRIER_ROW(OPCODE_SHARED_EXPERT_DOWNPROJ),  t] >= 4
//        Bar[L, BARRIER_ROW(OPCODE_ROUTED_EXPERT_DOWNPROJ),  t] >= 4
//      for every t in [tile_N_start_idx, tile_N_end_idx).
//   3. hidden_states[i] = bf16( f32(hidden_states[i])
//                             + routed_out_f32[i]
//                             + gate * f32(shared_out[i]) )
//      The final bf16 store performs one rounding after the fp32 combine.
//   4. routed_out_f32[i] = 0   — clears the accumulator for the next layer.
//   5. Bar[L, BARRIER_ROW(OPCODE_MOE_COMBINE_COMMIT), 0] += 1  (per tile block)
//      after tma::store_async_wait().
//
// reads   normed_hidden_states, shared_expert_gate_weights[L],
//         routed_out_f32[tile], shared_out[tile]
// writes  hidden_states[tile] +=, routed_out_f32[tile] = 0
//
// downstream: op1 of layer L+1 (and op7 after layer 23) waits for
//   Bar[L, BARRIER_ROW(OPCODE_MOE_COMBINE_COMMIT), 0] >= 128.
//
// Each tile instruction computes its gate directly from the route-owned tensors.
// ============================================================================
//
// IMPLEMENTATION NOTES
// --------------------
// * No pages, no semaphores, no TMA.  The gate dot is 16 warps x 128 columns
//   of direct gmem->register `warp::load`s reduced through the instruction
//   scratch; the tile itself is 16 elements, written with plain global stores.
//   The loader therefore releases all 13 pages immediately. `__threadfence()`
//   orders tile stores before the completion counter.
// * The two per-tile spins are executed by all 32 lanes of warp 0, so every
//   lane that later reads `routed_out_f32` / `shared_out` has issued its own
//   acquire fence.
// * The arithmetic is spelled with `__fadd_rn` / `__fmul_rn` so that the
//   single fp32 rounding chain exactly matches the Python VM (PyVM)
//   `(hidden.float() + routed) + gate * shared.float()` and cannot be
//   contracted into an FMA by the compiler.
// ============================================================================

template <typename Config, typename Globals> struct moe_combine_commit {
    static constexpr int opcode = OPCODE_MOE_COMBINE_COMMIT;

    // Four shared split-K segments and four routed slots feed each tile.
    static constexpr int EXPECTED_SHARED_SPLIT_K = 4;
    static constexpr int EXPECTED_ROUTED_SLOTS = 4;
    static_assert(EXPECTED_SHARED_SPLIT_K == Globals::shared_intermediate_dim /
                                                 Globals::split_k_shared,
                  "op11 signals once per split-K segment");
    static_assert(EXPECTED_ROUTED_SLOTS == Globals::num_experts_per_tok,
                  "op13 signals once per top-k slot");

    static constexpr int COLS_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS; // 128
    using rv_b = kittens::rv_bf<COLS_PER_WARP>;
    static constexpr int RV_OUTER = rv_b::outer_dim;
    static constexpr int TILE = Globals::matvec_block_size; // 16

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            start_block_idx = instruction[2];
            end_block_idx = instruction[3];
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return 0;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::laneid() < Config::NUM_PAGES) {
                auto pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
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
            parsed_instruction inst{s};
            const int warpid = kittens::warpid();
            const int laneid = kittens::laneid();

            // ---- step 1: gate = sigmoid(<gate_w[L], normed_hidden>) --------
            // normed_hidden_states was published by op8 and is transitively
            // guaranteed by the op11 / op13 counters spun on below, but the
            // dot is hoisted above the spin exactly as the dependency order requires.
            // The gmem->rv overload takes `coord<rv<...>>` in a deduced
            // position: a bare `coord<>` fails deduction.  `c` is in units of
            // rv::length (= COLS_PER_WARP).
            rv_b gate_w, normed;
            kittens::warp::load(
                gate_w, g.shared_expert_gate_weights,
                kittens::coord<rv_b>{0, 0, inst.layer_idx, warpid});
            kittens::warp::load(normed, g.normed_hidden_states,
                                kittens::coord<rv_b>{0, 0, 0, warpid});

            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < RV_OUTER; i++)
                acc += __bfloat162float(gate_w[i][0]) *
                       __bfloat162float(normed[i][0]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_xor_sync(0xffffffff, acc, off);

            float *scratch = (float *)s.scratch();
            if (laneid == 0) scratch[warpid] = acc;
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(8);

            if (warpid != 0) return; // the tile itself is 16 elements

            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
                dot += scratch[i];
            const float gate = 1.0f / (1.0f + __expf(-dot));

            for (int t = inst.start_block_idx; t < inst.end_block_idx; t++) {
                // ---- step 2: both per-tile counters (all lanes -> all fence)
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{
                        inst.layer_idx,
                        BARRIER_ROW(OPCODE_SHARED_EXPERT_DOWNPROJ), t}],
                    EXPECTED_SHARED_SPLIT_K);
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{
                        inst.layer_idx,
                        BARRIER_ROW(OPCODE_ROUTED_EXPERT_DOWNPROJ), t}],
                    EXPECTED_ROUTED_SLOTS);

                // ---- steps 3 + 4: one fp32 rounding, then clear ------------
                if (laneid < TILE) {
                    const kittens::coord<> at{0, 0, 0, t * TILE + laneid};
                    const float h = __bfloat162float(g.hidden_states[at]);
                    const float r = g.routed_out_f32[at];
                    const float sh = __bfloat162float(g.shared_out[at]);
                    const float v =
                        __fadd_rn(__fadd_rn(h, r), __fmul_rn(gate, sh));
                    g.hidden_states[at] = __float2bfloat16(v);
                    g.routed_out_f32[at] = 0.f;
                }
                kittens::warp::sync();
            }

            // ---- step 5 ----------------------------------------------------
            __threadfence();
            kittens::warp::sync();
            if (laneid == 0)
                atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), 0}],
                          inst.iters);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {}
    };
};
