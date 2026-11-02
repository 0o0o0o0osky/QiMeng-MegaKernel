#pragma once
#include "instruction_list.cuh"

// ============================================================================
// opcode 8 — Router_GEMV
//
// 60 instructions per layer, one per expert.
//
// Static instruction layout (6 words)
//   [0] opcode = 8
//   [1] layer_idx             0..23
//   [2] tile_N_start_idx      EXPERT INDEX 0..59  (granularity is 1 expert)
//   [3] tile_N_end_idx        exclusive
//   [4] write_normed_hidden   0/1 — exactly one instruction per layer has 1
//                                   (the one with tile_N_start_idx == 0)
//   [5] tile_N_granularity    == 1
//
// reads   hidden_states, mlp_norm_weights[L], router_weights[L]
// writes  router_logits_buf[e]  (fp32)
//         if write_normed_hidden: normed_hidden_states  (bf16, [2048])
//
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_OPROJ_GEMV_RESIDUAL), 0] >= 128
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_ROUTER_GEMV), 0] += 1   (per expert)
//                if write_normed_hidden:
//                Bar[L, BARRIER_ROW(OPCODE_ROUTER_GEMV), 1] += 1   (once)
//
// Router RMSNorm, dot products, and logits use fp32. Each expert instruction
// computes its RMSNorm locally so the 60 logits can run independently.
// ============================================================================
//
// IMPLEMENTATION NOTES
// --------------------
// 1. Direct register GEMV. The common pipeline's unit of work is a 16-row
//    output tile fetched with an `st_bf<16,512>` TMA descriptor.
//    Here `tile_N_granularity == 1`, so one instruction produces exactly ONE
//    row of `router_weights[L]` (60 rows total).  Driving the pipeline would
//    make every instruction fetch a 16-row tile and throw 15 rows away:
//    60 x 16 x 2048 x 2 B = 3.9 MiB/layer instead of 245 KiB — a 16x waste on
//    the one op that sits on the critical path. Every consumer warp
//    does a direct gmem->register `warp::load` of its own 128-column slice
//    (naive rv layout, lane `l` of outer step `w` holds element `w*32+l`),
//    which needs no TMA descriptor and no page at all.  The loader therefore
//    releases all 13 pages immediately, exactly like `megakernel::NoOp`.
//
// 2. RMSNorm and router dots accumulate in fp32. `normed_hidden_states` is
//    rounded once when published through its bf16 buffer.
//
// 3. Barrier signalling happens in the consumer (there is no TMA store to
//    drain, so the storer has nothing to do).  Every consumer thread issues
//    its own `__threadfence()` before the group sync that precedes the
//    `atomicAdd` — a fence only orders the writes of the thread executing it,
//    and `normed_hidden_states` is written by all 16 warps.
// ============================================================================

template <typename Config, typename Globals> struct router_gemv {
    static constexpr int opcode = OPCODE_ROUTER_GEMV;
    static constexpr int prev_opcode = OPCODE_OPROJ_GEMV_RESIDUAL;

    // O-projection signals once per 16-element hidden tile.
    static constexpr int EXPECTED_OPROJ_ARRIVALS = 128;
    static_assert(EXPECTED_OPROJ_ARRIVALS ==
                      Globals::hidden_dim / Globals::matvec_block_size,
                  "op4 signals one tile block per 16 hidden rows");

    static constexpr int COLS_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS; // 2048 / 16 = 128
    using rv_f = kittens::rv_fl<COLS_PER_WARP>;
    static constexpr int RV_OUTER = rv_f::outer_dim;
    static_assert(RV_OUTER * 32 == COLS_PER_WARP,
                  "naive rv covers exactly outer_dim*32 elements (lane l of "
                  "step w holds index w*32+l)");

    // scratch (4 KiB, per instruction ring slot): [0,64) rms partials,
    // [256,320) router dot partials.
    static constexpr int RMS_SCRATCH_OFFSET = 0;
    static constexpr int DOT_SCRATCH_OFFSET = 64; // in floats

    struct parsed_instruction {
        int layer_idx, expert_start, expert_end, write_normed, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            expert_start = instruction[2];
            expert_end = instruction[3];
            write_normed = instruction[4];
            // instruction[5] == tile_N_granularity == 1 (experts, not rows)
            iters = expert_end - expert_start;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return query; // no pages are held; identity permutation
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return 0; // no TMA, no cross-role handoff
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            // Nothing is staged in shared memory: hand every page back at once.
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

            // ---- wait for the o_proj residual of this layer -----------------
            if (warpid == 0 && laneid == 0) {
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                                BARRIER_ROW(prev_opcode), 0}],
                    EXPECTED_OPROJ_ARRIVALS);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(6);

            // ---- RMSNorm over the full hidden_dim, 128 columns per warp -----
            // NOTE: the gmem->rv overload takes `coord<rv<...>>` in a deduced
            // position, so a bare `coord<>` fails template deduction — the
            // coordinate type has to be spelled out.  `c` is in units of
            // rv::length (= COLS_PER_WARP), not elements.
            // Same arithmetic as `utils.cuh::rms_norm`, one warp per 128-column
            // slice: fp32 sum of squares -> rsqrtf -> fp32 scale -> fp32
            // multiply by the (bf16-stored, fp32-converted) norm weight.
            rv_f act, sq;
            kittens::warp::load(act, g.hidden_states,
                                kittens::coord<rv_f>{0, 0, 0, warpid});
            kittens::warp::copy(sq, act);
            kittens::warp::mul(sq, sq, sq);
            const float partial_sum = kittens::warp::sum(sq);

            float *rms_scratch = (float *)s.scratch() + RMS_SCRATCH_OFFSET;
            if (laneid == 0) rms_scratch[warpid] = partial_sum;
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(7);

            float full_sum = 0.f;
#pragma unroll
            for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
                full_sum += rms_scratch[i];
            const float variance = full_sum / (float)Globals::hidden_dim;
            const float rms_scale = rsqrtf(variance + g.rms_norm_eps);

            rv_f normed, norm_w;
            kittens::warp::mul(normed, act, rms_scale);
            kittens::warp::load(
                norm_w, g.mlp_norm_weights,
                kittens::coord<rv_f>{0, 0, inst.layer_idx, warpid});
            kittens::warp::mul(normed, normed, norm_w);

            // ---- one fp32 dot per expert -----------------------------------
            float *dot_scratch = (float *)s.scratch() + DOT_SCRATCH_OFFSET;
            for (int e = inst.expert_start; e < inst.expert_end; e++) {
                rv_f rw;
                kittens::warp::load(
                    rw, g.router_weights,
                    kittens::coord<rv_f>{0, inst.layer_idx, e, warpid});
                kittens::warp::mul(rw, rw, normed);
                const float acc = kittens::warp::sum(rw);

                if (laneid == 0) dot_scratch[warpid] = acc;
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(7);

                if (warpid == 0 && laneid == 0) {
                    float logit = 0.f;
#pragma unroll
                    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
                        logit += dot_scratch[i];
                    g.router_logits_buf[{e}] = logit;
                }
                // keeps the next expert from overwriting a slot warp 0 is
                // still reading
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(7);
            }

            // ---- publish normed_hidden_states (exactly one instr per layer) -
            // The buffer dtype requires one BF16 rounding step. The Python VM
            // (PyVM) rounds twice at this boundary.
            if (inst.write_normed) {
                kittens::bf16 *dst =
                    g.normed_hidden_states.raw_ptr + warpid * COLS_PER_WARP;
#pragma unroll
                for (int i = 0; i < RV_OUTER; i++)
                    dst[i * 32 + laneid] = __float2bfloat16(normed[i][0]);
            }

            // Every thread fences its own stores, then one thread signals.
            __threadfence();
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(7);

            if (warpid == 0 && laneid == 0) {
                atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), 0}],
                          inst.iters);
                if (inst.write_normed)
                    atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), 1}],
                              1);
            }
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {}
    };
};
