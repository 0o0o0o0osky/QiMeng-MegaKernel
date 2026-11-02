#pragma once
#include "instruction_list.cuh"

// ============================================================================
// opcode 9 — Router_SoftmaxTopK
//
// 1 instruction per layer.
//
// Static instruction layout (2 words)
//   [0] opcode = 9
//   [1] layer_idx   0..23
//
// reads   router_logits_buf[0..59]  (fp32)
// writes  expert_indices[0..3]  (int64, PHYSICAL expert ids)
//         expert_weights[0..3]  (fp32, softmax probabilities)
//
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_ROUTER_GEMV), 0] >= 60
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_ROUTER_SOFTMAX_TOPK), 0] += 1
//
// semantics
//   * fp32 softmax over all 60 logits, then top-4 of the probabilities.
//   * `norm_topk_prob = false`: selected probabilities retain their original
//     values from the 60-way softmax.
//   * tie-break identical to torch.topk: on equal values the SMALLER index wins.
//   * expert_indices is int64 (Globals::expert_indices_t is gl<int64_t,...>).
// ============================================================================
//
// IMPLEMENTATION NOTES
// --------------------
// One warp, 60 logits, two per lane (lane l holds e=l and e=l+32 for l<28).
// Everything is warp-local: max, sum and the four selection rounds are
// `__shfl_xor_sync` reductions, so no shared memory, no pages and no
// semaphores are needed — the loader hands all 13 pages straight back.
//
// Tie-break selects the smaller index for equal values. Selection operates on
// the softmax probabilities, matching the Python VM (PyVM) stage boundary.
// ============================================================================

template <typename Config, typename Globals> struct router_softmax_topk {
    static constexpr int opcode = OPCODE_ROUTER_SOFTMAX_TOPK;
    static constexpr int prev_opcode = OPCODE_ROUTER_GEMV;

    // One router-logit arrival per expert.
    static constexpr int EXPECTED_ROUTER_LOGITS = 60;
    static_assert(EXPECTED_ROUTER_LOGITS == Globals::num_experts,
                  "op8 signals once per expert");
    static constexpr int TOP_K = Globals::num_experts_per_tok; // 4
    static_assert(Globals::num_experts <= 2 * 32,
                  "one warp covers at most 64 experts with two slots per lane");

    struct parsed_instruction {
        int layer_idx;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
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
            if (kittens::warpid() != 0) return;

            parsed_instruction inst{s};
            const int laneid = kittens::laneid();

            // Every lane spins (and therefore every lane fences) before it
            // reads the logits its own reduction depends on.
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                            BARRIER_ROW(prev_opcode), 0}],
                EXPECTED_ROUTER_LOGITS);

            const int e0 = laneid;
            const int e1 = laneid + 32;
            const bool v0 = e0 < Globals::num_experts;
            const bool v1 = e1 < Globals::num_experts;

            float l0 = v0 ? g.router_logits_buf[{e0}] : -INFINITY;
            float l1 = v1 ? g.router_logits_buf[{e1}] : -INFINITY;

            // ---- fp32 softmax (max-shifted, exactly like torch) ------------
            float m = fmaxf(l0, l1);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));

            float x0 = v0 ? __expf(l0 - m) : 0.f;
            float x1 = v1 ? __expf(l1 - m) : 0.f;
            float denom = x0 + x1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                denom += __shfl_xor_sync(0xffffffff, denom, off);

            float p0 = v0 ? x0 / denom : -1.f;
            float p1 = v1 ? x1 / denom : -1.f;

            // ---- top-4, smaller index wins on ties -------------------------
            int i0 = e0, i1 = e1;
#pragma unroll
            for (int k = 0; k < TOP_K; k++) {
                float bv = p0;
                int bi = i0;
                if (p1 > bv || (p1 == bv && i1 < bi)) {
                    bv = p1;
                    bi = i1;
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    float ov = __shfl_xor_sync(0xffffffff, bv, off);
                    int oi = __shfl_xor_sync(0xffffffff, bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) {
                        bv = ov;
                        bi = oi;
                    }
                }
                if (laneid == 0) {
                    g.expert_indices[{k}] = (int64_t)bi;
                    g.expert_weights[{k}] = bv;
                }
                // strike the winner out of the candidate set
                if (i0 == bi) p0 = -1.f;
                if (i1 == bi) p1 = -1.f;
            }

            __threadfence();
            kittens::warp::sync();
            if (laneid == 0)
                atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), 0}], 1);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {}
    };
};
