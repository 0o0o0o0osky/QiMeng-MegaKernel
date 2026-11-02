#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"
#include <limits>

// opcode 3 — DecodeAttentionReduction
// 16 instructions per layer, **1 head each** (16 heads total).

// Static instruction layout (15 words)
//   [0]     opcode = 3
//   [1]     layer_idx           0..23

//   [3]     num_partials        == 8
//   [4]     len(reduction_list) == 8
//   [5..12] reduction_list      == 0,1,...,7
//   [13]    is_terminal         == 1
//   [14]    output_partial_idx  == 0
// reads   attn_out_intermediates[h, p, :], attn_lse_intermediates[h, p] (fp32)
// writes  attn_out[h * head_dim .. +head_dim]  (bf16)
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_DECODE_ATTENTION_SPLITKV),
//                    head_start_idx] >= 8
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_DECODE_ATTENTION_REDUCTION), 0] += 1
//                (+Q_HEADS_PER_INSTRUCTION == attn_reduction_size)
// The LSE TMA box is wider than the tensor.
//   `attn_lse_intermediates` is declared `gl<float,1,1,16,-1, sv_fl<16>>` but
//   the tensor has only `num_partials == 8` columns, so a TMA load zero-fills
//   columns 8..15.  An LSE of 0 is NOT neutral — `exp2(0) = 1` would inject
//   eight phantom partials of weight 1 each.  The merge loop below is bounded
//   by `inst.num_partials` and only ever indexes `reduction_list[i] < 8`, so
//   the padding is never read. The merge loop is therefore bounded by
//   `inst.num_partials`.
// `reduction_list` is read from the shared instruction words. `l_partial_sv`
// matches the 16-column descriptor, and this single-batch schedule emits only
// terminal reductions.

using namespace kittens;
using namespace megakernel;

template <typename Config, typename Globals> struct decode_attention_reduction {
    static constexpr int opcode = OPCODE_DECODE_ATTENTION_REDUCTION;
    static constexpr int prev_opcode = OPCODE_DECODE_ATTENTION_SPLITKV;
    // Each head waits for all attention partials.
    static constexpr int EXPECTED_ARRIVAL_COUNT = Globals::num_partials;
    // The signal amount matches the scheduler's attention-reduction size.
    static constexpr int Q_HEADS_PER_INSTRUCTION = 1;

    static constexpr int NUM_STAGES = Globals::num_partials;
    static constexpr int HEAD_DIM = Globals::head_dim;
    static constexpr int SHARED_DATA_PAGE = 0;
    //: first reduction_list element sits at word 5
    static constexpr int REDUCTION_LIST_WORD0 = 5;

    static_assert(Globals::num_attention_heads % Q_HEADS_PER_INSTRUCTION == 0,
                  "num_attention_heads must divide by Q_HEADS_PER_INSTRUCTION");
    static_assert(Q_HEADS_PER_INSTRUCTION <= Config::NUM_CONSUMER_WARPS,
                  "one consumer warp per head");

    using l_partial_sv = kittens::sv_fl<QWEN_MOE_ROUNDED_NUM_PARTIALS>;
    using o_sv = kittens::sv_fl<HEAD_DIM>;
    using o_rv = kittens::rv_fl<HEAD_DIM>;
    using o_final_sv = kittens::sv_bf<HEAD_DIM>;

    struct parsed_instruction {
        int layer_idx;
        int q_head_start_idx;
        int num_partials;
        int reduction_list_length;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            q_head_start_idx = instruction[2];
            num_partials = instruction[3];
            reduction_list_length = instruction[4];
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    //: Read reduction_list[i] straight from the shared instruction buffer.
    __device__ static inline int reduction_partial(megakernel::state<Config> &s,
                                                   int i) {
        return s.instruction()[REDUCTION_LIST_WORD0 + i];
    }

    // ---- semaphores (4*(2*NUM_STAGES) + 4*2 + 4 = 28) ----------------------
    __device__ static constexpr int
    O_partial_sem_idx(int q_head_local_idx, int stage, bool is_finished) {
        return q_head_local_idx * (NUM_STAGES * 2) + stage * 2 +
               (is_finished ? 1 : 0);
    }
    __device__ static constexpr int L_partial_sem_idx(int q_head_local_idx,
                                                      bool is_finished) {
        return (Q_HEADS_PER_INSTRUCTION * NUM_STAGES * 2) +
               q_head_local_idx * 2 + (is_finished ? 1 : 0);
    }
    __device__ static constexpr int Final_O_ready_sem_idx(int q_head_local_idx) {
        return (Q_HEADS_PER_INSTRUCTION * NUM_STAGES * 2) +
               (Q_HEADS_PER_INSTRUCTION * 2) + q_head_local_idx;
    }
    static constexpr int SEM_COUNT =
        Q_HEADS_PER_INSTRUCTION * ((NUM_STAGES * 2) + 3);
    static_assert(SEM_COUNT <= Config::DYNAMIC_SEMAPHORES,
                  "too many semaphores");

    __device__ static inline kittens::semaphore &
    O_partial_arrived(megakernel::state<Config> &s, int h, int stage) {
        return s.semaphores()[O_partial_sem_idx(h, stage, false)];
    }
    __device__ static inline kittens::semaphore &
    O_partial_finished(megakernel::state<Config> &s, int h, int stage) {
        return s.semaphores()[O_partial_sem_idx(h, stage, true)];
    }
    __device__ static inline kittens::semaphore &
    L_partial_all_arrived(megakernel::state<Config> &s, int h) {
        return s.semaphores()[L_partial_sem_idx(h, false)];
    }
    __device__ static inline kittens::semaphore &
    L_partial_all_finished(megakernel::state<Config> &s, int h) {
        return s.semaphores()[L_partial_sem_idx(h, true)];
    }
    __device__ static inline kittens::semaphore &
    final_O_ready(megakernel::state<Config> &s, int h) {
        return s.semaphores()[Final_O_ready_sem_idx(h)];
    }

    // ---- shared memory layout (one page) -----------------------------------
    static constexpr size_t size_per_head =
        sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv) + sizeof(o_final_sv);
    static constexpr size_t total_smem_needed =
        Q_HEADS_PER_INSTRUCTION * size_per_head;
    static_assert(total_smem_needed <= Config::PAGE_SIZE,
                  "reduction scratch exceeds one page");

    __device__ static inline void wait_shared_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0)
            s.wait_page_ready(s.pid(SHARED_DATA_PAGE));
    }
    __device__ static inline void
    finish_shared_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0)
            s.finish_page(s.pid(SHARED_DATA_PAGE), Config::NUM_CONSUMER_WARPS);
    }

    __device__ static inline char *head_base(megakernel::state<Config> &s,
                                             int h) {
        return reinterpret_cast<char *>(
                   s.pages[s.pid(SHARED_DATA_PAGE)].data) +
               h * size_per_head;
    }
    __device__ static inline l_partial_sv &
    get_L_partial_smem(megakernel::state<Config> &s, int h) {
        return *reinterpret_cast<l_partial_sv *>(head_base(s, h));
    }
    __device__ static inline o_sv &
    get_O_partial_smem(megakernel::state<Config> &s, int h, int stage) {
        return *reinterpret_cast<o_sv *>(head_base(s, h) +
                                         sizeof(l_partial_sv) +
                                         stage * sizeof(o_sv));
    }
    __device__ static inline o_final_sv &
    get_O_final_smem(megakernel::state<Config> &s, int h) {
        return *reinterpret_cast<o_final_sv *>(head_base(s, h) +
                                               sizeof(l_partial_sv) +
                                               NUM_STAGES * sizeof(o_sv));
    }

    struct controller {
        static_assert(
            qwen_moe_attention_release_lids_are_permutations(Config::NUM_PAGES),
            "attention release_lid helpers must permute every physical page");

        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return attention_reduction_release_lid_impl(Config::NUM_PAGES,
                                                        query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            for (int h = 0; h < Q_HEADS_PER_INSTRUCTION; ++h) {
                for (int stage = 0; stage < NUM_STAGES; stage++) {
                    init_semaphore(O_partial_arrived(s, h, stage), 0, 1);
                    init_semaphore(O_partial_finished(s, h, stage), 0, 1);
                }
                init_semaphore(L_partial_all_arrived(s, h), 0, 1);
                init_semaphore(L_partial_all_finished(s, h), 0, 1);
                init_semaphore(final_O_ready(s, h), 0, 1);
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            const auto laneid = kittens::warp::laneid();
            // Page 0 is the only one this op uses; the launcher waits for it
            // itself.  llama has the *loader* wait for page 0 and the
            // *launcher* TMA into it with no cross-warp ordering between them —
            // `wait_page_ready` is a phase-bit poll, so having the consumer of
            // the page do its own wait is both correct and free.
            if (laneid >= 1 && laneid < Config::NUM_PAGES) {
                s.wait_page_ready(s.pid(laneid));
                s.finish_page(s.pid(laneid), Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::warp::laneid() != 0)
                return;
#ifdef KITTENS_BLACKWELL
            s.wait_tensor_ready();
            arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            parsed_instruction inst{s};
            wait_shared_page(s);

            for (int h = 0; h < Q_HEADS_PER_INSTRUCTION; ++h) {
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                                BARRIER_ROW(prev_opcode),
                                                inst.q_head_start_idx + h}],
                    EXPECTED_ARRIVAL_COUNT);
            }

            // Whole LSE row per head.  Columns num_partials..15 are outside the
            // tensor and come back zero-filled; the consumer never reads them.
            for (int h = 0; h < Q_HEADS_PER_INSTRUCTION; ++h) {
                l_partial_sv &L_smem = get_L_partial_smem(s, h);
                kittens::tma::expect(L_partial_all_arrived(s, h), L_smem);
                kittens::tma::load_async<kittens::cache_policy::EVICT_FIRST>(
                    L_smem, g.attn_lse_intermediates,
                    {0, 0, inst.q_head_start_idx + h, 0},
                    L_partial_all_arrived(s, h));
            }

            for (int i = 0; i < inst.num_partials; ++i) {
                const int stage = i % NUM_STAGES;
                const int cur_partial_idx = reduction_partial(s, i);
                for (int h = 0; h < Q_HEADS_PER_INSTRUCTION; ++h) {
                    o_sv &O_smem = get_O_partial_smem(s, h, stage);
                    if (i >= NUM_STAGES) {
                        megakernel::mk_semaphore_wait(
                            O_partial_finished(s, h, stage),
                            (i / NUM_STAGES - 1) % 2);
                    }
                    kittens::tma::expect(O_partial_arrived(s, h, stage), O_smem);
                    kittens::tma::load_async<kittens::cache_policy::EVICT_FIRST>(
                        O_smem, g.attn_out_intermediates,
                        {0, inst.q_head_start_idx + h, cur_partial_idx, 0},
                        O_partial_arrived(s, h, stage));
                }
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            const int h = kittens::warpid();
            if (h >= Q_HEADS_PER_INSTRUCTION)
                return;

            parsed_instruction inst{s};

            o_rv accumulated_out, current_out;
            float accumulated_lse = -INFINITY;
            float current_lse;
            kittens::warp::zero(accumulated_out);

            megakernel::mk_semaphore_wait(L_partial_all_arrived(s, h), 0);
            l_partial_sv &L_smem = get_L_partial_smem(s, h);

            // Bounded by num_partials (== 8), never by l_partial_sv::length
            // (== 16).  See the TMA-box warning at the top of this file.
            for (int i = 0; i < inst.num_partials; ++i) {
                const int stage = i % NUM_STAGES;
                megakernel::mk_semaphore_wait(
                    O_partial_arrived(s, h, stage), (i / NUM_STAGES) % 2);

                o_sv &O_smem = get_O_partial_smem(s, h, stage);
                const int cur_partial_idx = reduction_partial(s, i);
                uint32_t src_ptr_L = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&L_smem.data[cur_partial_idx]));
                move<float>::lds(current_lse, src_ptr_L);
                kittens::warp::load(current_out, O_smem);

                const float max_lse = max(accumulated_lse, current_lse);
                const float accumulated_exp = exp2f(accumulated_lse - max_lse);
                const float current_exp = exp2f(current_lse - max_lse);
                const float new_denom = accumulated_exp + current_exp;

                kittens::warp::mul(accumulated_out, accumulated_out,
                                   accumulated_exp / new_denom);
                kittens::warp::mul(current_out, current_out,
                                   current_exp / new_denom);
                kittens::warp::add(accumulated_out, accumulated_out,
                                   current_out);

                accumulated_lse = max_lse + log2f(new_denom);

                kittens::warp::arrive(O_partial_finished(s, h, stage));
            }
            kittens::warp::arrive(L_partial_all_finished(s, h));

            o_final_sv &O_final_smem = get_O_final_smem(s, h);
            kittens::warp::store(O_final_smem, accumulated_out);
            kittens::warp::sync();
            kittens::warp::arrive(final_O_ready(s, h));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            const int laneid = kittens::warp::laneid();

            if (laneid < Q_HEADS_PER_INSTRUCTION) {
                o_final_sv &O_final_smem = get_O_final_smem(s, laneid);
                megakernel::mk_semaphore_wait(final_O_ready(s, laneid), 0);
                kittens::tma::store_async<kittens::cache_policy::NORMAL>(
                    g.attn_out, O_final_smem,
                    {0, 0, 0, inst.q_head_start_idx + laneid});
            }
            // RACE FIX: lanes 0..3 each issue a TMA store out of this page.
            // With independent thread scheduling lane 0 could leave the branch
            // and release the page while lanes 1..3 still have stores in
            // flight; the next instruction on this SM would then overwrite the
            // smem being read.  sync + drain BEFORE finish_page.
            kittens::warp::sync();
            kittens::tma::store_async_wait();
            kittens::warp::sync();
            finish_shared_page(s);

            kittens::warp::sync();
            if (laneid == 0) {
                // Device-scope release fence before the cross-CTA counter move.
                __threadfence();
                atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), 0}],
                          Q_HEADS_PER_INSTRUCTION);
            }
        }
    };
};
