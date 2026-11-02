#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"
#include <math.h>

// ============================================================================
// opcode 2 — DecodeAttentionSplitKV
//
// 128 instructions per layer = 16 kv_head x 8 partials (exactly one full wave).
// MHA: num_attention_heads == num_kv_heads == 16, so q_head == kv_head_idx.
//
// Static instruction layout (5 words)
//   [0] opcode = 2
//   [1] layer_idx      0..23
//   [2] kv_head_idx    0..15   (== q_head, GQA group is 1)
//   [3] num_partials   == 8
//   [4] partial_idx    0..7
//
// reads   q_post_rope (head slice), k_cache[L], v_cache[L]
// writes  attn_out_intermediates[h, partial_idx, :]  (fp32, head_dim)
//         attn_lse_intermediates[h, partial_idx]     (fp32)
//
// barrier WAITs (all three, expected == 8)
//   Bar[L, BARRIER_ROW(OPCODE_RMS_QKVPROJ_GEMV_ROPE),      q_head] >= 8
//   Bar[L, BARRIER_ROW(OPCODE_RMS_QKVPROJ_GEMV_ROPE), 16 + kv_head] >= 8
//   Bar[L, BARRIER_ROW(OPCODE_RMS_QKVPROJ_GEMV_ROPE), 32 + kv_head] >= 8
// barrier SIGNAL
//   Bar[L, BARRIER_ROW(OPCODE_DECODE_ATTENTION_SPLITKV), q_head] += 1
//
// notes
//   * `attn_scale` = 1/sqrt(128) comes from g.attn_scale, not a literal.
//   * K/V are TMA-loaded with the axis-1 descriptor on kv_cache_t.
//   * LSE stays fp32 and is base-2 (log2 sum exp2), matching the Python VM
//     (PyVM)
//     `logsumexp(...) / log(2)` and op3's `exp2` merge.
//
// design
//   * MHA gives one Q row per instruction (GQA_RATIO == 1).
//   * Two K/V stages exactly fill one 16 KiB page at head_dim 128.
//   * Q uses a same-warp cp.async load/wait and a zero-initialized tile.
//   * Online-softmax rescaling starts after the first initialized maximum.
//   * Launcher waits for K/V and consumer waits for Q once per instruction.
// ============================================================================

using namespace kittens;
using namespace megakernel;

template <typename Config, typename Globals> struct decode_attention_splitkv {
    static constexpr int opcode = OPCODE_DECODE_ATTENTION_SPLITKV;
    static constexpr int prev_opcode = OPCODE_RMS_QKVPROJ_GEMV_ROPE;
    // One arrival per QKV output block in the head.
    static constexpr int EXPECTED_ARRIVAL_COUNT = 8;

    static constexpr int NUM_STAGES = 2;
    static constexpr int GQA_RATIO =
        Globals::num_attention_heads / Globals::num_kv_heads;  // 1
    static constexpr int HEAD_DIM = Globals::head_dim;         // 128
    static constexpr int KV_BLOCK = Globals::kv_block_size;    // 16
    // Barrier slot bases for the K and V output ranges of opcode 1.
    static constexpr int K_SLOT_BASE = Globals::num_attention_heads;        // 16
    static constexpr int V_SLOT_BASE = K_SLOT_BASE + Globals::num_kv_heads; // 32

    static constexpr int QOL_PAGE = 0;
    static constexpr int KV_PAGE = 1;

    static_assert(GQA_RATIO == 1,
                  "Qwen1.5-MoE is MHA; a GQA ratio != 1 needs a Q-tile rework");

    using q_rt = kittens::rt_bf<16, HEAD_DIM>; // only row 0 is used
    using q_st = kittens::st_bf<16, HEAD_DIM>; // only row 0 is used
    using k_rt = kittens::rt_bf<KV_BLOCK, HEAD_DIM>;
    using v_rt = kittens::rt_bf<KV_BLOCK, HEAD_DIM, col_l>;
    using kv_st = kittens::st_bf<KV_BLOCK, HEAD_DIM>;
    using attn_fl_rt = kittens::rt_fl<16, KV_BLOCK>;
    using attn_bf_rt = kittens::rt_bf<16, KV_BLOCK>;
    using vec_rv = col_vec<kittens::rt_fl<16, HEAD_DIM>>; // only row 0 is used
    using o_rt = kittens::rt_fl<16, HEAD_DIM>;            // only row 0 is used
    using o_sv = kittens::sv_fl<HEAD_DIM>;
    using l_sv = kittens::sv_fl<16>; // only data[0] is used

    static_assert(sizeof(q_st) + sizeof(o_sv) + sizeof(l_sv) <=
                      Config::PAGE_SIZE,
                  "QOL page overflow");
    static_assert(2 * NUM_STAGES * sizeof(kv_st) <= Config::PAGE_SIZE,
                  "KV page overflow — reduce NUM_STAGES or add a page");

    struct parsed_instruction {
        int layer_idx;
        int kv_head_idx;
        int num_partials;
        int partial_idx;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            kv_head_idx = instruction[2];
            num_partials = instruction[3];
            partial_idx = instruction[4];
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
        //: MHA -> the Q head and the KV head are the same index.
        __device__ inline int q_head_idx() const { return kv_head_idx; }
    };

    // ---- semaphores (2 + 4*NUM_STAGES = 10; every one has a named waiter) ---
    //   O_arrived -> storer      L_arrived  -> storer
    //   K_arrived -> consumer    K_finished -> launcher
    //   V_arrived -> consumer    V_finished -> launcher
    __device__ static inline kittens::semaphore &
    O_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &
    L_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[1];
    }
    __device__ static inline kittens::semaphore &
    K_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &
    V_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[2 + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &
    K_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &
    V_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2 + 1];
    }
    static constexpr int SEM_COUNT = 2 + 4 * NUM_STAGES;
    static_assert(SEM_COUNT <= Config::DYNAMIC_SEMAPHORES,
                  "too many semaphores");

    // ---- pages -------------------------------------------------------------
    __device__ static inline void wait_QOL_page(megakernel::state<Config> &s) {
        s.wait_page_ready(s.pid(QOL_PAGE));
    }
    __device__ static inline void wait_KV_page(megakernel::state<Config> &s) {
        s.wait_page_ready(s.pid(KV_PAGE));
    }
    __device__ static inline void finish_QOL_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0)
            s.finish_page(s.pid(QOL_PAGE), Config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline void finish_KV_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0)
            s.finish_page(s.pid(KV_PAGE), Config::NUM_CONSUMER_WARPS);
    }

    __device__ static inline q_st &get_Q_smem(megakernel::state<Config> &s) {
        return *reinterpret_cast<q_st *>(s.pages[s.pid(QOL_PAGE)].data);
    }
    __device__ static inline o_sv &get_O_smem(megakernel::state<Config> &s) {
        return *reinterpret_cast<o_sv *>(
            reinterpret_cast<char *>(s.pages[s.pid(QOL_PAGE)].data) +
            sizeof(q_st));
    }
    __device__ static inline l_sv &get_L_smem(megakernel::state<Config> &s) {
        return *reinterpret_cast<l_sv *>(
            reinterpret_cast<char *>(s.pages[s.pid(QOL_PAGE)].data) +
            sizeof(q_st) + sizeof(o_sv));
    }
    __device__ static inline kv_st &get_K_smem(megakernel::state<Config> &s,
                                               int stage) {
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[s.pid(KV_PAGE)].data) +
            sizeof(kv_st) * (stage * 2));
    }
    __device__ static inline kv_st &get_V_smem(megakernel::state<Config> &s,
                                               int stage) {
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[s.pid(KV_PAGE)].data) +
            sizeof(kv_st) * (1 + stage * 2));
    }

    // ---- helpers -----------------------------------------------------------
    //: Extract register-tile row 0 into a shared vector.  In the 16x16 fp32 MMA
    //: layout row 0 lives in lanes 0..3: data[0] holds cols (lane%4)*2 and +1,
    //: data[2] the same pair shifted by 8.
    template <ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void store_row0(SV &dst, const RT &src) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == RT::cols, "dst length must match src cols.");
        using T2 = typename RT::dtype;
        using U = typename SV::dtype;
        using U2 = typename base_types::packing<U>::packed_type;

        const int laneid = kittens::warp::laneid();
        if (laneid < 4) {
            uint32_t dst_ptr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
#pragma unroll
            for (int j = 0; j < RT::width; j++) {
                U2 tmp0 = base_types::convertor<U2, T2>::convert(
                    src.tiles[0][j].data[0]);
                U2 tmp2 = base_types::convertor<U2, T2>::convert(
                    src.tiles[0][j].data[2]);
                const int col_idx = laneid * 2 + j * 16;
                move<U2>::sts(dst_ptr + sizeof(U) * col_idx, tmp0);
                move<U2>::sts(dst_ptr + sizeof(U) * (col_idx + 8), tmp2);
            }
        }
    }

    template <ducks::rt::row_layout RT>
    __device__ static inline void right_fill(
        RT &dst, const RT &src, const int col_idx,
        const typename base_types::packing<typename RT::dtype>::unpacked_type
            &val = 0) {
        if (col_idx >= dst.cols)
            return;
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int j = 0; j < dst.width; j++) {
#pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    const int col_idx_x = (j * dst.tile_size_col) +
                                          ((k / 2) * 8) +
                                          ((kittens::warp::laneid() % 4) * 2);
                    const int col_idx_y = col_idx_x + 1;
                    dst.tiles[i][j].data[k].x =
                        (col_idx_x >= col_idx) ? val
                                               : src.tiles[i][j].data[k].x;
                    dst.tiles[i][j].data[k].y =
                        (col_idx_y >= col_idx) ? val
                                               : src.tiles[i][j].data[k].y;
                }
            }
        }
    }

    //: One Q head == HEAD_DIM bf16 == 256 B; 16 lanes x 16 B covers it.
    __device__ static inline void
    load_Q_async(q_st &dst, const typename Globals::activations_t &src,
                 const int q_head_idx) {
        constexpr int elem_per_memcpy =
            sizeof(float4) / sizeof(typename q_st::dtype); // 8
        constexpr int lanes_needed = HEAD_DIM / elem_per_memcpy;
        static_assert(lanes_needed <= 32,
                      "one warp cannot cover this head_dim in 16 B chunks");

        typename Globals::activations_t::dtype *src_ptr =
            &src.raw_ptr[q_head_idx * HEAD_DIM];
        uint32_t dst_ptr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

        const int laneid = kittens::warp::laneid();
        if (laneid < lanes_needed) {
            const int col = laneid * elem_per_memcpy;
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                    dst.idx(dst_ptr, {0, col})),
                "l"(&src_ptr[col])
                : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    struct controller {
        static_assert(
            qwen_moe_attention_release_lids_are_permutations(Config::NUM_PAGES),
            "attention release_lid helpers must permute every physical page");

        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            // Real release order: the 11 unused pages (freed immediately by the
            // loader), then the KV page (consumer, after the last block), then
            // the QOL page (storer, last).
            return qwen_moe_attention_partial_release_lid_impl(
                Config::NUM_PAGES, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            init_semaphore(O_arrived(s), 0, 1);
            init_semaphore(L_arrived(s), 0, 1);
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(K_arrived(s, i), 0, 1);
                init_semaphore(V_arrived(s, i), 0, 1);
                init_semaphore(K_finished(s, i), 0, 1);
                init_semaphore(V_finished(s, i), 0, 1);
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            const auto laneid = kittens::warp::laneid();
            if (laneid >= 2 && laneid < Config::NUM_PAGES) {
                int unused_page = s.pid(laneid);
                s.wait_page_ready(unused_page);
                s.finish_page(unused_page, Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        //: K and V for this kv_head must be complete before any TMA is issued.
        static __device__ void wait_for_kv(const Globals &g,
                                           megakernel::state<Config> &s,
                                           parsed_instruction &inst) {
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                            BARRIER_ROW(prev_opcode),
                                            K_SLOT_BASE + inst.kv_head_idx}],
                EXPECTED_ARRIVAL_COUNT);
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                            BARRIER_ROW(prev_opcode),
                                            V_SLOT_BASE + inst.kv_head_idx}],
                EXPECTED_ARRIVAL_COUNT);
        }

        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::warp::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
                parsed_instruction inst{s};
                const int seq_len = g.pos_id + 1;
                const int total_attn_blocks =
                    (seq_len + KV_BLOCK - 1) / KV_BLOCK;
                const int blocks_per_partial =
                    (total_attn_blocks + inst.num_partials - 1) /
                    inst.num_partials;
                const int start_blk_idx = inst.partial_idx * blocks_per_partial;
                const int end_blk_idx =
                    min(start_blk_idx + blocks_per_partial, total_attn_blocks);

                wait_KV_page(s);
                if (start_blk_idx >= end_blk_idx) {
                    // Empty partial: nobody else will ever release the KV page.
                    finish_KV_page(s);
                    return;
                }

                wait_for_kv(g, s, inst);

                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    const int cur_blk_idx = start_blk_idx + i;
                    const int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);

                    if (i >= NUM_STAGES) {
                        megakernel::mk_semaphore_wait(
                            K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        megakernel::mk_semaphore_wait(
                            V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    }

                    kittens::tma::expect(K_arrived(s, stage), K_smem);
                    kittens::tma::load_async<
                        kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                        K_smem, g.k_cache,
                        {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0},
                        K_arrived(s, stage));
                    kittens::tma::expect(V_arrived(s, stage), V_smem);
                    kittens::tma::load_async<
                        kittens::dim::DEPTH, kittens::cache_policy::EVICT_FIRST>(
                        V_smem, g.v_cache,
                        {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0},
                        V_arrived(s, stage));
                }
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::warpid() != 0)
                return;

            parsed_instruction inst{s};
            const int q_head_idx = inst.q_head_idx();

            if (kittens::laneid() == 0) {
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                                BARRIER_ROW(prev_opcode),
                                                q_head_idx}],
                    EXPECTED_ARRIVAL_COUNT);
            }
            kittens::warp::sync();

            const int seq_len = g.pos_id + 1;
            const int total_attn_blocks = (seq_len + KV_BLOCK - 1) / KV_BLOCK;
            const int blocks_per_partial =
                (total_attn_blocks + inst.num_partials - 1) / inst.num_partials;
            const int start_blk_idx = inst.partial_idx * blocks_per_partial;
            const int end_blk_idx =
                min(start_blk_idx + blocks_per_partial, total_attn_blocks);

            // 1 / (sqrt(D_h) * ln 2): folds the base-e -> base-2 change into the
            // attention scale so the whole softmax runs on exp2.
            const float softmax_temp = g.attn_scale * 1.44269504089f;

            q_rt Q_reg;
            k_rt K_reg;
            v_rt V_reg;
            o_rt O_reg;
            attn_fl_rt attn_fl_reg;
            attn_bf_rt attn_bf_reg;
            vec_rv max_vec_reg, scaled_max_vec_reg, last_scaled_max_vec_reg,
                diff_scaled_max_vec_reg, norm_vec_reg, L_reg;

            kittens::warp::neg_infty(max_vec_reg);
            kittens::warp::zero(norm_vec_reg);
            kittens::warp::zero(O_reg);

            o_sv &O_smem = get_O_smem(s);
            l_sv &L_smem = get_L_smem(s);

            wait_QOL_page(s);
            q_st &Q_smem = get_Q_smem(s);
            // Only row 0 carries this head; zero the rest so the MMA never sees
            // uninitialised bf16.
            kittens::warp::zero(Q_smem);
            kittens::warp::sync();
            load_Q_async(Q_smem, g.q_post_rope, q_head_idx);
            kittens::warp::load_async_wait();
            kittens::warp::sync();
            kittens::warp::load(Q_reg, Q_smem);

            for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                const int stage = i % NUM_STAGES;
                kv_st &K_smem = get_K_smem(s, stage);
                kv_st &V_smem = get_V_smem(s, stage);

                // Q @ K^T
                kittens::warp::zero(attn_fl_reg);
                megakernel::mk_semaphore_wait(
                    K_arrived(s, stage), (i / NUM_STAGES) % 2);
                kittens::warp::load(K_reg, K_smem);
                kittens::warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                kittens::warp::sync();
                kittens::warp::arrive(K_finished(s, stage));

                // Mask the ragged tail of the last KV block.
                if ((i + start_blk_idx + 1) * KV_BLOCK > seq_len)
                    right_fill(attn_fl_reg, attn_fl_reg, seq_len % KV_BLOCK,
                               -INFINITY);

                kittens::warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg);
                kittens::warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                kittens::warp::mul(scaled_max_vec_reg, max_vec_reg,
                                   softmax_temp);

                kittens::warp::sub_row(attn_fl_reg, attn_fl_reg,
                                       scaled_max_vec_reg);
                kittens::warp::exp2(attn_fl_reg, attn_fl_reg);

                if (i != 0) {
                    kittens::warp::sub(diff_scaled_max_vec_reg,
                                       last_scaled_max_vec_reg,
                                       scaled_max_vec_reg);
                    kittens::warp::exp2(diff_scaled_max_vec_reg,
                                        diff_scaled_max_vec_reg);
                    kittens::warp::mul_row(O_reg, O_reg,
                                           diff_scaled_max_vec_reg);
                    kittens::warp::mul(norm_vec_reg, norm_vec_reg,
                                       diff_scaled_max_vec_reg);
                }

                megakernel::mk_semaphore_wait(
                    V_arrived(s, stage), (i / NUM_STAGES) % 2);
                kittens::warp::load(V_reg, V_smem);
                kittens::warp::copy(attn_bf_reg, attn_fl_reg);
                kittens::warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                kittens::warp::sync();
                kittens::warp::arrive(V_finished(s, stage));

                kittens::warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);
                kittens::warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
            }

            kittens::warp::sync();

            if (start_blk_idx < end_blk_idx) {
                finish_KV_page(s);
                kittens::warp::div_row(O_reg, O_reg, norm_vec_reg);
                kittens::warp::log2(L_reg, norm_vec_reg);
                // L_reg == log2(sum exp(qk)) == logsumexp / ln 2.
                kittens::warp::add(L_reg, L_reg, last_scaled_max_vec_reg);
            } else {
                // Empty partial: O stays 0 and LSE = -inf, the neutral element
                // for op3's exp2 merge.
                kittens::warp::neg_infty(L_reg);
            }

            store_row0(O_smem, O_reg);
            kittens::warp::sync();
            kittens::warp::arrive(O_arrived(s));

            kittens::warp::store(L_smem, L_reg);
            kittens::warp::sync();
            kittens::warp::arrive(L_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            const int laneid = kittens::warp::laneid();
            const int q_head_idx = inst.q_head_idx();

            o_sv &O_smem = get_O_smem(s);
            l_sv &L_smem = get_L_smem(s);

            if (laneid == 0) {
                megakernel::mk_semaphore_wait(O_arrived(s), 0);
                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                    g.attn_out_intermediates, O_smem,
                    {0, q_head_idx, inst.partial_idx, 0});

                // The consumer arrives O_arrived BEFORE it writes L_reg into
                // L_smem, so the LSE needs its own wait — reading it under
                // O_arrived alone would race (llama race fix, kept).
                megakernel::mk_semaphore_wait(L_arrived(s), 0);
                // attn_lse_intermediates carries only a sv_fl<16> descriptor,
                // which is wider than the 8-column tensor, so a single LSE goes
                // out as a plain store.
                float *dst =
                    (float *)&g.attn_lse_intermediates
                         .raw_ptr[q_head_idx *
                                      g.attn_lse_intermediates.cols() +
                                  inst.partial_idx];
                *dst = L_smem.data[0];
            }

            kittens::warp::sync();
            kittens::tma::store_async_wait();
            // finish_page strictly after warp::sync() + the TMA drain.
            if (laneid == 0)
                finish_QOL_page(s);

            kittens::warp::sync();
            if (laneid == 0) {
                // Device-scope release fence: the O partial (TMA) and the LSE
                // (plain store) must be visible to the reducing CTA before its
                // counter moves.
                __threadfence();
                atomicAdd(
                    &g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), q_head_idx}],
                    1);
            }
        }
    };
};
