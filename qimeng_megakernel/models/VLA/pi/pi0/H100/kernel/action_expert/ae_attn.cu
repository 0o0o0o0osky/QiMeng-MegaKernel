#pragma once

#include "kittens.cuh"
#include "common/ae_globals.cuh"
#include "common/ae_instruction_list.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"
#include "include/pipelines/wgmma_compat.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

template <typename Config, typename Globals, int Opcode, int BarrierRow>
struct ae_attn_partial {
    static constexpr int opcode = Opcode;

    static constexpr int HD        = dec::HEAD_DIM;                      // 256
    static constexpr int KV_BLK    = 64;
    static constexpr int NUM_WG    = Config::NUM_CONSUMER_WARPS / 4;     // 1
    static constexpr int KV_STAGES = 2;

    static_assert(NUM_WG == 1, "action_expert_config requires exactly one warpgroup");
    static_assert(NUM_WG + 2 * KV_STAGES + 1 == Config::NUM_PAGES,
                  "Partial layout requires one Q/output page, two K/V stage pages, and one spare page");

    static constexpr float SCALE       = 0.0625f;
    static constexpr float SCALE_LOG2E = SCALE * 1.44269504088896340736f;

    static constexpr int IDLE_LID = Config::NUM_PAGES - 1;   // 5

    using q_st  = st_bf<64, HD>;
    using kv_st = st_bf<KV_BLK, HD>;
    using o_st  = st_bf<64, HD>;

    struct parsed_instruction {
        int opcode, layer, head, kv_split, dep_layer, dep_row, dep_expected;
        int kv_blk0;
        int kv_nblk;
        int seq_len;
        int prefix_len;
        int state_key;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), head(i[2]), kv_split(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]),
              kv_blk0(i[7]), kv_nblk(i[8]), seq_len(i[9]),
              prefix_len(i[10]), state_key(i[11]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &q_arrived(state<Config> &s)          { return s.semaphores()[0]; }
    __device__ static inline semaphore &kv_arrived(state<Config> &s, int st) { return s.semaphores()[1 + st]; }
    __device__ static inline semaphore &kv_free(state<Config> &s, int st)    { return s.semaphores()[1 + KV_STAGES + st]; }
    __device__ static inline semaphore &out_ready(state<Config> &s)          { return s.semaphores()[1 + 2 * KV_STAGES]; }
    static constexpr int SEM_COUNT = 2 + 2 * KV_STAGES;

    __device__ static inline q_st  &q_at(state<Config> &s)          { return reinterpret_cast<q_st &>(s.pages[s.pid(0)]); }
    __device__ static inline o_st  &o_at(state<Config> &s)          { return reinterpret_cast<o_st &>(s.pages[s.pid(0)]); }
    __device__ static inline kv_st &k_at(state<Config> &s, int st)  { return reinterpret_cast<kv_st &>(s.pages[s.pid(1 + 2 * st)]); }
    __device__ static inline kv_st &v_at(state<Config> &s, int st)  { return reinterpret_cast<kv_st &>(s.pages[s.pid(2 + 2 * st)]); }

    struct controller {
        static __device__ int release_lid(const Globals &, typename Config::instruction_t &, int &query) {

            constexpr int order[6] = {IDLE_LID, 1, 2, 3, 4, 0};
            return order[query];
        }
        static __device__ int init_semaphores(const Globals &, state<Config> &s) {
            for (int i = 0; i < SEM_COUNT; i++) {

                int cnt = (i > KV_STAGES) ? Config::NUM_CONSUMER_WARPS : 1;
                init_semaphore(s.semaphores()[i], cnt);
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            if (laneid() == IDLE_LID) {
                int pid = s.pid(IDLE_LID);
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            if (laneid() != 0) { warp::sync(); return; }

            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

            tma::expect_bytes(q_arrived(s), sizeof(q_st));
            s.wait_page_ready(s.pid(0));

            tma::load_async(q_at(s), g.qkv_buf, {0, inst.head}, q_arrived(s));

#pragma unroll
            for (int st = 0; st < KV_STAGES; st++) {
                s.wait_page_ready(s.pid(1 + 2 * st));
                s.wait_page_ready(s.pid(2 + 2 * st));
            }

            for (int b = 0; b < inst.kv_nblk; b++) {
                const int st = b % KV_STAGES;
                if (b >= KV_STAGES) ::megakernel::mk_semaphore_wait(kv_free(s, st), ((b / KV_STAGES) - 1) & 1);
                tma::expect_bytes(kv_arrived(s, st), 2 * sizeof(kv_st));
                tma::load_async(k_at(s, st), g.k_cache, {inst.layer, inst.kv_blk0 + b, 0},
                                kv_arrived(s, st));
                tma::load_async(v_at(s, st), g.v_cache, {inst.layer, inst.kv_blk0 + b, 0},
                                kv_arrived(s, st));
            }
            warp::sync();
        }
    };

    struct launcher { static __device__ void run(const Globals &, state<Config> &) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            rt_fl<16, HD> o;
            warp::zero(o);
            col_vec<rt_fl<16, KV_BLK>> mx, nrm, prev, corr;
            warp::neg_infty(mx);
            warp::zero(nrm);

            ::megakernel::mk_semaphore_wait(q_arrived(s), 0);
            const float scale = SCALE_LOG2E;

            const bool holds_state_row = (warpgroup::warpid() == 0);
            const int  prefix_len = inst.prefix_len;
            const int  state_key  = inst.state_key;

            for (int b = 0; b < inst.kv_nblk; b++) {
                const int st = b % KV_STAGES;
                ::megakernel::mk_semaphore_wait(kv_arrived(s, st), (b / KV_STAGES) & 1);

                rt_fl<16, KV_BLK> att;
                warpgroup::mm_ABt(att, q_at(s), k_at(s, st));
                warpgroup::mma_async_wait();
                warp::mul(att, att, scale);

                const int base = (inst.kv_blk0 + b) * KV_BLK;

                const int valid = inst.seq_len - base;
                if (valid < KV_BLK)
                    warp::apply(att, att, [valid] __device__(int, int col, float x) {
                        return col < valid ? x : -INFINITY;
                    });

                // The state row attends only to prefix tokens and its own key.

                if (holds_state_row && base + KV_BLK > prefix_len)
                    warp::apply(att, att, [base, prefix_len, state_key] __device__(int row, int col, float x) {
                        if (row != 0) return x;
                        const int key = base + col;
                        return (key < prefix_len || key == state_key) ? x : -INFINITY;
                    });

                warp::copy(prev, mx);
                warp::row_max(mx, att, mx);
                warp::sub_row(att, att, mx);
                warp::exp2(att, att);

                warp::sub(corr, prev, mx);
                warp::exp2(corr, corr);
                warp::mul(nrm, nrm, corr);
                warp::row_sum(nrm, att, nrm);
                warp::mul_row(o, o, corr);

                rt_bf<16, KV_BLK> att_bf;
                warp::copy(att_bf, att);

                wgmma_compat::mma_AB(o, att_bf, v_at(s, st));
                warpgroup::mma_async_wait();

                warp::arrive(kv_free(s, st));
            }

#pragma unroll
            for (int st = 0; st < KV_STAGES; st++) {
                s.warp_finish_page(s.pid(1 + 2 * st), 1);   // K
                s.warp_finish_page(s.pid(2 + 2 * st), 1);   // V
            }

            warp::div_row(o, o, nrm);
            warpgroup::store(o_at(s), o);

            float *scr = reinterpret_cast<float *>(s.scratch());
            auto &m_sm = *reinterpret_cast<sv_fl<64> *>(scr);
            auto &l_sm = *reinterpret_cast<sv_fl<64> *>(scr + 64);
            warpgroup::store(m_sm, mx);
            warpgroup::store(l_sm, nrm);

            warpgroup::sync(4);

            if (warpgroup::warpid() == 0) {
                const int hb = inst.head * 2;
                for (int i = laneid(); i < dec::M_PAD; i += 32) {
                    g.attn_stats[coord<>{inst.kv_split, hb + 0, i}] = m_sm[i];
                    g.attn_stats[coord<>{inst.kv_split, hb + 1, i}] = l_sm[i];
                }
            }
            warp::sync();
            warp::arrive(out_ready(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(out_ready(s), 0);
            if (laneid() == 0) {

                tma::store_async(g.attn_partial, o_at(s), {inst.kv_split, 0, inst.head});
                tma::store_async_read_wait();
                s.finish_page(s.pid(0), Config::NUM_CONSUMER_WARPS);
            }
            tma::store_async_wait();
            warp::sync();
            if (laneid() == 0) {
                __threadfence();
                atomicAdd((uint32_t *)&g.Bar[coord<>{inst.layer, BarrierRow, 0}], 1);
            }
        }
    };
};

template <typename Config, typename Globals, int Opcode, int BarrierRow, int _col_blks = 2>
struct ae_attn_reduce {
    static constexpr int opcode = Opcode;

    static constexpr int COL_BLKS = _col_blks;                      // production: 8
    static constexpr int COL      = dec::HEAD_DIM / COL_BLKS;       // production: 32

    static_assert(dec::HEAD_DIM % COL_BLKS == 0 && COL % 32 == 0,
                  "The output tile width must be a multiple of 32 for TMA and shared-memory tiles");
    static constexpr int HALF   = COL;
    static constexpr int NUM_WG = Config::NUM_CONSUMER_WARPS / 4;   // 1

#ifndef AE_REDUCE_TILE_PAGES
#define AE_REDUCE_TILE_PAGES 1
#endif
    static constexpr int TILE_PAGES  = AE_REDUCE_TILE_PAGES;
    static constexpr int PER_PAGE    = Config::PAGE_SIZE / (64 * HALF * 2);
    static constexpr int STAGES      = TILE_PAGES * PER_PAGE;
    static constexpr int OUT_LID     = TILE_PAGES;
    static constexpr int IDLE_LID0   = TILE_PAGES + 1;
    static_assert(TILE_PAGES + 1 < Config::NUM_PAGES, "Reduction layout requires tile pages, one output page, and at least one spare page");

    static constexpr int STATS_STRIDE = 2 * dec::M_PAD;     // 128
    static constexpr int MAX_SPLITS   = Config::SCRATCH_BYTES / 4 / STATS_STRIDE;
    static_assert(NUM_WG == 1, "action_expert_config requires exactly one warpgroup");
    static_assert(MAX_SPLITS >= 16, "Scratch storage cannot hold all splits required by this configuration");

    using p_st = st_bf<64, HALF>;                           // production: 4KB

    struct parsed_instruction {

        int opcode, layer, head, col_blk, dep_layer, dep_row, dep_expected;
        int n_splits;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), head(i[2]), col_blk(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), n_splits(i[7]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &tile_arrived(state<Config> &s, int st) { return s.semaphores()[st]; }
    __device__ static inline semaphore &tile_free(state<Config> &s, int st)    { return s.semaphores()[STAGES + st]; }
    __device__ static inline semaphore &out_ready(state<Config> &s)            { return s.semaphores()[2 * STAGES]; }

    __device__ static inline semaphore &stats_ready(state<Config> &s)          { return s.semaphores()[2 * STAGES + 1]; }
    static constexpr int SEM_COUNT = 2 * STAGES + 2;

    __device__ static inline p_st &tile_at(state<Config> &s, int st) {
        return reinterpret_cast<p_st *>(&s.pages[s.pid(st / PER_PAGE)])[st % PER_PAGE];
    }
    __device__ static inline p_st &out_at(state<Config> &s) {
        return reinterpret_cast<p_st &>(s.pages[s.pid(OUT_LID)]);
    }

    __device__ static inline float *stats_base(state<Config> &s) {
        return reinterpret_cast<float *>(s.scratch());
    }

    struct controller {
        static __device__ int release_lid(const Globals &, typename Config::instruction_t &, int &query) {

            const int n_idle = Config::NUM_PAGES - IDLE_LID0;
            return (query < n_idle) ? (IDLE_LID0 + query) : (query - n_idle);
        }
        static __device__ int init_semaphores(const Globals &, state<Config> &s) {
            for (int i = 0; i < SEM_COUNT; i++) {

                int cnt = (i >= STAGES && i <= 2 * STAGES) ? Config::NUM_CONSUMER_WARPS : 1;
                init_semaphore(s.semaphores()[i], cnt);
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            if (laneid() >= IDLE_LID0 && laneid() < Config::NUM_PAGES) {
                int pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }

            if (laneid() == 0)
                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
            warp::sync();

            float *scr = stats_base(s);
            for (int idx = laneid(); idx < inst.n_splits * dec::M_PAD; idx += 32) {
                const int sp = idx / dec::M_PAD, i = idx % dec::M_PAD;
                scr[sp * STATS_STRIDE + i]              = g.attn_stats[coord<>{sp, inst.head * 2 + 0, i}];
                scr[sp * STATS_STRIDE + dec::M_PAD + i] = g.attn_stats[coord<>{sp, inst.head * 2 + 1, i}];
            }
            warp::sync();
            warp::arrive(stats_ready(s));

            if (laneid() != 0) { warp::sync(); return; }

#pragma unroll
            for (int p = 0; p < TILE_PAGES; p++) s.wait_page_ready(s.pid(p));
            s.wait_page_ready(s.pid(OUT_LID));

            for (int sp = 0; sp < inst.n_splits; sp++) {
                int st = sp % STAGES;
                if (sp >= STAGES) ::megakernel::mk_semaphore_wait(tile_free(s, st), ((sp / STAGES) - 1) & 1);
                tma::expect_bytes(tile_arrived(s, st), sizeof(p_st));
                tma::load_async(tile_at(s, st), g.attn_partial,
                                {sp, 0, inst.head * COL_BLKS + inst.col_blk}, tile_arrived(s, st));
            }
            warp::sync();
        }
    };

    struct launcher { static __device__ void run(const Globals &, state<Config> &) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            const float *scr = stats_base(s);
            const int woff = kittens::warpid() * 16;

            ::megakernel::mk_semaphore_wait(stats_ready(s), 0);

            col_vec<rt_fl<16, HALF>> M, wsum, m_i, l_i, w;
            warp::neg_infty(M);
            warp::zero(wsum);
            for (int sp = 0; sp < inst.n_splits; sp++) {
                warp::load(m_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * STATS_STRIDE + woff));
                warp::max(M, M, m_i);
            }
            for (int sp = 0; sp < inst.n_splits; sp++) {
                warp::load(m_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * STATS_STRIDE + woff));
                warp::load(l_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * STATS_STRIDE + dec::M_PAD + woff));
                warp::sub(m_i, m_i, M);
                warp::exp2(m_i, m_i);
                warp::mul(w, l_i, m_i);
                warp::add(wsum, wsum, w);
            }

            rt_fl<16, HALF> acc, tmp;
            warp::zero(acc);
            for (int sp = 0; sp < inst.n_splits; sp++) {
                int st = sp % STAGES;
                ::megakernel::mk_semaphore_wait(tile_arrived(s, st), (sp / STAGES) & 1);
                warpgroup::load(tmp, tile_at(s, st));

                warp::load(m_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * STATS_STRIDE + woff));
                warp::load(l_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * STATS_STRIDE + dec::M_PAD + woff));
                warp::sub(m_i, m_i, M);
                warp::exp2(m_i, m_i);
                warp::mul(w, l_i, m_i);

                warp::mul_row(tmp, tmp, w);
                warp::add(acc, acc, tmp);
                warp::arrive(tile_free(s, st));
            }
            warp::div_row(acc, acc, wsum);

            s.wait_page_ready(s.pid(OUT_LID));
            warpgroup::store(out_at(s), acc);
            warp::sync();
            warp::arrive(out_ready(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(out_ready(s), 0);
            if (laneid() == 0) {

                tma::store_async(g.attn_out, out_at(s), {inst.head, inst.col_blk});
                tma::store_async_read_wait();
#pragma unroll
                for (int p = 0; p < TILE_PAGES; p++) s.finish_page(s.pid(p), Config::NUM_CONSUMER_WARPS);
                s.finish_page(s.pid(OUT_LID), Config::NUM_CONSUMER_WARPS);
            }
            tma::store_async_wait();
            warp::sync();
            if (laneid() == 0) {
                __threadfence();
                atomicAdd((uint32_t *)&g.Bar[coord<>{inst.layer, BarrierRow, 0}], 1);
            }
        }
    };
};

template <typename Config, typename Globals>
using ae_attn_partial_op =
    ae_attn_partial<Config, Globals, OPCODE_AE_ATTN_PARTIAL, BARRIER_ROW(OPCODE_AE_ATTN_PARTIAL)>;

template <typename Config, typename Globals>
using ae_attn_reduce_op =
    ae_attn_reduce<Config, Globals, OPCODE_AE_ATTN_REDUCE, BARRIER_ROW(OPCODE_AE_ATTN_REDUCE), 8>;

#ifndef OPCODE_AE_ATTN_REDUCE_ALT
#define OPCODE_AE_ATTN_REDUCE_ALT  8
#endif

template <typename Config, typename Globals, int ColBlks>
using ae_attn_reduce_alt_op =
    ae_attn_reduce<Config, Globals, OPCODE_AE_ATTN_REDUCE_ALT, BARRIER_ROW(OPCODE_AE_ATTN_REDUCE_ALT), ColBlks>;

}  // namespace pi0
