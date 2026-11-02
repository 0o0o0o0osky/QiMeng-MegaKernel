#pragma once

#include "kittens.cuh"
#include "common/gemma_globals.cuh"
#include "common/gemma_instruction_list.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"
#include "include/pipelines/wgmma_compat.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

template <typename Config, typename Globals, int Opcode, int BarrierRow>
struct gemma_attn_partial {
    static constexpr int opcode = Opcode;

    static constexpr int HD      = enc::HEAD_DIM;        // 256
    static constexpr int KV_BLK  = 64;
    static constexpr int NUM_WG  = Config::NUM_CONSUMER_WARPS / 4;       // 2
    static constexpr int KV_STAGES = 2;

    static_assert(NUM_WG == 2 && NUM_WG + 2 * KV_STAGES == 6,
                  "Partial layout requires two Q/output pages and two K/V stage pages");

    static constexpr float SCALE       = 0.0625f;
    static constexpr float SCALE_LOG2E = SCALE * 1.44269504088896340736f;

    using q_st  = st_bf<64, HD>;
    using kv_st = st_bf<KV_BLK, HD>;
    using o_st  = st_bf<64, HD>;

    struct parsed_instruction {
        int opcode, layer, q_tile, head, kv_split, dep_layer, dep_row, dep_expected;
        int kv_blk0;
        int kv_nblk;
        int seq_len;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), q_tile(i[2]), head(i[3]), kv_split(i[4]),
              dep_layer(i[5]), dep_row(i[6]), dep_expected(i[7]),
              kv_blk0(i[8]), kv_nblk(i[9]), seq_len(i[10]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &q_arrived(state<Config> &s)       { return s.semaphores()[0]; }
    __device__ static inline semaphore &kv_arrived(state<Config> &s, int st) { return s.semaphores()[1 + st]; }
    __device__ static inline semaphore &kv_free(state<Config> &s, int st) { return s.semaphores()[1 + KV_STAGES + st]; }
    __device__ static inline semaphore &out_ready(state<Config> &s)       { return s.semaphores()[1 + 2 * KV_STAGES]; }
    static constexpr int SEM_COUNT = 2 + 2 * KV_STAGES;

    __device__ static inline q_st  &q_at(state<Config> &s, int wg)  { return reinterpret_cast<q_st &>(s.pages[s.pid(wg)]); }
    __device__ static inline o_st  &o_at(state<Config> &s, int wg)  { return reinterpret_cast<o_st &>(s.pages[s.pid(wg)]); }
    __device__ static inline kv_st &k_at(state<Config> &s, int st)  { return reinterpret_cast<kv_st &>(s.pages[s.pid(2 + 2 * st)]); }
    __device__ static inline kv_st &v_at(state<Config> &s, int st)  { return reinterpret_cast<kv_st &>(s.pages[s.pid(3 + 2 * st)]); }

    struct controller {
        static __device__ int release_lid(const Globals &, typename Config::instruction_t &, int &query) {

            constexpr int order[6] = {2, 3, 4, 5, 0, 1};
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
            if (laneid() != 0) { warp::sync(); return; }

            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

            tma::expect_bytes(q_arrived(s), NUM_WG * sizeof(q_st));
            for (int wg = 0; wg < NUM_WG; wg++) {
                s.wait_page_ready(s.pid(wg));
                tma::load_async(q_at(s, wg), g.qkv_buf,
                                {inst.q_tile * NUM_WG + wg, inst.head}, q_arrived(s));
            }

#pragma unroll
            for (int st = 0; st < KV_STAGES; st++) {
                s.wait_page_ready(s.pid(2 + 2 * st));
                s.wait_page_ready(s.pid(3 + 2 * st));
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
            const int wg = warpgroup::groupid();

            rt_fl<16, HD> o;
            warp::zero(o);
            col_vec<rt_fl<16, KV_BLK>> mx, nrm, prev, corr;
            warp::neg_infty(mx);
            warp::zero(nrm);

            ::megakernel::mk_semaphore_wait(q_arrived(s), 0);
            const float scale = SCALE_LOG2E;

            for (int b = 0; b < inst.kv_nblk; b++) {
                const int st = b % KV_STAGES;
                ::megakernel::mk_semaphore_wait(kv_arrived(s, st), (b / KV_STAGES) & 1);

                rt_fl<16, KV_BLK> att;
                warpgroup::mm_ABt(att, q_at(s, wg), k_at(s, st));
                warpgroup::mma_async_wait();
                warp::mul(att, att, scale);

                const int valid = inst.seq_len - (inst.kv_blk0 + b) * KV_BLK;
                if (valid < KV_BLK)
                    warp::apply(att, att, [valid] __device__(int, int col, float x) {
                        return col < valid ? x : -INFINITY;
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
                s.warp_finish_page(s.pid(2 + 2 * st), 1);   // K
                s.warp_finish_page(s.pid(3 + 2 * st), 1);   // V
            }

            warp::div_row(o, o, nrm);
            warpgroup::store(o_at(s, wg), o);

            float *scr = reinterpret_cast<float *>(s.scratch());
            auto &m_sm = *reinterpret_cast<sv_fl<64> *>(scr + wg * 64);
            auto &l_sm = *reinterpret_cast<sv_fl<64> *>(scr + 128 + wg * 64);
            warpgroup::store(m_sm, mx);
            warpgroup::store(l_sm, nrm);

            warpgroup::sync(wg + 4);

            if (warpgroup::warpid() == 0) {
                const int row0 = inst.q_tile * enc::TILE_M + wg * 64;
                const int hb = inst.head * 2;
                for (int i = laneid(); i < 64; i += 32) {
                    g.attn_stats[coord<>{inst.kv_split, hb + 0, row0 + i}] = m_sm[i];
                    g.attn_stats[coord<>{inst.kv_split, hb + 1, row0 + i}] = l_sm[i];
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
                for (int wg = 0; wg < NUM_WG; wg++)
                    tma::store_async(g.attn_partial, o_at(s, wg),
                                     {inst.kv_split, inst.q_tile * NUM_WG + wg, inst.head});
                tma::store_async_read_wait();
                for (int wg = 0; wg < NUM_WG; wg++)
                    s.finish_page(s.pid(wg), Config::NUM_CONSUMER_WARPS);
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

template <typename Config, typename Globals, int Opcode, int BarrierRow>
struct gemma_attn_reduce {
    static constexpr int opcode = Opcode;
    static constexpr int HALF   = enc::HEAD_DIM / 2;        // 128
    static constexpr int NUM_WG = Config::NUM_CONSUMER_WARPS / 4;
    static constexpr int STAGES = 2;

    static constexpr int MAX_SPLITS = Config::SCRATCH_BYTES / 4 / 256;
    static_assert(MAX_SPLITS >= 4, "Scratch storage cannot hold the (m, l) pairs for all splits");

    using p_st = st_bf<64, HALF>;

    struct parsed_instruction {
        int opcode, layer, q_tile, head, half, dep_layer, dep_row, dep_expected;
        int n_splits;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), q_tile(i[2]), head(i[3]), half(i[4]),
              dep_layer(i[5]), dep_row(i[6]), dep_expected(i[7]), n_splits(i[8]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &tile_arrived(state<Config> &s, int st) { return s.semaphores()[st]; }
    __device__ static inline semaphore &tile_free(state<Config> &s, int st)    { return s.semaphores()[STAGES + st]; }
    __device__ static inline semaphore &out_ready(state<Config> &s)            { return s.semaphores()[2 * STAGES]; }

    __device__ static inline semaphore &stats_ready(state<Config> &s)          { return s.semaphores()[2 * STAGES + 1]; }
    static constexpr int SEM_COUNT = 2 * STAGES + 2;

    __device__ static inline p_st &tile_at(state<Config> &s, int st, int wg) {
        return reinterpret_cast<p_st *>(&s.pages[s.pid(st)])[wg];
    }
    __device__ static inline p_st &out_at(state<Config> &s, int wg) {
        return reinterpret_cast<p_st *>(&s.pages[s.pid(2)])[wg];
    }

    __device__ static inline float *stats_base(state<Config> &s) {
        return reinterpret_cast<float *>(s.scratch());
    }

    struct controller {
        static __device__ int release_lid(const Globals &, typename Config::instruction_t &, int &query) {
            constexpr int order[6] = {3, 4, 5, 0, 1, 2};
            return order[query];
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
            if (laneid() >= 3 && laneid() < Config::NUM_PAGES) {
                int pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }

            if (laneid() == 0)
                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
            warp::sync();

            float *scr = stats_base(s);
            const int row0 = inst.q_tile * enc::TILE_M;
            for (int idx = laneid(); idx < inst.n_splits * enc::TILE_M; idx += 32) {
                const int sp = idx / enc::TILE_M, i = idx % enc::TILE_M;
                scr[sp * 256 + i]       = g.attn_stats[coord<>{sp, inst.head * 2 + 0, row0 + i}];
                scr[sp * 256 + 128 + i] = g.attn_stats[coord<>{sp, inst.head * 2 + 1, row0 + i}];
            }
            warp::sync();
            warp::arrive(stats_ready(s));

            if (laneid() != 0) { warp::sync(); return; }

            s.wait_page_ready(s.pid(0));
            s.wait_page_ready(s.pid(1));
            s.wait_page_ready(s.pid(2));
            for (int sp = 0; sp < inst.n_splits; sp++) {
                int st = sp % STAGES;
                if (sp >= STAGES) ::megakernel::mk_semaphore_wait(tile_free(s, st), ((sp / STAGES) - 1) & 1);
                tma::expect_bytes(tile_arrived(s, st), NUM_WG * sizeof(p_st));
                for (int wg = 0; wg < NUM_WG; wg++)
                    tma::load_async(tile_at(s, st, wg), g.attn_partial,
                                    {sp, inst.q_tile * NUM_WG + wg, inst.head * 2 + inst.half},
                                    tile_arrived(s, st));
            }
            warp::sync();
        }
    };

    struct launcher { static __device__ void run(const Globals &, state<Config> &) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            const int wg = warpgroup::groupid();
            const float *scr = stats_base(s);
            const int woff = kittens::warpid() * 16;

            ::megakernel::mk_semaphore_wait(stats_ready(s), 0);

            col_vec<rt_fl<16, HALF>> M, wsum, m_i, l_i, w;
            warp::neg_infty(M);
            warp::zero(wsum);
            for (int sp = 0; sp < inst.n_splits; sp++) {
                warp::load(m_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * 256 + woff));
                warp::max(M, M, m_i);
            }
            for (int sp = 0; sp < inst.n_splits; sp++) {
                warp::load(m_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * 256 + woff));
                warp::load(l_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * 256 + 128 + woff));
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
                warpgroup::load(tmp, tile_at(s, st, wg));

                warp::load(m_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * 256 + woff));
                warp::load(l_i, *reinterpret_cast<const sv_fl<16> *>(scr + sp * 256 + 128 + woff));
                warp::sub(m_i, m_i, M);
                warp::exp2(m_i, m_i);
                warp::mul(w, l_i, m_i);

                warp::mul_row(tmp, tmp, w);
                warp::add(acc, acc, tmp);
                warp::arrive(tile_free(s, st));
            }
            warp::div_row(acc, acc, wsum);

            s.wait_page_ready(s.pid(2));
            warpgroup::store(out_at(s, wg), acc);
            warp::sync();
            warp::arrive(out_ready(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(out_ready(s), 0);
            if (laneid() == 0) {
                for (int wg = 0; wg < NUM_WG; wg++)
                    tma::store_async(g.attn_out, out_at(s, wg),
                                     {inst.q_tile * NUM_WG + wg, inst.head * 2 + inst.half});
                tma::store_async_read_wait();
                s.finish_page(s.pid(0), Config::NUM_CONSUMER_WARPS);
                s.finish_page(s.pid(1), Config::NUM_CONSUMER_WARPS);
                s.finish_page(s.pid(2), Config::NUM_CONSUMER_WARPS);
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
using gemma_attn_reduce_op =
    gemma_attn_reduce<Config, Globals, OPCODE_GEMMA_ATTN_REDUCE, BARRIER_ROW(OPCODE_GEMMA_ATTN_REDUCE)>;

template <typename Config, typename Globals>
using gemma_attn_partial_op =
    gemma_attn_partial<Config, Globals, OPCODE_GEMMA_ATTN_PARTIAL, BARRIER_ROW(OPCODE_GEMMA_ATTN_PARTIAL)>;

}  // namespace pi0
