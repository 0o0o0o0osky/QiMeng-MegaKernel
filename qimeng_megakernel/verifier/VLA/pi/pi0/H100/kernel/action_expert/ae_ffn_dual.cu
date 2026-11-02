#pragma once

#include "kittens.cuh"
#include "common/ae_globals.cuh"
#include "common/ae_instruction_list.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"

#ifndef OPCODE_AE_FFN_DUAL
#define OPCODE_AE_FFN_DUAL 10
#endif

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

template <typename Config, typename Globals, int Opcode, int TILE_N = 32, int STAGES = 4>
struct ae_ffn_dual {
    static constexpr int opcode      = Opcode;
    static constexpr int barrier_row = BARRIER_ROW(Opcode);

    static constexpr int K_ITERS = dec::WIDTH_K_ITERS;                 // 16
    static constexpr int NUM_WG  = Config::NUM_CONSUMER_WARPS / 4;     // 1
    static constexpr int TILE_M  = 64 * NUM_WG;                        // 64
    static_assert(NUM_WG == 1, "action_expert_config requires exactly one warpgroup");
    static_assert(K_ITERS % STAGES == 0, "K_ITERS must be divisible by STAGES to preserve phase parity");
    static_assert(TILE_N % 32 == 0 && TILE_N <= 128, "WGMMA requires N >= 32");

    using a_st = st_bf<64, PIPELINE_K_DIM>;      // 8 KB
    using b_st = st_bf<TILE_N, PIPELINE_K_DIM>;  // TILE_N=32 -> 4 KB
    using o_st = st_bf<64, TILE_N>;              // TILE_N=32 -> 4 KB

    static constexpr int A_PER_PAGE = Config::PAGE_SIZE / sizeof(a_st);   // 4
    static constexpr int B_PER_PAGE = Config::PAGE_SIZE / sizeof(b_st);   // 8
    static constexpr int A_PAGES    = (STAGES + A_PER_PAGE - 1) / A_PER_PAGE;  // STAGES=8 -> 2
    static_assert(STAGES <= B_PER_PAGE, "Each weight stream must fit in one page");

    static constexpr int BG_LID    = A_PAGES;
    static constexpr int BU_LID    = A_PAGES + 1;
    static constexpr int OUT_LID   = A_PAGES + 2;
    static constexpr int IDLE_LID0 = A_PAGES + 3;
    static_assert(IDLE_LID0 <= Config::NUM_PAGES, "Insufficient pages for A, Wgate, Wup, and output");

    static constexpr int STAGE_BYTES = sizeof(a_st) + 2 * sizeof(b_st);

    struct parsed_instruction {
        int opcode, plane, local_row, local_col, dep_layer, dep_row, dep_expected,
            bar_layer, out_row, out_col, norm_sel;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), plane(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), bar_layer(i[7]),
              out_row(i[8]), out_col(i[9]), norm_sel(i[10]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &inputs_arrived(state<Config> &s, int st)  { return s.semaphores()[st]; }
    __device__ static inline semaphore &inputs_finished(state<Config> &s, int st) { return s.semaphores()[STAGES + st]; }
    __device__ static inline semaphore &output_arrived(state<Config> &s)          { return s.semaphores()[2 * STAGES]; }
    __device__ static inline semaphore &factor_ready(state<Config> &s)            { return s.semaphores()[2 * STAGES + 1]; }
    static constexpr int SEM_COUNT = 2 * STAGES + 2;

    __device__ static inline a_st &a_tile(state<Config> &s, int st) {
        return reinterpret_cast<a_st *>(&s.pages[s.pid(st / A_PER_PAGE)])[st % A_PER_PAGE];
    }
    __device__ static inline b_st &bg_tile(state<Config> &s, int st) {
        return reinterpret_cast<b_st *>(&s.pages[s.pid(BG_LID)])[st];
    }
    __device__ static inline b_st &bu_tile(state<Config> &s, int st) {
        return reinterpret_cast<b_st *>(&s.pages[s.pid(BU_LID)])[st];
    }
    __device__ static inline o_st &out_tile(state<Config> &s) {
        return reinterpret_cast<o_st &>(s.pages[s.pid(OUT_LID)]);
    }

    static constexpr int SCALE_OFF_FLOATS = 1024;
    __device__ static inline float *row_scale_base(state<Config> &s) {
        return reinterpret_cast<float *>(s.scratch()) + SCALE_OFF_FLOATS;
    }
    __device__ static inline sv_fl<16> &row_scale_smem(state<Config> &s) {
        return *reinterpret_cast<sv_fl<16> *>(row_scale_base(s) + kittens::warpid() * 16);
    }

    struct controller {
        static __device__ int release_lid(const Globals &, typename Config::instruction_t &, int &query) {

            int q = query;
            const int n_idle = Config::NUM_PAGES - IDLE_LID0;
            if (q < n_idle) return IDLE_LID0 + q;
            q -= n_idle;
            if (q < A_PAGES) return q;
            q -= A_PAGES;
            if (q == 0) return BG_LID;
            if (q == 1) return BU_LID;
            return OUT_LID;
        }
        static __device__ int init_semaphores(const Globals &, state<Config> &s) {
            if (laneid() < STAGES) {
                init_semaphore(inputs_arrived(s, laneid()), 1);
                init_semaphore(inputs_finished(s, laneid()), NUM_WG);
            }
            if (laneid() == 0) {
                init_semaphore(output_arrived(s), Config::NUM_CONSUMER_WARPS);
                init_semaphore(factor_ready(s), 1);
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
            if (laneid() != 0) { warp::sync(); return; }

            for (int iter = 0; iter < K_ITERS; iter++) {
                const int st = iter % STAGES;
                if (iter >= STAGES) ::megakernel::mk_semaphore_wait(inputs_finished(s, st), ((iter / STAGES) & 1) == 0);

                auto &sem = inputs_arrived(s, st);
                tma::expect_bytes(sem, STAGE_BYTES);

                if (iter < STAGES) {
                    s.wait_page_ready(s.pid(BG_LID));
                    s.wait_page_ready(s.pid(BU_LID));
                }
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    bg_tile(s, st), g.gate_w, {inst.plane, inst.local_col, iter}, sem);
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    bu_tile(s, st), g.up_w, {inst.plane, inst.local_col, iter}, sem);

                if (iter == 0)
                    gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

                if (iter < STAGES && (iter % A_PER_PAGE) == 0)
                    s.wait_page_ready(s.pid(iter / A_PER_PAGE));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    a_tile(s, st), g.x, {inst.local_row, iter}, sem);
            }
            warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
            const float *src = &g.rms_factor[coord<>{inst.norm_sel, 0}];
            float *dst = row_scale_base(s);
#pragma unroll
            for (int i = laneid(); i < TILE_M; i += 32) dst[i] = src[i];
            warp::sync();
            warp::arrive(factor_ready(s));
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            rt_fl<16, TILE_N> g_acc, u_acc;

            {
                ::megakernel::mk_semaphore_wait(inputs_arrived(s, 0), 0);
                warpgroup::mm_ABt(g_acc, a_tile(s, 0), bg_tile(s, 0));
                warpgroup::mm_ABt(u_acc, a_tile(s, 0), bu_tile(s, 0));
                warpgroup::mma_async_wait();
                warpgroup::arrive(inputs_finished(s, 0));
            }
            for (int i = 1; i < K_ITERS; i++) {
                const int st = i % STAGES;
                ::megakernel::mk_semaphore_wait(inputs_arrived(s, st), (i / STAGES) & 1);
                warpgroup::mma_ABt(g_acc, a_tile(s, st), bg_tile(s, st));
                warpgroup::mma_ABt(u_acc, a_tile(s, st), bu_tile(s, st));
                warpgroup::mma_async_wait();
                warpgroup::arrive(inputs_finished(s, st));

                //       i == K_ITERS - STAGES + (b - 1)

#pragma unroll
                for (int p = 0; p < A_PAGES; p++) {
                    constexpr int per = A_PER_PAGE;
                    const int b = (p + 1) * per < STAGES ? (p + 1) * per : STAGES;
                    if (i == K_ITERS - STAGES + b - 1) s.warp_finish_page(s.pid(p), 1);
                }
                if (i == K_ITERS - 1) {
                    s.warp_finish_page(s.pid(BG_LID), 1);
                    s.warp_finish_page(s.pid(BU_LID), 1);
                }
            }

            {
                ::megakernel::mk_semaphore_wait(factor_ready(s), 0);
                kittens::col_vec<rt_fl<16, TILE_N>> sc;
                warp::load(sc, row_scale_smem(s));
                warp::mul_row(g_acc, g_acc, sc);
                warp::mul_row(u_acc, u_acc, sc);
            }

#pragma unroll
            for (int i = 0; i < decltype(g_acc)::height; i++)
#pragma unroll
                for (int j = 0; j < decltype(g_acc)::width; j++)
#pragma unroll
                    for (int k = 0; k < decltype(g_acc)::packed_per_tile; k++) {
                        float x = g_acc.tiles[i][j].data[k].x;
                        float z = 1.5957691216f * (x + 0.044715f * x * x * x);
                        u_acc.tiles[i][j].data[k].x *= x / (1.f + __expf(-z));
                        x = g_acc.tiles[i][j].data[k].y;
                        z = 1.5957691216f * (x + 0.044715f * x * x * x);
                        u_acc.tiles[i][j].data[k].y *= x / (1.f + __expf(-z));
                    }

            warpgroup::store(out_tile(s), u_acc);
            warp::sync();
            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(output_arrived(s), 0);
            if (laneid() == 0) {

                tma::store_async(g.ffn_buf, out_tile(s), {inst.out_row, inst.out_col});
                tma::store_async_read_wait();
                s.finish_page(s.pid(OUT_LID), Config::NUM_CONSUMER_WARPS);
            }
            tma::store_async_wait();
            warp::sync();
            if (laneid() == 0) {
                __threadfence();
                atomicAdd((uint32_t *)&g.Bar[coord<>{inst.bar_layer, barrier_row, 0}], 1);
            }
        }
    };
};

template <typename Config, typename Globals>
using ae_ffn_dual_op = ae_ffn_dual<Config, Globals, OPCODE_AE_FFN_DUAL>;

}  // namespace pi0
