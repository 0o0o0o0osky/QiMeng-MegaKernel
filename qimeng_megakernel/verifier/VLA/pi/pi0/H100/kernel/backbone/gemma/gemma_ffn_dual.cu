#pragma once

#include "kittens.cuh"
#include "common/gemma_globals.cuh"
#include "common/gemma_instruction_list.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

template <typename Config, typename Globals, auto A_Ptr, int Opcode, int BarrierRow>
struct gemma_ffn_dual {
    static constexpr int opcode = Opcode;

    static constexpr int STAGES  = PIPELINE_K_STAGES;                    // 4
    static constexpr int K_ITERS = enc::WIDTH_K_ITERS;                   // 32
    static constexpr int NUM_WG  = Config::NUM_CONSUMER_WARPS / 4;       // 2
    static constexpr int TILE_M  = 64 * NUM_WG;                          // 128
    static constexpr int TILE_N  = 128;
    static_assert(STAGES == 4 && NUM_WG == 2, "The page table requires four stages and two warpgroups");
    static_assert(K_ITERS % STAGES == 0, "K must be padded to a multiple of K_GRANULARITY");

    using a_st = st_bf<64, PIPELINE_K_DIM>;      //  8 KB
    using b_st = st_bf<TILE_N, PIPELINE_K_DIM>;  // 16 KB
    using o_st = st_bf<64, TILE_N>;              // 16 KB

    static constexpr int STAGE_BYTES = (2 * sizeof(a_st)) + 2 * sizeof(b_st);

    struct parsed_instruction {
        int opcode, layer, local_row, local_col, dep_layer, dep_row, dep_expected, norm_sel;

        int band_slot;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), norm_sel(i[7]),
              band_slot(i[8]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &inputs_arrived(state<Config> &s, int st)  { return s.semaphores()[st]; }
    __device__ static inline semaphore &inputs_finished(state<Config> &s, int st) { return s.semaphores()[STAGES + st]; }
    __device__ static inline semaphore &output_arrived(state<Config> &s)          { return s.semaphores()[2 * STAGES]; }

    __device__ static inline semaphore &factor_ready(state<Config> &s)            { return s.semaphores()[2 * STAGES + 1]; }
    static constexpr int SEM_COUNT = 2 * STAGES + 2;

    __device__ static inline int a_page(state<Config> &s, int st)  { return s.pid(st / 2); }
    __device__ static inline int bg_page(state<Config> &s, int st) { return s.pid(2 + st / 2); }
    __device__ static inline int bu_page(state<Config> &s, int st) { return s.pid(4 + st / 2); }
    __device__ static inline a_st (&a_tile(state<Config> &s, int st))[2] {
        return reinterpret_cast<a_st(*)[2]>(&s.pages[a_page(s, st)])[st & 1];
    }
    __device__ static inline b_st &bg_tile(state<Config> &s, int st) {
        return reinterpret_cast<b_st *>(&s.pages[bg_page(s, st)])[st & 1];
    }
    __device__ static inline b_st &bu_tile(state<Config> &s, int st) {
        return reinterpret_cast<b_st *>(&s.pages[bu_page(s, st)])[st & 1];
    }

    __device__ static inline o_st &out_tile(state<Config> &s, int wg) {
        return reinterpret_cast<o_st *>(&s.pages[s.pid(0)])[wg];
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

            constexpr int order[6] = {2, 4, 3, 5, 1, 0};
            return order[query];
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
            if (laneid() != 0) { warp::sync(); return; }

            for (int iter = 0; iter < K_ITERS; iter++) {
                const int st = iter % STAGES;
                ::megakernel::mk_semaphore_wait(inputs_finished(s, st), (iter % (2 * STAGES)) < STAGES);

                auto &sem = inputs_arrived(s, st);
                tma::expect_bytes(sem, STAGE_BYTES);

                if (iter < STAGES) {
                    s.wait_page_ready(bg_page(s, st));
                    s.wait_page_ready(bu_page(s, st));
                }
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    bg_tile(s, st), g.gate_w, {inst.layer, inst.local_col, iter}, sem);
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    bu_tile(s, st), g.up_w, {inst.layer, inst.local_col, iter}, sem);

                if (iter == 0)
                    gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

                if ((iter < STAGES) && (iter % 2 == 0)) s.wait_page_ready(a_page(s, st));
                auto(&a_smem)[2] = a_tile(s, st);
#pragma unroll
                for (int i = 0; i < NUM_WG; i++)
                    tma::load_async<dim::ROW, cache_policy::NORMAL>(
                        a_smem[i], g.*A_Ptr, {NUM_WG * inst.local_row + i, iter}, sem);
            }
            warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
            const int r0 = inst.local_row * TILE_M;
            float *dst = row_scale_base(s);
            if (inst.norm_sel < 2) {

                const float *src = &g.rms_factor[coord<>{inst.norm_sel, r0}];
#pragma unroll
                for (int i = laneid(); i < TILE_M; i += 32) dst[i] = src[i];
            } else {

                const int base = (inst.norm_sel == 2) ? enc::RMS_STATS_ATTN_ROW0
                                                      : enc::RMS_STATS_FFN_ROW0;
                for (int i = laneid(); i < TILE_M; i += 32) {
                    float ssum = 0.f;
#pragma unroll
                    for (int j = 0; j < enc::RMS_STATS_TILES; j++)
                        ssum += g.rms_factor[coord<>{base + j, r0 + i}];
                    dst[i] = rsqrtf(ssum * (1.0f / enc::WIDTH) + GEMMA_RMS_EPS);
                }
            }
            warp::sync();
            warp::arrive(factor_ready(s));
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            const int wg = warpgroup::groupid();
            rt_fl<16, TILE_N> g_acc, u_acc;

            {
                ::megakernel::mk_semaphore_wait(inputs_arrived(s, 0), 0);
                auto(&a_smem)[2] = a_tile(s, 0);
                warpgroup::mm_ABt(g_acc, a_smem[wg], bg_tile(s, 0));
                warpgroup::mm_ABt(u_acc, a_smem[wg], bu_tile(s, 0));
                warpgroup::mma_async_wait();
                warpgroup::arrive(inputs_finished(s, 0));
            }
            for (int i = 1; i < K_ITERS; i++) {
                const int st = i % STAGES;
                ::megakernel::mk_semaphore_wait(inputs_arrived(s, st), (i % (2 * STAGES)) >= STAGES);
                auto(&a_smem)[2] = a_tile(s, st);
                warpgroup::mma_ABt(g_acc, a_smem[wg], bg_tile(s, st));
                warpgroup::mma_ABt(u_acc, a_smem[wg], bu_tile(s, st));
                warpgroup::mma_async_wait();
                warpgroup::arrive(inputs_finished(s, st));

                if (i == K_ITERS - 3) {
                    s.warp_finish_page(s.pid(2), 1);   // Wgate stage 0,1
                    s.warp_finish_page(s.pid(4), 1);   // Wup   stage 0,1
                } else if (i == K_ITERS - 1) {
                    s.warp_finish_page(s.pid(3), 1);   // Wgate stage 2,3
                    s.warp_finish_page(s.pid(5), 1);   // Wup   stage 2,3
                    s.warp_finish_page(s.pid(1), 1);   // A stage 2,3
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
                        float u = 1.5957691216f * (x + 0.044715f * x * x * x);
                        u_acc.tiles[i][j].data[k].x *= x / (1.f + __expf(-u));
                        x = g_acc.tiles[i][j].data[k].y;
                        u = 1.5957691216f * (x + 0.044715f * x * x * x);
                        u_acc.tiles[i][j].data[k].y *= x / (1.f + __expf(-u));
                    }

            group<Config::NUM_CONSUMER_WARPS>::sync(6);

            warpgroup::store(out_tile(s, wg), u_acc);
            warp::sync();
            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(output_arrived(s), 0);
            if (laneid() == 0) {
#pragma unroll
                for (int i = 0; i < NUM_WG; i++)
                    tma::store_async(g.up_buf, out_tile(s, i),
                                     {NUM_WG * inst.local_row + i, inst.local_col});
                tma::store_async_read_wait();
                s.finish_page(s.pid(0), Config::NUM_CONSUMER_WARPS);
            }
            tma::store_async_wait();
            warp::sync();
            if (laneid() == 0) {
                __threadfence();
                atomicAdd((uint32_t *)&g.Bar[coord<>{inst.layer, BarrierRow, 0}], 1);
                if (inst.band_slot > 0)
                    atomicAdd((uint32_t *)&g.Bar[coord<>{inst.layer, BarrierRow, inst.band_slot}], 1);
            }
        }
    };
};

template <typename Config, typename Globals>
using gemma_ffn_dual_op =
    gemma_ffn_dual<Config, Globals, &Globals::encoder_x, OPCODE_GEMMA_GATE, BARRIER_ROW(OPCODE_GEMMA_GATE)>;

}  // namespace pi0
