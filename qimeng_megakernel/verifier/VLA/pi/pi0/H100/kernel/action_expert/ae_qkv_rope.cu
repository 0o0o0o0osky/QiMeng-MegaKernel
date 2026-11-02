#pragma once

#include "kittens.cuh"
#include "common/ae_globals.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"
#include "include/pipelines/matmul_pipeline.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

template <typename Config, typename Globals, int Opcode, int _tile_n = 128>
struct ae_qkv_rope {
    static constexpr int opcode = Opcode;
    static constexpr int barrier_row = BARRIER_ROW(Opcode);

    static constexpr int TILE_N     = _tile_n;
    static constexpr int Q_TILES_N  = dec::NUM_Q_HEADS * dec::HEAD_DIM / TILE_N;   // 16 / 32
    static constexpr int KV_TILES_N = dec::HEAD_DIM / TILE_N;                      //  2 /  4
    static constexpr int K_TILE0    = Q_TILES_N;                                   // 16 / 32
    static constexpr int V_TILE0    = Q_TILES_N + KV_TILES_N;                      // 18 / 36

    static constexpr int COLS_PER_HEAD = dec::HEAD_DIM / TILE_N;                   //  2 /  4
    static_assert(dec::HEAD_DIM % TILE_N == 0, "HEAD_DIM must be divisible by TILE_N so that each RoPE pair stays within one tile");
    static_assert(V_TILE0 + KV_TILES_N == dec::QKV_OUT / TILE_N, "QKV tile partition is inconsistent");

    struct parsed_instruction {
        int opcode, plane, local_row, local_col, dep_layer, dep_row, dep_expected,
            bar_layer, kv_row_blk, norm_sel;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), plane(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), bar_layer(i[7]),
              kv_row_blk(i[9]), norm_sel(i[10]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}

        __device__ inline bool is_k() const { return local_col >= K_TILE0 && local_col < V_TILE0; }
        __device__ inline bool is_v() const { return local_col >= V_TILE0; }
        __device__ inline bool rope() const { return local_col < V_TILE0; }

        __device__ inline int  rope_col() const { return local_col % COLS_PER_HEAD; }
    };

    struct dep_waiter {
        __device__ static inline void gmem_wait(const Globals &g, state<Config> &s,
                                                const parsed_instruction &inst) {
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
        }
    };
    using pipe = matmul_pipeline<Config, Globals, parsed_instruction, dep_waiter,
                                 &Globals::x, &Globals::qkv_w, dec::WIDTH_K_ITERS, TILE_N>;
    static_assert(pipe::TILE_N == TILE_N, "The output tile width must match the pipeline tile width");
    static_assert(pipe::MATMULS_PER_STAGE == 1, "The Action Expert uses one warpgroup and one M=64 output tile");

    __device__ static inline semaphore &output_arrived(state<Config> &s) { return s.semaphores()[pipe::SEM_COUNT]; }
    __device__ static inline semaphore &rope_arrived(state<Config> &s)   { return s.semaphores()[pipe::SEM_COUNT + 1]; }

    static constexpr int SCALE_OFF_FLOATS = 1024;
    __device__ static inline float *row_scale_base(state<Config> &s) {
        return reinterpret_cast<float *>(s.scratch()) + SCALE_OFF_FLOATS;
    }
    __device__ static inline sv_fl<16> &row_scale_smem(state<Config> &s) {
        return *reinterpret_cast<sv_fl<16> *>(row_scale_base(s) + kittens::warpid() * 16);
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &inst, int &query) {
            return pipe::template release_lid<2>(g, inst, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if (laneid() == 0) {
                init_semaphore(output_arrived(s), Config::NUM_CONSUMER_WARPS);
                init_semaphore(rope_arrived(s), 1);
            }
            return pipe::init_semaphores(s) + 2;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
            {
                const float *src = &g.rms_factor[coord<>{inst.norm_sel, 0}];
                float *dst = row_scale_base(s);
#pragma unroll
                for (int i = laneid(); i < pipe::TILE_M; i += 32) dst[i] = src[i];
            }
            pipe::template loader_loop<dim::ROW>(s, g, inst.plane);

            if (inst.rope()) {

                if (laneid() == 0) {
                    s.wait_page_ready(pipe::get_output_page(s, 0));
                    tma::expect_bytes(rope_arrived(s), sizeof(typename pipe::o_st));
                    tma::load_async(pipe::get_output_tile(s, 0), g.rope,
                                    {0, inst.rope_col()}, rope_arrived(s));
                }
                warp::sync();
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            rt_fl<16, pipe::TILE_N> acc = pipe::template matmul_loop<2>(s, g);

            {
                kittens::col_vec<rt_fl<16, pipe::TILE_N>> sc;
                warp::load(sc, row_scale_smem(s));
                warp::mul_row(acc, acc, sc);
            }

            if (inst.rope()) {
                ::megakernel::mk_semaphore_wait(rope_arrived(s), 0);
                rt_fl<16, pipe::TILE_N> rp;
                warpgroup::load(rp, pipe::get_output_tile(s, 0));

#pragma unroll
                for (int i = 0; i < decltype(acc)::height; i++)
#pragma unroll
                    for (int j = 0; j < decltype(acc)::width; j++)
#pragma unroll
                        for (int k = 0; k < decltype(acc)::packed_per_tile; k++) {
                            const float x0 = acc.tiles[i][j].data[k].x;
                            const float x1 = acc.tiles[i][j].data[k].y;
                            const float c  = rp.tiles[i][j].data[k].x;
                            const float sn = rp.tiles[i][j].data[k].y;
                            acc.tiles[i][j].data[k].x = fmaf(-x1, sn, x0 * c);
                            acc.tiles[i][j].data[k].y = fmaf(x0, sn, x1 * c);
                        }
            }

            warpgroup::store(pipe::get_output_tile(s, 0), acc);
            warp::sync();
            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(output_arrived(s), 0);

            if (laneid() == 0) {
                auto &tile = pipe::get_output_tile(s, 0);
                tma::store_async(g.qkv_buf, tile, {0, inst.local_col});

                if (inst.is_k())
                    tma::store_async(g.k_cache, tile,
                                     {inst.plane, inst.kv_row_blk, inst.local_col - K_TILE0});
                else if (inst.is_v())
                    tma::store_async(g.v_cache, tile,
                                     {inst.plane, inst.kv_row_blk, inst.local_col - V_TILE0});
                tma::store_async_read_wait();
                s.finish_page(pipe::get_output_page(s, 0), Config::NUM_CONSUMER_WARPS);
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
using ae_qkv_rope_op = ae_qkv_rope<Config, Globals, OPCODE_AE_QKV_ROPE, 32>;

#ifndef OPCODE_AE_QKV_ROPE_ALT
#define OPCODE_AE_QKV_ROPE_ALT  7
#endif

template <typename Config, typename Globals, int TileN>
using ae_qkv_rope_alt_op = ae_qkv_rope<Config, Globals, OPCODE_AE_QKV_ROPE_ALT, TileN>;

}  // namespace pi0
