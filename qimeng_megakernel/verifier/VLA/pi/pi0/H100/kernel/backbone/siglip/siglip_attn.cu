#pragma once

#include "kittens.cuh"
#include "common/globals.cuh"
#include "common/instruction_list.cuh"
#include "include/megakernel.cuh"
#include "include/pipelines/wgmma_compat.cuh"
#include "include/gmem_dep.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

template <typename Config, typename Globals, int Opcode, int BarrierRow>
struct siglip_attn {
    static constexpr int opcode = Opcode;

    static constexpr int HD      = vis::HEAD_DIM_PAD;         // 80
    static constexpr int SEQ_V   = vis::TOKENS_PER_VIEW;      // 256
    static constexpr int KV_BLK  = 128;
    static constexpr int NUM_KV  = SEQ_V / KV_BLK;            // 2
    static constexpr int Q_ROWS  = 64;
    static constexpr int NUM_WG  = Config::NUM_CONSUMER_WARPS / 4;
    static_assert(NUM_WG == 2, "SigLIP attention requires two consumer warpgroups");

    static constexpr float SCALE = 0.11785113019775793f;  // 1/sqrt(72)
    static constexpr float SCALE_LOG2E = SCALE * 1.44269504088896340736f;

    using q_st  = st_bf<Q_ROWS, HD>;
    using kv_st = st_bf<KV_BLK, HD>;

    struct parsed_instruction {
        int opcode, layer, view, q_tile, head, dep_layer, dep_row, dep_expected;
        int flags;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), view(i[2]), q_tile(i[3]), head(i[4]),
              dep_layer(i[5]), dep_row(i[6]), dep_expected(i[7]), flags(i[11]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &q_arrived(state<Config> &s)          { return s.semaphores()[0]; }
    __device__ static inline semaphore &k_arrived(state<Config> &s, int b)   { return s.semaphores()[1 + b]; }
    __device__ static inline semaphore &v_arrived(state<Config> &s, int b)   { return s.semaphores()[1 + NUM_KV + b]; }
    __device__ static inline semaphore &output_arrived(state<Config> &s)     { return s.semaphores()[1 + 2 * NUM_KV]; }
    static constexpr int SEM_COUNT = 2 + 2 * NUM_KV;

    __device__ static inline q_st &q_tile_at(state<Config> &s, int wg) {
        return reinterpret_cast<q_st *>(&s.pages[s.pid(0)])[wg];
    }
    __device__ static inline kv_st &k_tile_at(state<Config> &s, int b) {
        return reinterpret_cast<kv_st &>(s.pages[s.pid(1 + b)]);
    }
    __device__ static inline kv_st &v_tile_at(state<Config> &s, int b) {
        return reinterpret_cast<kv_st &>(s.pages[s.pid(1 + NUM_KV + b)]);
    }
    __device__ static inline q_st &out_tile_at(state<Config> &s, int wg) {
        return reinterpret_cast<q_st *>(&s.pages[s.pid(5)])[wg];
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &inst, int &query) {
            constexpr int ret_order[6] = {1, 2, 3, 4, 0, 5};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            for (int i = 0; i < SEM_COUNT; i++) {
                if (i == SEM_COUNT - 1)
                    init_semaphore(s.semaphores()[i], Config::NUM_CONSUMER_WARPS);  // output_arrived
                else
                    init_semaphore(s.semaphores()[i], 1);
            }
            return SEM_COUNT;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            if (laneid() == 0) {
                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

                s.wait_page_ready(s.pid(0));

                tma::expect_bytes(q_arrived(s), NUM_WG * sizeof(q_st));
                for (int wg = 0; wg < NUM_WG; wg++) {
                    tma::load_async(q_tile_at(s, wg), g.qkv_buf,
                                    {inst.view * 4 + inst.q_tile * 2 + wg, inst.head}, q_arrived(s));
                }

#pragma unroll
                for (int b = 0; b < NUM_KV; b++) {
                    s.wait_page_ready(s.pid(1 + b));
                    kv_st &k = k_tile_at(s, b);
                    tma::expect(k_arrived(s, b), k);
                    tma::load_async(k, g.qkv_buf, {inst.view * 2 + b, vis::NUM_HEADS + inst.head},
                                    k_arrived(s, b));

                    s.wait_page_ready(s.pid(1 + NUM_KV + b));
                    kv_st &v = v_tile_at(s, b);
                    tma::expect(v_arrived(s, b), v);
                    tma::load_async(v, g.qkv_buf, {inst.view * 2 + b, 2 * vis::NUM_HEADS + inst.head},
                                    v_arrived(s, b));
                }
            }
            warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            const int wg = warpgroup::groupid();

            rt_fl<16, HD> o;
            warp::zero(o);
            col_vec<rt_fl<16, KV_BLK>> max_vec, norm_vec, prev_max, corr;
            warp::neg_infty(max_vec);
            warp::zero(norm_vec);

            ::megakernel::mk_semaphore_wait(q_arrived(s), 0);

#pragma unroll
            for (int b = 0; b < NUM_KV; b++) {
                ::megakernel::mk_semaphore_wait(k_arrived(s, b), 0);

                rt_fl<16, KV_BLK> att;
                warpgroup::mm_ABt(att, q_tile_at(s, wg), k_tile_at(s, b));
                warpgroup::mma_async_wait();

                const float scale = SCALE_LOG2E;
                warp::mul(att, att, scale);

                warp::copy(prev_max, max_vec);
                warp::row_max(max_vec, att, max_vec);      // running max
                warp::sub_row(att, att, max_vec);
                warp::exp2(att, att);

                warp::sub(corr, prev_max, max_vec);
                warp::exp2(corr, corr);
                warp::mul(norm_vec, norm_vec, corr);
                warp::row_sum(norm_vec, att, norm_vec);
                warp::mul_row(o, o, corr);

                rt_bf<16, KV_BLK> att_bf;
                warp::copy(att_bf, att);

                ::megakernel::mk_semaphore_wait(v_arrived(s, b), 0);

                wgmma_compat::mma_AB(o, att_bf, v_tile_at(s, b));
                warpgroup::mma_async_wait();
            }

            warp::div_row(o, o, norm_vec);

            s.wait_page_ready(s.pid(5));
            warpgroup::store(out_tile_at(s, wg), o);
            warp::sync();

#pragma unroll
            for (int b = 0; b < NUM_KV; b++) {
                s.warp_finish_page(s.pid(1 + b), 1);
                s.warp_finish_page(s.pid(1 + NUM_KV + b), 1);
            }
            s.warp_finish_page(s.pid(0), 1);

            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(output_arrived(s), 0);

            if (laneid() == 0) {
#pragma unroll
                for (int wg = 0; wg < NUM_WG; wg++) {
                    tma::store_async(g.attn_out, out_tile_at(s, wg),
                                     {inst.view * 4 + inst.q_tile * 2 + wg, inst.head});
                }
                tma::store_async_read_wait();
                s.finish_page(s.pid(5), Config::NUM_CONSUMER_WARPS);
            }
            tma::store_async_wait();
            warp::sync();
            if (laneid() == 0) {
                mk_phase_release((uint32_t *)&g.Bar[coord<>{inst.layer, BarrierRow, 0}],
                                 inst.flags & 8);
            }
        }
    };
};

template <typename Config, typename Globals>
using siglip_attn_op = siglip_attn<Config, Globals, OPCODE_SIGLIP_ATTN, BARRIER_ROW(OPCODE_SIGLIP_ATTN)>;

}  // namespace pi0
