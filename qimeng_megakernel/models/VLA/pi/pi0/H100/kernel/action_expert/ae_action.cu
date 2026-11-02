#pragma once

#include "kittens.cuh"
#include "common/ae_globals.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"
#include "include/pipelines/matmul_pipeline.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

static constexpr int AE_ACT_B_MLP_ROW   = 10;   // b_mlp     [1024]
static constexpr int AE_ACT_B_OUT_ROW   = 11;
static constexpr int AE_ACT_STATE_ROW   = 12;
static constexpr int AE_ACT_BIAS_ROWS   = 13;   // 0..9 = b_time[step]

#define AE_ACT_EPI(name, cb, sl, rs, sa, ro, f32)                                     \
    struct name { static constexpr bool col_bias = cb; static constexpr bool silu = sl; \
                  static constexpr bool row_scale = rs; static constexpr bool store_add = sa; \
                  static constexpr bool row_override = ro; static constexpr bool fp32_out = f32; \
                  static_assert(!(sa && f32), "The store_add target is the BF16 noise tensor"); }
//                       col_bias silu  row_scale store_add row_override fp32_out
AE_ACT_EPI(AeActEpiIn,   true,    true,  false,    false,    false,      false);
AE_ACT_EPI(AeActEpiMlp,  true,    false, false,    false,    true,       true );
AE_ACT_EPI(AeActEpiOut,  true,    false, true,     true,     false,      false);
#undef AE_ACT_EPI

template <typename Config, typename Globals, typename Epi,
          auto A_Ptr, auto B_Ptr, auto Out_Ptr,
          int K_ITERS, int Opcode, int TILE_N = 32>
struct ae_action_proj {
    static constexpr int opcode = Opcode;
    static constexpr int barrier_row = BARRIER_ROW(Opcode);

    struct parsed_instruction {
        int opcode, plane, local_row, local_col, dep_layer, dep_row, dep_expected,
            bar_layer, out_row, out_col, bias_row, norm_sel, n_valid_row;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), plane(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), bar_layer(i[7]),
              out_row(i[8]), out_col(i[9]), bias_row(i[10]), norm_sel(i[11]),
              n_valid_row(i[12]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    struct dep_waiter {
        __device__ static inline void gmem_wait(const Globals &g, state<Config> &s,
                                                const parsed_instruction &inst) {
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
        }
    };
    using pipe = matmul_pipeline<Config, Globals, parsed_instruction, dep_waiter, A_Ptr, B_Ptr,
                                 K_ITERS, TILE_N>;
    static_assert(pipe::MATMULS_PER_STAGE == 1, "The Action Expert requires exactly one warpgroup");

    __device__ static inline semaphore &output_arrived(state<Config> &s) {
        return s.semaphores()[pipe::SEM_COUNT];
    }

    using o_fl = st_fl<64, pipe::TILE_N>;
    static_assert(!Epi::fp32_out || sizeof(o_fl) <= Config::PAGE_SIZE, "The FP32 output tile must fit in one page");
    __device__ static inline o_fl &out_tile_fl(state<Config> &s) {
        return *reinterpret_cast<o_fl *>(&s.pages[s.pid(pipe::OUTPUT_LID)]);
    }

    static constexpr int RMS_OFF   = 1024;
    static constexpr int BIAS_OFF  = 1024 + 64;
    static constexpr int STATE_OFF = 1024 + 64 + 256;
    static_assert(STATE_OFF + TILE_N <= Config::SCRATCH_BYTES / 4, "Scratch storage is too small");
    __device__ static inline float *scr(state<Config> &s, int off) {
        return reinterpret_cast<float *>(s.scratch()) + off;
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &inst, int &query) {
            return pipe::template release_lid<2>(g, inst, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if (laneid() == 0) init_semaphore(output_arrived(s), Config::NUM_CONSUMER_WARPS);
            return pipe::init_semaphores(s) + 1;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            // Dependent global memory inputs are valid only after acquire/wait.
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

            if constexpr (Epi::col_bias) {
                const int c0 = inst.out_col * TILE_N;
                float *dst = scr(s, BIAS_OFF);
                for (int j = laneid(); j < TILE_N; j += 32)
                    dst[j] = g.act_bias[coord<>{inst.bias_row, c0 + j}];
            }
            if constexpr (Epi::row_override) {
                const int c0 = inst.out_col * TILE_N;
                float *dst = scr(s, STATE_OFF);
                for (int j = laneid(); j < TILE_N; j += 32)
                    dst[j] = g.act_bias[coord<>{AE_ACT_STATE_ROW, c0 + j}];
            }
            if constexpr (Epi::row_scale) {
                const float *src = &g.rms_factor[coord<>{inst.norm_sel, 0}];
                float *dst = scr(s, RMS_OFF);
                for (int i = laneid(); i < pipe::TILE_M; i += 32) dst[i] = src[i];
            }
            warp::sync();
            pipe::template loader_loop<dim::ROW>(s, g, inst.plane);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            rt_fl<16, pipe::TILE_N> acc = pipe::template matmul_loop<2>(s, g);

            const int row0 = kittens::warpid() * 16;

            if constexpr (Epi::row_scale) {
                const float *rms = scr(s, RMS_OFF);
                warp::apply(acc, acc, [rms, row0] __device__(int row, int, float x) {
                    return x * rms[row0 + row];
                });
            }
            if constexpr (Epi::col_bias) {
                const float *b = scr(s, BIAS_OFF);
                warp::apply(acc, acc, [b] __device__(int, int col, float x) { return x + b[col]; });
            }
            if constexpr (Epi::silu) {
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

                warp::apply(acc, acc, [] __device__(int, int, float x) {
                    return x / (1.0f + __expf(-x));
                });
            }
            if constexpr (Epi::row_override) {

                const float *st = scr(s, STATE_OFF);
                const int nv = inst.n_valid_row;
                warp::apply(acc, acc, [st, row0, nv] __device__(int row, int col, float x) {
                    const int gr = row0 + row;
                    if (gr == 0) return st[col];
                    return gr < nv ? x : 0.0f;
                });
            }

            if constexpr (Epi::fp32_out) warpgroup::store(out_tile_fl(s), acc);
            else                         warpgroup::store(pipe::get_output_tile(s, 0), acc);
            warp::sync();
            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(output_arrived(s), 0);
            auto &Out = g.*Out_Ptr;
            if (laneid() == 0) {

                if constexpr (Epi::fp32_out)
                    tma::store_async(Out, out_tile_fl(s), {inst.out_row, inst.out_col});
                else if constexpr (Epi::store_add)

                    tma::store_add_async(Out, pipe::get_output_tile(s, 0), {inst.out_row, inst.out_col});
                else
                    tma::store_async(Out, pipe::get_output_tile(s, 0), {inst.out_row, inst.out_col});
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
using ae_act_in_op = ae_action_proj<Config, Globals, AeActEpiIn,
                                    &Globals::noise, &Globals::act_w_in, &Globals::act_h,
                                    dec::SPLIT_K_ITERS, OPCODE_AE_ACT_IN>;

template <typename Config, typename Globals>
using ae_act_mlp_op = ae_action_proj<Config, Globals, AeActEpiMlp,
                                     &Globals::act_h, &Globals::act_w_mlp, &Globals::x_acc,
                                     dec::WIDTH_K_ITERS, OPCODE_AE_ACT_MLP>;

template <typename Config, typename Globals>
using ae_act_out_op = ae_action_proj<Config, Globals, AeActEpiOut,
                                     &Globals::x, &Globals::act_w_out, &Globals::noise,
                                     dec::WIDTH_K_ITERS, OPCODE_AE_ACT_OUT>;

}  // namespace pi0
