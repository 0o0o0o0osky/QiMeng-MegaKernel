#pragma once

#include "kittens.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"
#include "include/pipelines/matmul_pipeline.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

#define AE_EPI(name, rs, sa, gl, gm, gg, ab) \
    struct name { static constexpr bool row_scale = rs; static constexpr bool store_add = sa; \
                  static constexpr bool gelu = gl; static constexpr bool gate_mul = gm; \
                  static constexpr bool gelu_gate = gg; static constexpr bool add_bf16 = ab; \
                  static_assert(!(sa && ab), "store_add and add_bf16 are mutually exclusive because their output tile types differ"); \
                  static_assert(!(gg && !gm), "gelu_gate requires gate_mul"); }
//                          row_scale store_add gelu  gate_mul gelu_gate add_bf16
AE_EPI(AeEpiStoreAdd,       false,    true,     false, false,  false,    false);
AE_EPI(AeEpiPlain,          false,    false,    false, false,  false,    false);
AE_EPI(AeEpiRowScale,       true,     false,    false, false,  false,    false);
AE_EPI(AeEpiRowScaleGelu,   true,     false,    true,  false,  false,    false);
AE_EPI(AeEpiRowScaleGate,   true,     false,    false, true,   false,    false);

AE_EPI(AeEpiRowScaleAdd,    true,     false,    false, false,  false,    true );
AE_EPI(AeEpiRowScaleGeluGateAdd,
                            true,     false,    false, true,   true,     true );
#undef AE_EPI

//       gl.cuh(71): "SKILL ISSUE: Requested a TMA descriptor for a type not
//                    initialized in the global layout."

#ifndef AE_OPROJ_TILE_N
#define AE_OPROJ_TILE_N 64
#endif

constexpr int ae_default_tile_n(int opcode) {
    return opcode == OPCODE_AE_OPROJ ? AE_OPROJ_TILE_N
                                     : 128;
}

template <typename Config, typename Globals, typename Epi,
          auto A_Ptr, auto B_Ptr, auto Out_Ptr,
          int K_ITERS, int Opcode, auto Aux_Ptr = A_Ptr,
          int TILE_N = ae_default_tile_n(Opcode)>
struct ae_gemm {
    static constexpr int opcode = Opcode;
    static constexpr int barrier_row = BARRIER_ROW(Opcode);

    struct parsed_instruction {
        int opcode, plane, local_row, local_col, dep_layer, dep_row, dep_expected,
            bar_layer, out_row, out_col, norm_sel;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), plane(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), bar_layer(i[7]),
              out_row(i[8]), out_col(i[9]), norm_sel(i[10]) {}
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

    __device__ static inline semaphore &output_arrived(state<Config> &s) {
        return s.semaphores()[pipe::SEM_COUNT];
    }
    __device__ static inline semaphore &gate_arrived(state<Config> &s) {
        return s.semaphores()[pipe::SEM_COUNT + 1];
    }

    using o_fl = st_fl<64, pipe::TILE_N>;
    static_assert(!Epi::store_add || sizeof(o_fl) <= Config::PAGE_SIZE,
                  "The FP32 output tile must fit in one page");
    __device__ static inline o_fl &out_tile_fl(state<Config> &s) {
        return *reinterpret_cast<o_fl *>(&s.pages[s.pid(pipe::OUTPUT_LID)]);
    }

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
                if constexpr (Epi::gate_mul) init_semaphore(gate_arrived(s), 1);
            }
            return pipe::init_semaphores(s) + 1 + (Epi::gate_mul ? 1 : 0);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            if constexpr (Epi::row_scale) {

                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
                const float *src = &g.rms_factor[coord<>{inst.norm_sel, 0}];
                float *dst = row_scale_base(s);
#pragma unroll
                for (int i = laneid(); i < pipe::TILE_M; i += 32) dst[i] = src[i];
            }

            pipe::template loader_loop<dim::ROW>(s, g, inst.plane);

            if constexpr (Epi::gate_mul) {

                if (laneid() == 0) {
                    s.wait_page_ready(pipe::get_output_page(s, 0));
                    tma::expect_bytes(gate_arrived(s), sizeof(typename pipe::o_st));
                    tma::load_async(pipe::get_output_tile(s, 0), g.*Aux_Ptr,
                                    {inst.out_row, inst.out_col}, gate_arrived(s));
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
            rt_fl<16, pipe::TILE_N> acc = pipe::template matmul_loop<2>(s, g);
            if constexpr (Epi::row_scale) {

                kittens::col_vec<rt_fl<16, pipe::TILE_N>> sc;
                warp::load(sc, row_scale_smem(s));
                warp::mul_row(acc, acc, sc);
            }
            if constexpr (Epi::gelu) {
                // gelu_tanh(x) = x * sigmoid(1.5957691216 * (x + 0.044715 x^3))

                warp::apply(acc, acc, [] __device__(int, int, float x) {
                    const float e = __expf(1.5957691216f * (x + 0.044715f * x * x * x));
                    return x * e / (e + 1.0f);
                });
            }
            if constexpr (Epi::gate_mul) {

                ::megakernel::mk_semaphore_wait(gate_arrived(s), 0);
                rt_fl<16, pipe::TILE_N> gate;
                warpgroup::load(gate, pipe::get_output_tile(s, 0));
                if constexpr (Epi::gelu_gate) {

                    warp::apply(gate, gate, [] __device__(int, int, float x) {
                        const float e = __expf(1.5957691216f * (x + 0.044715f * x * x * x));
                        return x * e / (e + 1.0f);
                    });
                }
                warp::mul(acc, acc, gate);
            }
            if constexpr (Epi::store_add) {
                warpgroup::store(out_tile_fl(s), acc);
            } else {
                warpgroup::store(pipe::get_output_tile(s, warpgroup::groupid()), acc);
            }
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
                if constexpr (Epi::store_add) {

                    tma::store_add_async(Out, out_tile_fl(s), {inst.out_row, inst.out_col});
                } else {
#pragma unroll
                for (int i = 0; i < pipe::MATMULS_PER_STAGE; i++) {
                    auto &out_smem = pipe::get_output_tile(s, i);

                    if constexpr (Epi::add_bf16) {

                        tma::store_add_async(Out, out_smem,
                                             {pipe::MATMULS_PER_STAGE * inst.out_row + i, inst.out_col});
                    } else {
                        tma::store_async(Out, out_smem,
                                         {pipe::MATMULS_PER_STAGE * inst.out_row + i, inst.out_col});
                    }
                }
                }
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

}  // namespace pi0
