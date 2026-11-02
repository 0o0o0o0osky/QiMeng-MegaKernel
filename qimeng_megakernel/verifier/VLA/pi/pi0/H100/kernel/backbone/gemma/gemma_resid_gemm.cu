#pragma once

#include "kittens.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"
#include "include/pipelines/matmul_pipeline.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

template <typename Config, typename Globals,
          auto A_Ptr, auto B_Ptr, auto Out_Ptr,
          int K_ITERS, int Opcode, int BarrierRow, int STATS_ROW0,
          auto Shadow_Ptr = Out_Ptr, bool WRITE_SHADOW = false>
struct gemma_resid_stats_gemm {
    static constexpr int opcode = Opcode;

    struct parsed_instruction {
        int opcode, layer, local_row, local_col, dep_layer, dep_row, dep_expected, norm_sel;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), norm_sel(i[7]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    struct dep_waiter {
        __device__ static inline void gmem_wait(const Globals &g, state<Config> &s,
                                                const parsed_instruction &inst) {
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
        }
    };
    using pipe = matmul_pipeline<Config, Globals, parsed_instruction, dep_waiter, A_Ptr, B_Ptr, K_ITERS>;

    __device__ static inline semaphore &output_arrived(state<Config> &s) { return s.semaphores()[pipe::SEM_COUNT]; }
    __device__ static inline semaphore &resid_arrived(state<Config> &s)  { return s.semaphores()[pipe::SEM_COUNT + 1]; }

    __device__ static inline sv_fl<16> &stats_smem(state<Config> &s) {
        return *reinterpret_cast<sv_fl<16> *>(
            reinterpret_cast<float *>(s.scratch()) + kittens::warpid() * 16);
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &inst, int &query) {
            return pipe::template release_lid<2>(g, inst, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if (laneid() == 0) {
                init_semaphore(output_arrived(s), Config::NUM_CONSUMER_WARPS);
                init_semaphore(resid_arrived(s), 1);
            }
            return pipe::init_semaphores(s) + 2;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            pipe::template loader_loop<dim::ROW>(s, g, inst.layer);

            if (laneid() == 0) {
                s.wait_page_ready(pipe::get_output_page(s, 0));
                tma::expect_bytes(resid_arrived(s),
                                  pipe::MATMULS_PER_STAGE * sizeof(typename pipe::o_st));
#pragma unroll
                for (int i = 0; i < pipe::MATMULS_PER_STAGE; i++)
                    tma::load_async(pipe::get_output_tile(s, i), g.*Out_Ptr,
                                    {pipe::MATMULS_PER_STAGE * inst.local_row + i, inst.local_col},
                                    resid_arrived(s));
            }
            warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            rt_fl<16, pipe::TILE_N> acc = pipe::template matmul_loop<2>(s, g);

            ::megakernel::mk_semaphore_wait(resid_arrived(s), 0);
            rt_fl<16, pipe::TILE_N> r;
            warpgroup::load(r, pipe::get_output_tile(s, warpgroup::groupid()));
            warp::add(acc, acc, r);

            warp::mul(r, acc, acc);
            col_vec<rt_fl<16, pipe::TILE_N>> ss;
            warp::zero(ss);
            warp::row_sum(ss, r, ss);
            warp::store(stats_smem(s), ss);
            warp::sync();
            if (laneid() < 16) {
                const int row = inst.local_row * pipe::TILE_M + kittens::warpid() * 16 + laneid();
                g.rms_factor[coord<>{STATS_ROW0 + inst.local_col, row}] = stats_smem(s)[laneid()];
            }

            warpgroup::store(pipe::get_output_tile(s, warpgroup::groupid()), acc);
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
#pragma unroll
                for (int i = 0; i < pipe::MATMULS_PER_STAGE; i++) {
                    auto &out_smem = pipe::get_output_tile(s, i);

                    tma::store_async(Out, out_smem,
                                     {pipe::MATMULS_PER_STAGE * inst.local_row + i, inst.local_col});
                    if constexpr (WRITE_SHADOW)
                        tma::store_async(g.*Shadow_Ptr, out_smem,
                                         {pipe::MATMULS_PER_STAGE * inst.local_row + i, inst.local_col});
                }
                tma::store_async_read_wait();

                s.finish_page(pipe::get_output_page(s, 0), Config::NUM_CONSUMER_WARPS);
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

}  // namespace pi0
