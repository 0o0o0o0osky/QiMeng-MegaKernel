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
          int K_ITERS_PER_SPLIT, int Opcode, int BarrierRow>
struct gemma_splitk_resid_gemm {
    static constexpr int opcode = Opcode;

    struct parsed_instruction {
        int opcode, layer, local_row, local_col, dep_layer, dep_row, dep_expected, k_split;

        int dep_slot;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), k_split(i[7]),
              dep_slot(i[8]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    struct dep_waiter {
        __device__ static inline void gmem_wait(const Globals &g, state<Config> &s,
                                                const parsed_instruction &inst) {
            if (inst.dep_expected <= 0) return;
            ::megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[kittens::coord<>{inst.dep_layer, inst.dep_row, inst.dep_slot}],
                inst.dep_expected);
        }
    };
    using pipe = matmul_pipeline<Config, Globals, parsed_instruction, dep_waiter,
                                 A_Ptr, B_Ptr, K_ITERS_PER_SPLIT>;

    __device__ static inline semaphore &output_arrived(state<Config> &s) { return s.semaphores()[pipe::SEM_COUNT]; }

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
            const int k0 = inst.k_split * K_ITERS_PER_SPLIT;
            pipe::template loader_loop<dim::ROW>(s, g, inst.layer, k0);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            rt_fl<16, pipe::TILE_N> acc = pipe::template matmul_loop<2>(s, g);
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

                    tma::store_add_async(Out, out_smem,
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
