#pragma once

#include "kittens.cuh"
#include "../util.cuh"
#include "../../common/pi0_config.cuh"

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

struct no_gmem_wait {
    template <typename Globals, typename Config, typename parsed_instruction>
    __device__ static inline void gmem_wait(const Globals &, state<Config> &, const parsed_instruction &) {}
};

template <typename Config, typename Globals, typename parsed_instruction, typename gmem_waiter, auto A_Ptr,
          auto B_Ptr, int _num_iters, int _tile_n = 128>
struct matmul_pipeline {
    static_assert(Config::PAGE_SIZE == 32768, "matmul_pipeline assumes 32KB pages");
    static_assert(Config::NUM_PAGES == 6, "matmul_pipeline's page map is hardcoded for 6 pages");
    static_assert(Config::SCRATCH_BYTES >= 8192);
    static_assert(Config::NUM_CONSUMER_WARPS == 4 || Config::NUM_CONSUMER_WARPS == 8,
                  "matmul_pipeline supports 1 or 2 consumer warpgroups");

    static constexpr int INPUT_PIPELINE_STAGES = PIPELINE_K_STAGES;
    static_assert(INPUT_PIPELINE_STAGES == 4, "release_lid / page map are written for 4 stages");

    static constexpr int MATMULS_PER_STAGE = Config::NUM_CONSUMER_WARPS / 4;

    static constexpr int TILE_M = 64 * MATMULS_PER_STAGE;

    static_assert(_num_iters % INPUT_PIPELINE_STAGES == 0,
                  "num_iters must be a multiple of 4 -- pad K to a multiple of 256 (see pi0_config.cuh k_pad)");
    static_assert(_num_iters >= INPUT_PIPELINE_STAGES, "Invalid number of iters");

    static constexpr int TILE_N = _tile_n;
    static_assert(TILE_N % 32 == 0 && TILE_N <= 256, "TILE_N must be a multiple of 32 and no greater than 256");

    using a_st = st_bf<64, PIPELINE_K_DIM>;              //  8 KB
    using b_st = st_bf<TILE_N, PIPELINE_K_DIM>;
    using o_st = st_bf<64, TILE_N>;

    static constexpr int SEM_COUNT = 2 * INPUT_PIPELINE_STAGES;

    static constexpr bool COMPACT_A = (MATMULS_PER_STAGE == 1);

    static constexpr int A_PAGES          = COMPACT_A ? 1 : 2;
    static constexpr int A_STAGES_PER_PAGE = INPUT_PIPELINE_STAGES / A_PAGES;
    static constexpr int B_LID0           = COMPACT_A ? 1 : 2;
    static constexpr int B_STAGES_PER_PAGE = 2;
    static_assert(A_STAGES_PER_PAGE * MATMULS_PER_STAGE * sizeof(a_st) <= Config::PAGE_SIZE,
                  "The A stages assigned to one page exceed the page capacity");
    static_assert(2 * sizeof(b_st) <= Config::PAGE_SIZE, "Two B stages must fit in one page");

    __device__ static inline int get_a_page(state<Config> &s, int stage) {
        if constexpr (COMPACT_A) return s.pid(0);
        else                     return s.pid(stage / 2);
    }
    __device__ static inline int get_b_page(state<Config> &s, int stage) { return s.pid(B_LID0 + stage / 2); }

    __device__ static inline a_st (&get_a_tile(state<Config> &s, int stage))[MATMULS_PER_STAGE] {
        if constexpr (COMPACT_A)
            return reinterpret_cast<a_st(*)[MATMULS_PER_STAGE]>(&s.pages[s.pid(0)])[stage];
        else
            return reinterpret_cast<a_st(*)[MATMULS_PER_STAGE]>(&s.pages[get_a_page(s, stage)])[stage & 1];
    }
    __device__ static inline b_st &get_b_tile(state<Config> &s, int stage) {
        return reinterpret_cast<b_st *>(&s.pages[get_b_page(s, stage)])[stage & 1];
    }

    static constexpr int OUTPUT_LID = 4;
    __device__ static inline int get_output_page(state<Config> &s, int) { return s.pid(OUTPUT_LID); }
    __device__ static inline o_st &get_output_tile(state<Config> &s, int wg) {
        return reinterpret_cast<o_st *>(&s.pages[s.pid(OUTPUT_LID)])[wg];
    }

    __device__ static inline semaphore &inputs_arrived(state<Config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline semaphore &inputs_finished(state<Config> &s, int stage) {
        return s.semaphores()[INPUT_PIPELINE_STAGES + stage];
    }

    __device__ static inline void loader_input_wait(state<Config> &s, int iter) {
        int input_stage = iter % INPUT_PIPELINE_STAGES;
        ::megakernel::mk_semaphore_wait(
            inputs_finished(s, input_stage),
            (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);
    }

    template <int _pages_in_reserve = 0>
    __device__ static inline int release_lid(const Globals &g, typename Config::instruction_t &instruction,
                                             int &query) {
        static_assert(INPUT_PIPELINE_STAGES == 4, "release_lid table is written for 4 stages");
        static_assert(_num_iters % INPUT_PIPELINE_STAGES == 0, "Invalid number of iters");

        constexpr int order_mps2[6] = {5, 2, 0, 3, 1, 4};
        constexpr int order_mps1[6] = {5, 3, 1, 2, 0, 4};
        constexpr int keep = _pages_in_reserve % 6;
        if constexpr (COMPACT_A) return order_mps1[(query + keep) % 6];
        else                     return order_mps2[(query + keep) % 6];
    }

    __device__ static inline int init_semaphores(state<Config> &s) {
        if (laneid() < INPUT_PIPELINE_STAGES) {
            init_semaphore(inputs_arrived(s, laneid()), 1);
            init_semaphore(inputs_finished(s, laneid()), MATMULS_PER_STAGE);
        }
        return SEM_COUNT;
    }

    template <int _axis = dim::ROW>
    __device__ static inline void loader_loop(state<Config> &s, const Globals &g,
                                              int layer_idx = 0, int k0 = 0) {
        parsed_instruction inst{s};

        if constexpr (COMPACT_A) {
            if (laneid() == 1 || laneid() == 2) {
                int pid = s.pid(laneid() == 1 ? 5 : 3);
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        } else {
            if (laneid() == 1) {
                int pid = s.pid(5);
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }

        if (laneid() == 0) {
            for (int iter = 0; iter < _num_iters; iter++) {
                int input_stage = iter % INPUT_PIPELINE_STAGES;
                loader_input_wait(s, iter);

                auto &sem = inputs_arrived(s, input_stage);
                tma::expect_bytes(sem,
                                  sizeof(bf16) * ((a_st::rows * MATMULS_PER_STAGE) + b_st::rows) * PIPELINE_K_DIM);

                int a_page = get_a_page(s, input_stage), b_page = get_b_page(s, input_stage);
                auto &a_smem = get_a_tile(s, input_stage);   // a_st(&)[MATMULS_PER_STAGE]
                auto &b_smem = get_b_tile(s, input_stage);

                if (iter < INPUT_PIPELINE_STAGES) {
                    s.wait_page_ready(b_page);
                }
                tma::load_async<_axis, cache_policy::NORMAL>(
                    b_smem, g.*B_Ptr,
                    {layer_idx, inst.local_col, k0 + iter}, sem);

                if (iter == 0) {
                    gmem_waiter::gmem_wait(g, s, inst);
                }

                if ((iter < INPUT_PIPELINE_STAGES) && (iter % A_STAGES_PER_PAGE == 0)) {
                    s.wait_page_ready(a_page);
                }
#pragma unroll
                for (int i = 0; i < MATMULS_PER_STAGE; i++) {
                    tma::load_async<_axis, cache_policy::NORMAL>(
                        a_smem[i], g.*A_Ptr,
                        {MATMULS_PER_STAGE * inst.local_row + i, k0 + iter},
                        sem);
                }
            }
        }
    }

    template <int _pages_to_not_release = 0>
    __device__ static __forceinline__ rt_fl<16, TILE_N> matmul_loop(state<Config> &s, const Globals &g) {
        parsed_instruction inst{s};
        int groupid = warpgroup::groupid();

        int input_stage = 0;
        rt_fl<16, TILE_N> out;

        {
            ::megakernel::mk_semaphore_wait(inputs_arrived(s, input_stage), 0);
            auto &a_smem = get_a_tile(s, input_stage);
            auto &b_smem = get_b_tile(s, input_stage);

            warpgroup::mm_ABt(out, a_smem[groupid], b_smem);
            warpgroup::mma_async_wait();
            warpgroup::arrive(inputs_finished(s, input_stage));

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
        }

        for (int i = 1; i < _num_iters; i++) {
            ::megakernel::mk_semaphore_wait(
                inputs_arrived(s, input_stage),
                (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
            auto &a_smem = get_a_tile(s, input_stage);
            auto &b_smem = get_b_tile(s, input_stage);

            warpgroup::mma_ABt(out, a_smem[groupid], b_smem);

            warpgroup::mma_async_wait();
            warpgroup::arrive(inputs_finished(s, input_stage));

            if constexpr (COMPACT_A) {

                if (i == _num_iters - 3) {
                    s.warp_finish_page(s.pid(1), 1);
                } else if (i == _num_iters - 1) {
                    s.warp_finish_page(s.pid(2), 1);   // B stage 2,3
                    s.warp_finish_page(s.pid(0), 1);   // A stage 0..3
                }
            } else {

                if (i == _num_iters - 3) {
                    s.warp_finish_page(s.pid(2), 1);
                    s.warp_finish_page(s.pid(0), 1);
                } else if (i == _num_iters - 1) {
                    s.warp_finish_page(s.pid(3), 1);   // B stage 2,3
                    s.warp_finish_page(s.pid(1), 1);   // A stage 2,3
                }
            }

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
        }

        return out;
    }
};

}  // namespace pi0
