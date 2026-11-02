#include "utils.cuh"

using namespace kittens;
using namespace megakernel;
using globals = qwen_2_5_3b_globals;

// Same as LLaMA 1B: split_k by 2048, 16 warps, base matvec_pipeline (no 3072).
// Qwen uses 6 segments: 5*2048 + 768 (last segment uses REDUCTION_DIM=1024, zero-pad 768->1024).
//
// Each segment reduces in fp32; the six segment results accumulate into the
// bf16 hidden-state tensor through TMA store-add.
static constexpr int SPLIT_K_GRANULARITY =
    QWEN_3B_DOWNPROJ_SPLIT_K_GRANULARITY;
static constexpr int NUM_DOWNPROJ_SPLIT_K = QWEN_3B_UPGATE_SIGNAL_BUCKETS;

template <typename Config, typename Globals>
struct downproj {
    static constexpr int opcode = OPCODE_DOWN_PROJ_RESIDUAL;
    static constexpr int prev_opcode = OPCODE_RMS_UPGATE_SILU;

    struct parsed_instruction {
        int layer, start_block_idx, end_block_idx, split_K_idx,
            start_reduction_col, iters;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer = instruction[1];
            start_block_idx = instruction[3];
            end_block_idx = instruction[4];
            split_K_idx = instruction[5];
            start_reduction_col = split_K_idx * SPLIT_K_GRANULARITY;
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        __device__ static inline void load_iter(megakernel::state<Config> &s, const globals &g,
                                                parsed_instruction &inst, int iter, int col_idx,
                                                kittens::st_bf<16, 512> &weight_chunk,
                                                kittens::semaphore &sem) {
            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                weight_chunk, g.down_weights,
                coord<>{inst.layer,
                        (inst.start_block_idx + iter) * Globals::matvec_block_size,
                        inst.start_reduction_col + 512 * col_idx},
                sem);
        }
        __device__ static inline void store(megakernel::state<Config> &s, const globals &g,
                                            parsed_instruction &inst, int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;
            uint8_t *output_scratch_start = pipeline_2048::get_output_start(s, output_stage);
            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);
            kittens::rv_fl<16> output_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline_2048::SCRATCH_BYTES_PER_WARP>(output_scratch_start, output_rv);
            kittens::warp::sync();
            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();
            if (kittens::warp::laneid() == 0) {
                kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                    g.hidden_states, output_smem_bf, {block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline_2048 = matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics, 2048>;
    using pipeline_1024 = matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics, 1024>;

    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &instruction, int &query) {
            parsed_instruction inst{instruction};
            if (inst.split_K_idx < NUM_DOWNPROJ_SPLIT_K - 1)
                return pipeline_2048::release_lid(g, instruction, query);
            return pipeline_1024::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.split_K_idx < NUM_DOWNPROJ_SPLIT_K - 1)
                return pipeline_2048::init_semaphores(s);
            return pipeline_1024::init_semaphores(s);
        }
    };
    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.split_K_idx < NUM_DOWNPROJ_SPLIT_K - 1)
                pipeline_2048::loader_loop(s, g);
            else
                pipeline_1024::loader_loop(s, g);
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, megakernel::state<Config> &s) {
            if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            }
        }
    };
    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.split_K_idx < NUM_DOWNPROJ_SPLIT_K - 1) {
                using sv_t = kittens::sv_bf<pipeline_2048::REDUCTION_DIM_PER_WARP>;
                using rv_t = kittens::rv_fl<pipeline_2048::REDUCTION_DIM_PER_WARP>;
                if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                    int activation_page = pipeline_2048::get_activation_page(s);
                    s.wait_page_ready(activation_page);
                    megakernel::mk_barrier_wait(
                        (volatile int32_t *)&g.Bar[{inst.layer,
                                                    BARRIER_ROW(prev_opcode),
                                                    inst.split_K_idx}],
                        QWEN_3B_UPGATE_SIGNAL_BUCKET_EXPECTED(
                            inst.split_K_idx));
                }
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
                sv_t &activations_smem =
                    reinterpret_cast<sv_t *>(&pipeline_2048::get_activations(s))[kittens::warpid()];
                kittens::warp::load(activations_smem, g.silu_out,
                                    coord<>{inst.start_reduction_col +
                                            kittens::warpid() * pipeline_2048::REDUCTION_DIM_PER_WARP});
                kittens::warp::sync();
                rv_t activations_vec;
                kittens::warp::load(activations_vec, activations_smem);
                kittens::warp::sync();
                s.warp_finish_page(pipeline_2048::get_activation_page(s), 1);
                pipeline_2048::consumer_loop(s, g, activations_vec);
            } else {
                using sv_t = kittens::sv_bf<pipeline_1024::REDUCTION_DIM_PER_WARP>;
                using rv_t = kittens::rv_fl<pipeline_1024::REDUCTION_DIM_PER_WARP>;
                if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                    int activation_page = pipeline_1024::get_activation_page(s);
                    s.wait_page_ready(activation_page);
                    megakernel::mk_barrier_wait(
                        (volatile int32_t *)&g.Bar[{inst.layer,
                                                    BARRIER_ROW(prev_opcode),
                                                    inst.split_K_idx}],
                        QWEN_3B_UPGATE_SIGNAL_BUCKET_EXPECTED(
                            inst.split_K_idx));
                }
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
                sv_t &activations_smem =
                    reinterpret_cast<sv_t *>(&pipeline_1024::get_activations(s))[kittens::warpid()];
                // Last segment: 768 elements (10240..11008); warps 0..11 full, warps 12..15 zero
                if (inst.start_reduction_col + (kittens::warpid() + 1) * pipeline_1024::REDUCTION_DIM_PER_WARP <= Globals::intermediate_dim) {
                    kittens::warp::load(activations_smem, g.silu_out,
                                        coord<>{inst.start_reduction_col +
                                                kittens::warpid() * pipeline_1024::REDUCTION_DIM_PER_WARP});
                } else {
                    for (int i = 0; i < pipeline_1024::REDUCTION_DIM_PER_WARP; i++)
                        activations_smem[i] = kittens::bf16(0.f);
                }
                kittens::warp::sync();
                rv_t activations_vec;
                kittens::warp::load(activations_vec, activations_smem);
                kittens::warp::sync();
                s.warp_finish_page(pipeline_1024::get_activation_page(s), 1);
                pipeline_1024::consumer_loop(s, g, activations_vec);
            }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.split_K_idx < NUM_DOWNPROJ_SPLIT_K - 1)
                pipeline_2048::storer_loop(s, g);
            else
                pipeline_1024::storer_loop(s, g);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                kittens::tma::store_async_wait();
                asm volatile("fence.proxy.async.global;" ::: "memory");
                atomicAdd(&g.Bar[{inst.layer, BARRIER_ROW(opcode), 0}],
                          inst.iters);
            }
        }
    };
};
