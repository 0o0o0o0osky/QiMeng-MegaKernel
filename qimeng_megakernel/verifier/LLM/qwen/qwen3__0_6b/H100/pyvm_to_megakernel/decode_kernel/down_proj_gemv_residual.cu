#include "utils.cuh"

using namespace kittens;
using namespace megakernel;
using globals = qwen3_0_6b_globals;

// DownProj split-K: 3 segments of 1024 (intermediate 3072).
static constexpr int SPLIT_K_GRANULARITY = 1024;
static constexpr int NUM_DOWNPROJ_SPLIT_K = 3;

template <typename Config, typename Globals>
struct downproj {
    static constexpr int opcode = OPCODE_DOWN_PROJ_RESIDUAL;
    static constexpr int prev_opcode = OPCODE_RMS_UPGATE_SILU;
    static constexpr int EXPECTED_ARRIVAL_COUNT =
        QWEN3_0_6B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET;

    // Scheduler emits 16-element block indices (one block per tile).
    static constexpr int OUTPUT_BLOCKS_PER_TILE = 1;

    struct parsed_instruction {
        int layer, start_block_idx, end_block_idx, split_K_idx, start_reduction_col, iters;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer = instruction[1];
            int tile_start = instruction[3];
            int tile_end = instruction[4];
            split_K_idx = instruction[5];
            start_block_idx = tile_start * OUTPUT_BLOCKS_PER_TILE;
            end_block_idx = tile_end * OUTPUT_BLOCKS_PER_TILE;
            start_reduction_col = split_K_idx * SPLIT_K_GRANULARITY;
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    template <int START_COL, int RED_DIM>
    struct pipeline_specifics {
        __device__ static inline void load_iter(megakernel::state<Config> &s, const globals &g,
                                                parsed_instruction &inst, int iter, int col_idx,
                                                kittens::st_bf<16, 512> &weight_chunk,
                                                kittens::semaphore &sem) {
            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                weight_chunk, g.down_weights,
                coord<>{inst.layer,
                        (inst.start_block_idx + iter) * Globals::matvec_block_size,
                        inst.start_reduction_col + START_COL + 512 * col_idx},
                sem);
        }
        __device__ static inline void store(megakernel::state<Config> &s, const globals &g,
                                            parsed_instruction &inst, int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;
            constexpr int SCRATCH_BYTES_PER_WARP = 16 * sizeof(float);
            constexpr int SCRATCH_BYTES_PER_STAGE =
                SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;
            uint8_t *output_scratch_start =
                (uint8_t *)s.scratch() + (output_stage * SCRATCH_BYTES_PER_STAGE);
            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);
            kittens::rv_fl<16> output_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          SCRATCH_BYTES_PER_WARP>(output_scratch_start, output_rv);
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

    // Split-K: each instruction reduces 1024 cols (2 pages). Extra stage pages are simply unused.
    using pipeline =
        matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics<0, 1024>, 1024>;

    template <typename PipelineT>
    __device__ static inline void run_segment_consumer(
        const Globals &g, megakernel::state<Config> &s, parsed_instruction &inst, int start_col) {
        using sv_t = kittens::sv_bf<PipelineT::REDUCTION_DIM_PER_WARP>;
        using rv_t = kittens::rv_fl<PipelineT::REDUCTION_DIM_PER_WARP>;
        sv_t &activations_smem =
            reinterpret_cast<sv_t *>(&PipelineT::get_activations(s))[kittens::warpid()];
        kittens::warp::load(activations_smem, g.silu_out,
                            coord<>{inst.start_reduction_col + start_col +
                                    kittens::warpid() * PipelineT::REDUCTION_DIM_PER_WARP});
        kittens::warp::sync();
        rv_t activations_vec;
        kittens::warp::load(activations_vec, activations_smem);
        kittens::warp::sync();
        s.warp_finish_page(PipelineT::get_activation_page(s), 1);
        PipelineT::consumer_loop(s, g, activations_vec);
    }

    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };
    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::loader_loop(s, g);
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
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{
                        inst.layer, BARRIER_ROW(prev_opcode), inst.split_K_idx}],
                    EXPECTED_ARRIVAL_COUNT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
            run_segment_consumer<pipeline>(g, s, inst, 0);
        }
    };
    struct storer {
        static __device__ void run(const globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::storer_loop(s, g);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                kittens::tma::store_async_wait();
                // One arrival per 16-element block (match PyVM: tile_N_end_idx - tile_N_start_idx).
                atomicAdd(&g.Bar[{inst.layer, BARRIER_ROW(opcode), 0}],
                          inst.iters);
            }
        }
    };
};
