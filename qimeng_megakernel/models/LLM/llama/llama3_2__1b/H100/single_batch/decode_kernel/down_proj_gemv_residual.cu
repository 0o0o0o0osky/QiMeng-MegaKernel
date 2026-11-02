#include "utils.cuh"

using namespace kittens;
using namespace megakernel;
using globals = llama_1b_globals;

static constexpr int SPLIT_K_GRANULARITY =
    LLAMA_1B_DOWNPROJ_SPLIT_K_GRANULARITY;

template <typename Config, typename Globals>
struct downproj {
    static constexpr int opcode = OPCODE_DOWN_PROJ_RESIDUAL;
    static constexpr int prev_opcode = OPCODE_RMS_UPGATE_SILU;

    // Each split-K instruction waits only for the UPGATE bucket that produces
    // its input segment. The barrier slot equals split_K_idx.
    static constexpr int EXPECTED_ARRIVAL_COUNT =
        LLAMA_1B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET;
    static_assert(EXPECTED_ARRIVAL_COUNT * LLAMA_1B_UPGATE_SIGNAL_BUCKETS ==
                      llama_1b_globals::intermediate_dim /
                          llama_1b_globals::matvec_block_size,
                  "arrivals must be conserved: buckets * per-bucket == the 512 "
                  "output blocks upgate still emits in total");

    // Python: [opcode, layer_idx, tile_M_idx, tile_N_start_idx,
    //          tile_N_end_idx, split_K_idx, ...]
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
            uint8_t *output_scratch_start = pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);
            kittens::rv_fl<16> output_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(output_scratch_start, output_rv);
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

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>;

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
            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            parsed_instruction inst{s};
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);
                // Wait only on this split-K bucket. inst.split_K_idx is in
                // [0, LLAMA_1B_UPGATE_SIGNAL_BUCKETS) by construction of
                // schedule_downproj, which is the same range upgate signals
                // into -- every slot waited on is a slot that gets filled, and
                // every arrival lands in a slot someone waits on.
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{inst.layer,
                                                BARRIER_ROW(prev_opcode),
                                                inst.split_K_idx}],
                    EXPECTED_ARRIVAL_COUNT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
            sv_t &activations_smem =
                reinterpret_cast<sv_t *>(&pipeline::get_activations(s))[kittens::warpid()];
            kittens::warp::load(activations_smem, g.silu_out,
                                coord<>{inst.start_reduction_col +
                                        kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();
            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();
            s.warp_finish_page(pipeline::get_activation_page(s), 1);
            pipeline::consumer_loop(s, g, activations_vec);
        }
    };
    struct storer {
        static __device__ void run(const globals &g, megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                kittens::tma::store_async_wait();
                // All in-place TMA reductions above have completed before the
                // opcode-6 counter publishes this instruction's output tiles.
                asm volatile("fence.proxy.async.global;" ::: "memory");
                atomicAdd(&g.Bar[{inst.layer, BARRIER_ROW(opcode), 0}],
                          inst.iters);
            }
        }
    };
};
