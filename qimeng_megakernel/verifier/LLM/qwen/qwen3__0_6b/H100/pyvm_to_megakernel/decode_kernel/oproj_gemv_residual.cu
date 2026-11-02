#include "utils.cuh"

using namespace kittens;
using namespace megakernel;
using globals = qwen3_0_6b_globals;

// Reuse Qwen2.5-style oproj: attn_out (q_hidden_dim=2048) @ o_weights -> hidden_states (1024) + residual.
// REDUCTION_DIM must be q_hidden_dim (attn_out size), not hidden_dim (output size).

template <typename Config, typename Globals>
struct o_proj {
    static constexpr int opcode = OPCODE_O_PROJ_RESIDUAL;
    static constexpr int prev_opcode = OPCODE_ATTENTION_REDUCTION;
    static constexpr int EXPECTED_ARRIVAL_COUNT = Globals::num_attention_heads;

    // Scheduler emits (tile_N_start_idx, tile_N_end_idx) in 16-element block indices (one block per tile).
    static constexpr int OUTPUT_BLOCKS_PER_TILE = 1;

    struct parsed_instruction {
        int layer, start_block_idx, end_block_idx, reduction_block_idx,
            start_reduction_col, iters;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer = instruction[1];
            int tile_start = instruction[3];
            int tile_end = instruction[4];
            reduction_block_idx = instruction[5];
            start_reduction_col = reduction_block_idx * Globals::q_hidden_dim;
            start_block_idx = tile_start * OUTPUT_BLOCKS_PER_TILE;
            end_block_idx = tile_end * OUTPUT_BLOCKS_PER_TILE;
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
                weight_chunk, g.o_weights,
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

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics,
                                     Globals::q_hidden_dim>;

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
            auto *attention_ready = (volatile int32_t *)&g.Bar[{
                inst.layer, BARRIER_ROW(prev_opcode),
                inst.reduction_block_idx}];
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);
                megakernel::mk_barrier_wait(attention_ready,
                                            EXPECTED_ARRIVAL_COUNT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
            if (kittens::laneid() == 0 && kittens::warpid() != 0) {
                // Warp 0 established readiness.  Every other warp still needs
                // its own acquire/cache invalidation before reading its slice,
                // but no longer needs to contend in the spin loop.
                while (megakernel::mk_bar_load_acquire(
                           (const int *)attention_ready) <
                       EXPECTED_ARRIVAL_COUNT) {
                }
            }
            kittens::warp::sync();
            sv_t &activations_smem =
                reinterpret_cast<sv_t *>(&pipeline::get_activations(s))[kittens::warpid()];
            kittens::warp::load(activations_smem, g.attn_out,
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
                // One arrival per 16-element block (match PyVM).
                atomicAdd(&g.Bar[{inst.layer, BARRIER_ROW(opcode), 0}],
                          inst.iters);
            }
        }
    };
};
