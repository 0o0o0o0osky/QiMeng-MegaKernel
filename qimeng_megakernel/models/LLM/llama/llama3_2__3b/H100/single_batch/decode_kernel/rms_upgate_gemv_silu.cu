#include "instruction_list.cuh"
#include "utils.cuh"

using namespace kittens;
using namespace megakernel;

using globals = llama_3b_globals;

template <typename Config, typename Globals> struct rms_upgate_silu {
    static constexpr int opcode = OPCODE_RMS_UPGATE_SILU;
    static constexpr int prev_opcode = OPCODE_O_PROJ_RESIDUAL;
    static constexpr int EXPECTED_ARRIVAL_COUNT =
        Globals::hidden_dim / Globals::matvec_block_size;

    // [opcode, layer_idx, tile_M_idx, len(tile_N_block_idxs), block0, block1, ...]
    struct parsed_instruction {
        int layer_idx, num_blocks, iters;
        int *block_idxs;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            num_blocks = instruction[3];
            iters = 2 * num_blocks;
            block_idxs = instruction + 4;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{inst.layer_idx,
                                             BARRIER_ROW(prev_opcode), 0}],
                EXPECTED_ARRIVAL_COUNT);
        }

        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const globals &g, parsed_instruction &inst,
                  int iter, int col_idx, kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            auto block_idx = inst.block_idxs[iter / 2];
            if (iter % 2 == 0) {
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.up_weights,
                    {inst.layer_idx, block_idx, col_idx}, sem);
            } else {
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.gate_weights,
                    {inst.layer_idx, block_idx, col_idx}, sem);
            }
        }

        static __device__ inline void store(megakernel::state<Config> &s, const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            if (output_idx % 2 == 0) {
                return;
            }

            auto true_output_idx = output_idx / 2;

            auto prev_output_idx = (output_idx - 1);
            auto prev_output_stage = prev_output_idx % 2;

            int block_idx = inst.block_idxs[true_output_idx];

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            uint8_t *prev_output_scratch_start =
                pipeline::get_output_start(s, prev_output_stage);

            kittens::sv_bf<16> &out_smem =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            kittens::rv_fl<16> up_out, gate_out, gate_scratch;

            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                prev_output_scratch_start, up_out);
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, gate_out);

            // neg
            kittens::warp::mul(gate_scratch, gate_out, -1.f);
            kittens::warp::exp(gate_scratch, gate_scratch);
            kittens::warp::add(gate_scratch, gate_scratch, 1.f);
            kittens::warp::div(gate_out, gate_out, gate_scratch);

            kittens::warp::mul(gate_out, up_out, gate_out);

            kittens::warp::sync();

            kittens::warp::store(out_smem, gate_out);

            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                kittens::tma::store_async<cache_policy::EVICT_LAST>(g.silu_out, out_smem,
                                                           {block_idx});
                kittens::tma::store_async_wait();

                // TMA completion makes the async-proxy write complete; bridge
                // it before the downstream generic warp::load observes the
                // matching bucket counter.
                asm volatile("fence.proxy.async.global;" ::: "memory");

                parsed_instruction inst{s};
                atomicAdd(
                    &g.Bar[{inst.layer_idx, BARRIER_ROW(opcode),
                            block_idx /
                                LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET}],
                    1);
            }

            kittens::warp::sync();
        }
    };

    using pipeline =
        rms_matvec_pipeline<Config, Globals, parsed_instruction,
                            pipeline_specifics, &Globals::hidden_states,
                            &Globals::mlp_norm_weights>;
    static_assert(pipeline::OUTPUT_PIPELINE_STAGES == 2);

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer_idx);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::consumer_loop(s, g);
        }
    };

    struct storer {

        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::storer_loop<2>(s, g);
        }
    };
};
