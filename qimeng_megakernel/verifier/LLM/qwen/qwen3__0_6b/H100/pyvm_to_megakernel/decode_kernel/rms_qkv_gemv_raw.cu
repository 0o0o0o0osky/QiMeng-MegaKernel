#include "instruction_list.cuh"
#include "utils.cuh"
using namespace kittens;
using namespace megakernel;

using globals = qwen3_0_6b_globals;

template <typename Config, typename Globals> struct rms_qkv_raw {
    static constexpr int opcode = OPCODE_RMS_QKV_GEMV;
    // Serialized start/end words are half-open indices in this native block unit.
    static constexpr int OUTPUT_BLOCK_SIZE = Globals::matvec_block_size;
    static_assert(OUTPUT_BLOCK_SIZE == 16, "RMS/QKV output blocks must contain 16 outputs");

    static constexpr int Q_DIM  = Globals::num_attention_heads * Globals::head_dim;
    static constexpr int KV_DIM = Globals::head_dim * Globals::num_kv_heads;
    static constexpr int BLOCKS_PER_HEAD = Globals::head_dim / Globals::matvec_block_size;
    static constexpr int K_BLK_START = Q_DIM / Globals::matvec_block_size;
    static constexpr int V_BLK_START = K_BLK_START + KV_DIM / Globals::matvec_block_size;
    static constexpr int EXPECTED_ARRIVAL_COUNT = QWEN3_0_6B_DOWNPROJ_BARRIER_COUNT;

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            (void)instruction[2];
            // Python serializes block indices directly; no head/tile conversion here.
            start_block_idx = instruction[3];
            end_block_idx = instruction[4];
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {

        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.layer_idx > 0) {
                megakernel::mk_barrier_wait(
                    (volatile int32_t*)&g.Bar[{
                        inst.layer_idx - 1,
                        BARRIER_ROW(OPCODE_DOWN_PROJ_RESIDUAL), 0}],
                    EXPECTED_ARRIVAL_COUNT);
            }
        }

        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const globals &g, parsed_instruction &inst,
                  int iter, int col_idx, kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            auto block_idx = inst.start_block_idx + iter;
            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                weight_chunk, g.qkv_weights,
                {inst.layer_idx, block_idx, col_idx}, sem);
        }

        static __device__ inline void store(megakernel::state<Config> &s, const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);

            kittens::sv_bf<16> &qkv_proj_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            kittens::rv_fl<16> qkv_proj;

            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, qkv_proj);

            int n_heads = Globals::num_attention_heads;
            int n_kv = Globals::num_kv_heads;
            [[maybe_unused]] int bar_slot = (block_idx < K_BLK_START)
                ? (block_idx / BLOCKS_PER_HEAD)
                : (block_idx < V_BLK_START)
                    ? (n_heads + (block_idx - K_BLK_START) / BLOCKS_PER_HEAD)
                    : (n_heads + n_kv + (block_idx - V_BLK_START) / BLOCKS_PER_HEAD);

            kittens::warp::sync();
            kittens::warp::store(qkv_proj_smem_bf, qkv_proj);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                kittens::tma::store_async<cache_policy::EVICT_LAST>(
                    g.qkv_raw, qkv_proj_smem_bf, {0, 0, 0, block_idx});
                kittens::tma::store_async_wait();
                atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode), bar_slot}], 1);
            }
            kittens::warp::sync();
        }
    };

    using pipeline =
        rms_matvec_pipeline<Config, Globals, parsed_instruction,
                            pipeline_specifics, &Globals::hidden_states,
                            &Globals::attn_norm_weights>;

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            pipeline::init_semaphores(s);
            return pipeline::SEM_COUNT;
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
            pipeline::storer_loop(s, g);
        }
    };
};
