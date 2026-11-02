#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"

// opcode 12 — RoutedExpert_UpGate_SiLU
// 132 instructions per layer = 4 top-k slots x split(88 tiles, 33).
// 1408 / 16 = 88 output blocks per expert.
// Static instruction layout (7 words)
//   [0] opcode = 12
//   [1] layer_idx           0..23
//   [2] expert_idx          top-k slot 0..3; resolved to a physical id on device
//   [3] tile_N_start_idx    0..87
//   [4] tile_N_end_idx      exclusive
//   [5] tile_M_granularity  == 1
//   [6] tile_N_granularity  == 16
// device-side expert dereference in `loader_gmem_wait`:
//   physical_expert = expert_indices[expert_idx]           // int64
//   row             = layer_idx * 60 + physical_expert     // 0..1439
//   weights         = expert_gate_up_proj_weights_flat[row]  // [2816, 2048]
//                     rows 0..1407 = gate, rows 1408..2815 = up
// reads   normed_hidden_states (K = 2048), expert_gate_up_proj_weights_flat
// writes  routed_silu_buffer[expert_idx, tile] = silu(gate) * up
// barrier WAIT   Bar[L, BARRIER_ROW(OPCODE_ROUTER_SOFTMAX_TOPK), 0] >= 1
// barrier SIGNAL Bar[L, BARRIER_ROW(OPCODE_ROUTED_EXPERT_UPGATE_SILU), 0] += 1
//                per tile block  (352 per layer = 4 x 88)
// IMPLEMENTATION NOTES
// * K = Globals::hidden_dim (2048): STAGE_PAGES = 4, ACTIVE_CONSUMER_WARPS =
//   16, TOTAL_PAGES = 13, SEM_COUNT = 12. The reduction dimension is 2048;
//   1408 is the output extent.
// * gate and up are two 16-row tiles of the SAME flat weight row, so one
//   output block costs two pipeline iterations:  `iters = 2 * num_blocks`,
//   even iter = gate (row  block*16), odd iter = up (row 1408 + block*16).
//   `storer_loop<2>` retires the two output stages together; `store()` folds
//   silu(gate) * up on the odd iteration only.  (Same shape as the llama
//   rms_upgate_silu op; `iters` must stay the pipeline iteration count or
//   `release_lid` mis-predicts the page release order.)
// * `loader_gmem_wait` is the whole point of this op: the TMA source address
//   depends on `expert_indices`, which only exists after op9 lands.  The hook
//   runs on loader lane 0 BEFORE the first `wait_page_ready`, spins with
//   `megakernel::mk_barrier_wait` (never a bare spin — this is the most likely
//   deadlock site in the kernel), then writes the physical id back into the
//   SAME `parsed_instruction` object that `load_iter` later reads.
//   `expert_id` remains -1 until this wait resolves and publishes the selected
//   physical expert id to the loader.
// * matvec_reduce uses `pipeline::ACTIVE_CONSUMER_WARPS`, the common reduction
//   geometry shared with the K=1408 routed down-projection.

template <typename Config, typename Globals> struct routed_expert_upgate_silu {
    static constexpr int opcode = OPCODE_ROUTED_EXPERT_UPGATE_SILU;

    // Top-k publishes one selected-expert vector per layer.
    static constexpr int WAIT_ROW = BARRIER_ROW(OPCODE_ROUTER_SOFTMAX_TOPK);
    static constexpr int EXPECTED_ARRIVAL_COUNT = 1;
    static constexpr int SIGNAL_ROW =
        BARRIER_ROW(OPCODE_ROUTED_EXPERT_UPGATE_SILU);

    //: row offset of the `up` half inside one flat gate_up weight row.
    static constexpr int UP_ROW_OFFSET = Globals::moe_intermediate_dim; // 1408
    static constexpr int COLS_PER_PAGE = 512;

    struct parsed_instruction {
        int layer_idx, expert_slot, start_block_idx, end_block_idx;
        int num_blocks, iters;
        //: physical expert id — INVALID until `loader_gmem_wait` fills it in.
        int expert_id;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            expert_slot = instruction[2];
            start_block_idx = instruction[3];
            end_block_idx = instruction[4];
            num_blocks = end_block_idx - start_block_idx;
            iters = 2 * num_blocks; // gate pass + up pass
            expert_id = -1;         // poison; see header note
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        // ---- the routed-expert hook (utils.cuh `loader_loop`) --------------
        __device__ static inline void
        loader_gmem_wait(megakernel::state<Config> &s, const Globals &g,
                         parsed_instruction &inst) {
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[kittens::coord<>{inst.layer_idx,
                                                            WAIT_ROW, 0}],
                EXPECTED_ARRIVAL_COUNT);

            inst.expert_id =
                (int)g.expert_indices[kittens::coord<>{inst.expert_slot}];
        }

        // ST names the 2048-column stage-page layout.
        template <typename ST>
        __device__ static inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  ST &weight_chunk, kittens::semaphore &sem) {
            const int block_idx = inst.start_block_idx + (iter >> 1);
            // even iter -> gate half, odd iter -> up half
            const int row = ((iter & 1) ? UP_ROW_OFFSET : 0) +
                            block_idx * Globals::matvec_block_size;
            kittens::tma::load_async<kittens::dim::ROW,
                                     kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.expert_gate_up_proj_weights_flat,
                kittens::coord<>{inst.layer_idx * Globals::num_experts +
                                     inst.expert_id,
                                 row, COLS_PER_PAGE * col_idx},
                sem);
        }

        __device__ static inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            if ((output_idx & 1) == 0) {
                return; // gate half: its partials are folded on the next iter
            }

            const int block_idx = inst.start_block_idx + (output_idx >> 1);
            // OUTPUT_PIPELINE_STAGES == 3 (static_assert below).
            const int gate_stage = (output_idx - 1) % 3;

            uint8_t *up_scratch = pipeline::get_output_start(s, output_stage);
            uint8_t *gate_scratch_start = pipeline::get_output_start(s, gate_stage);

            kittens::sv_bf<Globals::matvec_block_size> &out_smem =
                *reinterpret_cast<kittens::sv_bf<Globals::matvec_block_size> *>(
                    up_scratch);

            kittens::rv_fl<Globals::matvec_block_size> up_out, gate_out, tmp;
            matvec_reduce<Config, kittens::sv_fl<Globals::matvec_block_size>,
                          kittens::rv_fl<Globals::matvec_block_size>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(gate_scratch_start,
                                                           gate_out);
            matvec_reduce<Config, kittens::sv_fl<Globals::matvec_block_size>,
                          kittens::rv_fl<Globals::matvec_block_size>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(up_scratch, up_out);

            // silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
            kittens::warp::mul(tmp, gate_out, -1.f);
            kittens::warp::exp(tmp, tmp);
            kittens::warp::add(tmp, tmp, 1.f);
            kittens::warp::div(gate_out, gate_out, tmp);
            kittens::warp::mul(gate_out, gate_out, up_out);

            // out_smem aliases `up_scratch`, which up_out was just read from.
            kittens::warp::sync();
            kittens::warp::store(out_smem, gate_out);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                    g.routed_silu_buffer, out_smem,
                    kittens::coord<>{0, 0, inst.expert_slot,
                                     block_idx * Globals::matvec_block_size});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics, Globals::hidden_dim>;
    static_assert(pipeline::OUTPUT_PIPELINE_STAGES == 3,
                  "store() hard-codes the 3-stage output ring");
    static_assert(pipeline::ACTIVE_CONSUMER_WARPS == Config::NUM_CONSUMER_WARPS,
                  "K = hidden_dim must light up every consumer warp");

    struct controller {
        static __device__ int
        release_lid(const Globals &g, typename Config::instruction_t &instruction,
                    int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::loader_loop(s, g);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            parsed_instruction inst{s};

            const int activation_page = pipeline::get_activation_page(s);
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                s.wait_page_ready(activation_page);
                // normed_hidden_states is published by op8 (row 7 slot 1); op9
                // waits for all 60 op8s, so row 8 >= 1 transitively covers it.
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[kittens::coord<>{inst.layer_idx,
                                                                WAIT_ROW, 0}],
                    EXPECTED_ARRIVAL_COUNT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            rv_t activations_vec;
            kittens::warp::zero(activations_vec);
            if (kittens::warpid() < pipeline::ACTIVE_CONSUMER_WARPS) {
                sv_t &activations_smem = reinterpret_cast<sv_t *>(
                    &pipeline::get_activations(s))[kittens::warpid()];
                kittens::warp::load(
                    activations_smem, g.normed_hidden_states,
                    kittens::coord<>{kittens::warpid() *
                                     pipeline::REDUCTION_DIM_PER_WARP});
                kittens::warp::sync();
                kittens::warp::load(activations_vec, activations_smem);
            }
            // finish_page strictly after warp::sync(), and by ALL warps.
            kittens::warp::sync();
            s.warp_finish_page(activation_page, 1);

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::storer_loop<2>(s, g);
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                kittens::tma::store_async_wait();
                // Device-scope release fence before signalling the cross-CTA
                // barrier: op13 must not observe the count without the data.
                __threadfence();
                atomicAdd(&g.Bar[kittens::coord<>{inst.layer_idx, SIGNAL_ROW, 0}],
                          inst.num_blocks);
            }
        }
    };
};
