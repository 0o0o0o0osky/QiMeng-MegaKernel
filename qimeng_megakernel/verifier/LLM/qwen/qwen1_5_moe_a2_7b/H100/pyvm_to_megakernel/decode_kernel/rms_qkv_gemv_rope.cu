#pragma once
#include "instruction_list.cuh"
#include "utils.cuh"

// ============================================================================
// opcode 1 — RMS_QKVProj_GEMV_RoPE
//
// 132 instructions per layer.  Emitted first in every layer.
//
// Static instruction layout (7 words)
//   [0] opcode = 1
//   [1] layer_idx            0..23
//   [2] tile_M_idx           == 0
//   [3] tile_N_start_idx     0..383   (output block, 16 rows each)
//   [4] tile_N_end_idx       exclusive
//   [5] tile_M_granularity   == 1
//   [6] tile_N_granularity   == 16
//   iters = tile_N_end_idx - tile_N_start_idx
//
// reads   hidden_states, attn_norm_weights[L], qkv_weights[L], qkv_bias[L],
//         rope_cos[pos_id], rope_sin[pos_id]
// writes  block 0..127   -> q_post_rope       (post-RoPE)
//         block 128..255 -> k_cache[L, pos_id, head, :]   (post-RoPE)
//         block 256..383 -> v_cache[L, pos_id, head, :]
//         Q/K output dimensions are interleaved, so RoPE is a plain
//         adjacent-pair rotation against the interleaved cos/sin tables.
//
// barrier WAIT   (only when layer_idx > 0)
//         Bar[layer_idx - 1, BARRIER_ROW(OPCODE_MOE_COMBINE_COMMIT), 0] >= 128
// barrier SIGNAL (storer, after tma::store_async_wait())
//         Bar[layer_idx, BARRIER_ROW(OPCODE_RMS_QKVPROJ_GEMV_ROPE),
//             block_idx / 8] += 1        // slot 0..47: Q 0-15, K 16-31, V 32-47
//
// pipeline
//   rms_matvec_pipeline<..., &Globals::hidden_states, &Globals::attn_norm_weights>
//   (K = Globals::hidden_dim = 2048 -> 4 pages, 16/16 warps)
//
// Bias and RoPE use plain global vector loads.
//   `Config::SCRATCH_BYTES` is 4096 and matvec_pipeline already owns
//   [0, 3072) for the three output stages plus [3072, 3136) for the RMSNorm
//   cross-warp partials. The remaining 960 B cannot hold two `sv_fl<128>`
//   rows. `warp::load(rv, gl, coord)` needs no TMA descriptor, semaphore, or
//   scratch allocation.
// ============================================================================

using namespace kittens;
using namespace megakernel;

template <typename Config, typename Globals> struct rms_qkvproj_gemv_rope {
    static constexpr int opcode = OPCODE_RMS_QKVPROJ_GEMV_ROPE;
    static constexpr int prev_opcode = OPCODE_MOE_COMBINE_COMMIT;
    // The next layer starts after all 128 combine tiles commit.
    static constexpr int EXPECTED_ARRIVAL_COUNT = 128;

    static constexpr int BLK = Globals::matvec_block_size;            // 16
    static constexpr int BLOCKS_PER_HEAD = Globals::head_dim / BLK;   // 8
    //: first K block == 2048/16, first V block == 4096/16
    static constexpr int K_BLK_START =
        Globals::num_attention_heads * BLOCKS_PER_HEAD;               // 128
    static constexpr int V_BLK_START =
        K_BLK_START + Globals::num_kv_heads * BLOCKS_PER_HEAD;        // 256
    static constexpr int TOTAL_BLKS =
        V_BLK_START + Globals::num_kv_heads * BLOCKS_PER_HEAD;        // 384

    static_assert(TOTAL_BLKS * BLK == Globals::qkv_dim,
                  "Q||K||V block partition must cover qkv_dim exactly");

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            // instruction[2] == tile_M_idx, always 0
            start_block_idx = instruction[3];
            end_block_idx = instruction[4];
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        // Called by rms_matvec_pipeline::consumer_loop from warp 0 / lane 0,
        // before hidden_states is read.
        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (inst.layer_idx > 0) {
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{inst.layer_idx - 1,
                                                BARRIER_ROW(prev_opcode), 0}],
                    EXPECTED_ARRIVAL_COUNT);
            }
        }

        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            kittens::tma::load_async<kittens::dim::ROW,
                                     kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.qkv_weights,
                kittens::coord<>{inst.layer_idx,
                                 (inst.start_block_idx + iter) * BLK,
                                 512 * col_idx},
                sem);
        }

        static __device__ inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            const int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<BLK> &out_smem_bf =
                *reinterpret_cast<kittens::sv_bf<BLK> *>(output_scratch_start);

            kittens::rv_fl<BLK> acc, aux;
            matvec_reduce<Config, kittens::sv_fl<BLK>, kittens::rv_fl<BLK>,
                          pipeline::SCRATCH_BYTES_PER_WARP,
                          pipeline::ACTIVE_CONSUMER_WARPS>(
                output_scratch_start, acc);

            // ---- qkv_bias --------------------------------------------------
            // qkv_bias is 2-D [24, 6144]; a vec coord multiplies only its last
            // component by the vector length, so {0,0,L,block} == element
            // L*6144 + block*16.
            kittens::warp::load(aux, g.qkv_bias,
                                {0, 0, inst.layer_idx, block_idx});
            kittens::warp::add(acc, acc, aux);

            // ---- interleaved RoPE (Q and K only) ---------------------------
            if (block_idx < V_BLK_START) {
                kittens::rv_fl<BLK> rope_cos, rope_sin;
                const int rope_blk = block_idx % BLOCKS_PER_HEAD;
                kittens::warp::load(rope_cos, g.rope_cos,
                                    {0, 0, (int)g.pos_id, rope_blk});
                kittens::warp::load(rope_sin, g.rope_sin,
                                    {0, 0, (int)g.pos_id, rope_blk});
                kittens::warp::sync();
                // rv_fl<16> is `naive` layout: element i lives in lane i.
                // Pairs are (2k, 2k+1): out[2k]   = x[2k]  *cos - x[2k+1]*sin
                //                       out[2k+1] = x[2k+1]*cos + x[2k]  *sin
                const int lane = kittens::laneid();
                const int mod = (lane & 1) ? -1 : 1;
                float pair_val = __shfl_sync(MASK_ALL, acc[0][0], lane + mod);
                if (lane < BLK) {
                    acc[0][0] = acc[0][0] * rope_cos[0][0] +
                                float(-mod) * pair_val * rope_sin[0][0];
                }
            }

            kittens::warp::sync();
            kittens::warp::store(out_smem_bf, acc);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                if (block_idx < K_BLK_START) {
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.q_post_rope, out_smem_bf, {0, 0, 0, block_idx});
                } else if (block_idx < V_BLK_START) {
                    const int b = block_idx - K_BLK_START;
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.k_cache, out_smem_bf,
                        {inst.layer_idx, (int)g.pos_id, b / BLOCKS_PER_HEAD,
                         b % BLOCKS_PER_HEAD});
                } else {
                    const int b = block_idx - V_BLK_START;
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.v_cache, out_smem_bf,
                        {inst.layer_idx, (int)g.pos_id, b / BLOCKS_PER_HEAD,
                         b % BLOCKS_PER_HEAD});
                }
                // Only the *read* of the smem source has to complete before the
                // next output stage reuses this scratch; the global write is
                // drained once, in the storer epilogue, before the barrier.
                kittens::tma::store_async_read_wait();
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
            return pipeline::init_semaphores(s);
        }
    };
    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer_idx);
        }
    };
    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };
    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::consumer_loop(s, g);
        }
    };
    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
            // warp::sync() BEFORE the drain + signal: Volta+ ITS does not
            // reconverge automatically, so lane 0 must not run ahead of the
            // other lanes' work in storer_loop.
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                kittens::tma::store_async_wait();
                // Device-scope release fence: q_post_rope / kv_cache must be
                // visible to the attention CTAs before their counter moves.
                __threadfence();
                for (int b = inst.start_block_idx; b < inst.end_block_idx; ++b) {
                    atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode),
                                      b / BLOCKS_PER_HEAD}],
                              1);
                }
            }
        }
    };
};
