// Multi-batch decode attention reduction kernel.
//
// Handles one DecodeAttentionReduction instruction per CTA (128 threads / 4 warps).
// Each warp handles one Q head (ATTN_REDUCTION_SIZE = 4 heads per instruction).
//
// Two-pass vectorized reduction:
//   Pass 1: Load all partial LSEs for this head; find global max_lse.
//   Pass 2: weight_k = exp2(lse_k - max_lse); accumulate weighted O; normalize.
//   Output: attn_out[batch_idx, head] in bf16; barrier update.
//
// Instruction serialization (DecodeAttentionReduction):
//   words[0] = 3  (opcode)
//   words[1] = layer_idx
//   words[2] = batch_idx
//   words[3] = head_start_idx
//   words[4] = num_partials
//   words[5] = len(reduction_list)
//   words[6..6+len-1] = reduction_list (partial indices to merge)
//   words[6+len] = is_terminal  (1 = write to attn_out, 0 = write back to inter)
//   words[7+len] = output_partial_idx  (for non-terminal)
//   words[8+len] = valid_partials      (-1 = len(reduction_list))
//   words[9+len] = is_active           (0=barrier-only, 1=active, -1=check mask)

#include "globals.cuh"
#include "llama_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

namespace megakernel {

constexpr int kRedHeadDim    = LlamaConfig::HEAD_DIM;    // 64
constexpr int kRedNThreads   = KernelConfig::NUM_THREADS; // 128
constexpr int kRedNWarps     = kRedNThreads / 32;         // 4
constexpr int kRedGQARatio   = LlamaConfig::NUM_Q_HEADS / LlamaConfig::NUM_KV_HEADS;  // 4
// ATTN_REDUCTION_SIZE = 4 heads per reduction instruction (warp-per-head).
constexpr int kRedGroupSize  = 4;
static_assert(kRedNWarps >= kRedGroupSize,
              "attention reduction requires at least four warps");
constexpr int kRedWarpSize   = 32;
constexpr int kRedMaxPartials = 128;
constexpr int kRedElemsPerThread = kRedHeadDim / kRedWarpSize;  // 2

struct AttnReductionSmem {
    alignas(128) float lse_workspace[kRedGroupSize * kRedMaxPartials];
    alignas(128) float data_workspace[2 * kRedGroupSize * kRedHeadDim];
};

__device__ __forceinline__ void decode_attention_reduction(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx
) {
    const int tid     = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    AttnReductionSmem& s = *reinterpret_cast<AttnReductionSmem*>(smem.main_workspace);
#ifdef ENABLE_PROFILING
    uint64_t prof_stage_begin = 0;
#endif

    // ========================================================================
    // 1. Parse instruction (from smem.instruction[] loaded by mk_kernel)
    // ========================================================================
    const int layer_idx      = smem.instruction[1];
    const int batch_idx      = smem.instruction[2];
    const int head_start_idx = smem.instruction[3];
    const int num_partials   = smem.instruction[4];
    const int list_len       = smem.instruction[5];
    // reduction_list at words 6..6+list_len-1
    // is_terminal, output_partial_idx, valid_partials, is_active follow.
    const int is_terminal_word  = 6 + list_len;
    const int out_partial_word  = 7 + list_len;
    const int valid_parts_word  = 8 + list_len;
    const int is_active_word    = 9 + list_len;

    const int is_terminal      = (is_terminal_word  < 32) ? smem.instruction[is_terminal_word]  : 1;
    const int output_part_idx  = (out_partial_word   < 32) ? smem.instruction[out_partial_word]   : 0;
    const int valid_partials   = (valid_parts_word   < 32) ? smem.instruction[valid_parts_word]   : list_len;
    const int is_active        = (is_active_word     < 32) ? smem.instruction[is_active_word]     : -1;

    // Number of partials to actually merge (may be < list_len due to sparse fill).
    const int n_merge_unclamped = (valid_partials >= 0) ? valid_partials : list_len;
    const int n_merge = min(kRedMaxPartials, max(0, min(n_merge_unclamped, list_len)));

    // Each warp processes one Q head.
    const int my_head = head_start_idx + warp_id;
    const bool warp_active = warp_id < kRedGroupSize;
    auto attn_lse_inter = g.get_attn_lse_inter_mb();
    auto attn_out_inter = g.get_attn_out_inter_mb();
    auto attn_out = g.get_attn_out_mb();
    auto barriers = g.get_barrier();
    auto t_lse = make_tensor(
        make_smem_ptr(s.lse_workspace),
        make_shape(Int<kRedGroupSize>{}, Int<kRedMaxPartials>{}),
        LayoutRight{});
    auto t_data = make_tensor(
        make_smem_ptr(s.data_workspace),
        make_shape(Int<2>{}, Int<kRedGroupSize>{}, Int<kRedHeadDim>{}),
        LayoutRight{});

    // ========================================================================
    // 2. Check active status
    // ========================================================================
    const bool active = (is_active == 1) ||
                        (is_active == -1 && row_active(g, batch_idx));

    // ========================================================================
    // 3. Barrier: wait for SplitKV (opcode 2) to complete for this batch/head.
    //    barriers[layer, 1, b*stride + my_head] must reach num_partials.
    // ========================================================================
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < kRedGroupSize; ++i) {
            int abs_head = head_start_idx + i;
            volatile int32_t* bar = g.barrier_for_batch(
                layer_idx, OPCODE_ATTENTION_PARTIAL - 1, batch_idx, abs_head);
            barrier_wait(bar, num_partials, blockIdx.x,
                                      OPCODE_ATTENTION_REDUCTION, abs_head, layer_idx, &g, inst_idx);
        }
    }
    __syncthreads();

    // ========================================================================
    // 4. Handle inactive rows: zero-fill attn_out and update barrier.
    // ========================================================================
    if (!active) {
        if (warp_active) {
            for (int d = lane_id; d < kRedHeadDim; d += kRedWarpSize) {
                attn_out(batch_idx, my_head, d) = bfloat16_t(0.f);
            }
        }
        __syncthreads();
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            prof_stage_begin = prof_clock_now();
        }
#endif
        if (tid == 0) {
            barrier_publish(
                &barriers(layer_idx, OPCODE_ATTENTION_REDUCTION - 1, batch_idx),
                kRedGroupSize);
        }
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            PROF_ADD_STAGE_CYCLES(
                g, static_cast<int>(blockIdx.x), inst_idx, PROF_FIELD_PUBLISH, prof_clock_now() - prof_stage_begin);
        }
#endif
        return;
    }

    // ========================================================================
    // 5. Load partial LSEs to shared memory.
    // ========================================================================
    if (warp_active) {
        #pragma unroll 2
        for (int step = lane_id; step < n_merge; step += kRedWarpSize) {
            int part_idx = smem.instruction[6 + step];
            if (part_idx < 0 || part_idx >= g.max_splits) {
#if DEBUG_PRINT
                if (lane_id == 0) {
                    printf("reduction invalid partial_idx=%d max_splits=%d step=%d\n",
                           part_idx, g.max_splits, step);
                }
#endif
                __trap();
            }
            float lse = attn_lse_inter(batch_idx, my_head, part_idx);
            t_lse(warp_id, step) = splitkv_lse_is_contributing(lse) ? lse : -INFINITY;
        }
        __syncwarp();
    }

    // ========================================================================
    // 6. Reduction loop with double-buffered partial O staging.
    // ========================================================================
    if (warp_active) {
        float acc_o[kRedElemsPerThread] = {0.f, 0.f};
        float acc_lse = -INFINITY;
        bool has_contrib = false;

        auto load_stage_async = [&](int part_idx, int stage_idx) {
            if (part_idx < 0 || part_idx >= g.max_splits) {
#if DEBUG_PRINT
                if (lane_id == 0) {
                    printf("reduction invalid output partial_idx=%d max_splits=%d step=%d\n",
                           part_idx, g.max_splits, stage_idx);
                }
#endif
                __trap();
            }
            if (lane_id < 16) {
                int vec_offset = lane_id * 4;
                cp_async_cg<16>(&t_data(stage_idx, warp_id, vec_offset),
                                &attn_out_inter(batch_idx, my_head, part_idx, vec_offset));
            }
        };

        if (n_merge > 0) {
            load_stage_async(smem.instruction[6], 0);
            cp_async_fence();
        }

        int stage = 0;
        for (int step = 0; step < n_merge; ++step) {
            if (step + 1 < n_merge) {
                int next_part_idx = smem.instruction[6 + step + 1];
                load_stage_async(next_part_idx, stage ^ 1);
                cp_async_fence();
            }

            if (step + 1 < n_merge) cp_async_wait<1>();
            else                    cp_async_wait<0>();
            __syncwarp();

            float curr_lse = t_lse(warp_id, step);
            float curr_o[kRedElemsPerThread];
            #pragma unroll
            for (int j = 0; j < kRedElemsPerThread; ++j) {
                curr_o[j] = t_data(stage, warp_id, lane_id + j * kRedWarpSize);
            }
            has_contrib |= splitkv_merge_normalized_o_log2(acc_o, acc_lse, curr_o, curr_lse);
            stage ^= 1;
        }

        // ====================================================================
        // 7. Write output.
        // ====================================================================
        #pragma unroll
        for (int j = 0; j < kRedElemsPerThread; ++j) {
            int d = lane_id + j * kRedWarpSize;
            float out_val = has_contrib ? acc_o[j] : 0.f;
            if (is_terminal) {
                attn_out(batch_idx, my_head, d) = bfloat16_t(out_val);
            } else {
                attn_out_inter(batch_idx, my_head, output_part_idx, d) = out_val;
            }
        }

        if (!is_terminal && lane_id == 0) {
            attn_lse_inter(batch_idx, my_head, output_part_idx) =
                has_contrib ? acc_lse : kSplitKvLseMergedEmptyLog2;
        }
    }
    __syncthreads();

    // ========================================================================
    // 8. Update barrier: row 2 is aggregated per batch for downstream OProj.
    // ========================================================================
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    if (tid == 0) {
        barrier_publish(
            &barriers(layer_idx, OPCODE_ATTENTION_REDUCTION - 1, batch_idx),
            kRedGroupSize);
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(
            g, static_cast<int>(blockIdx.x), inst_idx, PROF_FIELD_PUBLISH, prof_clock_now() - prof_stage_begin);
    }
#endif
}

}  // namespace megakernel
