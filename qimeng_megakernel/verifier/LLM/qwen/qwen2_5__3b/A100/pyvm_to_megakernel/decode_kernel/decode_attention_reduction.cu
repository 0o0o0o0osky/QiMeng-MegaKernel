#include "globals.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int kAttentionHeadDim = QwenConfig::HEAD_DIM;
constexpr int kAttentionReductionSize =
    QwenConfig::ATTN_REDUCTION_SIZE;
constexpr int kAttentionMaxPartials =
    DecodeScheduleConfig::ATTENTION_REDUCTION_WORKSPACE_CAPACITY;
constexpr int kAttentionMaxReductionList =
    DecodeScheduleConfig::ATTENTION_REDUCTION_LIST_CAPACITY;
constexpr int kFloatsPerAsyncVector = sizeof(uint128_t) / sizeof(float);
constexpr int kAttentionValuesPerLane =
    kAttentionHeadDim / KernelConfig::WARP_SIZE;
constexpr int kAttentionVectorsPerHead =
    kAttentionHeadDim / kFloatsPerAsyncVector;

static_assert(QwenConfig::NUM_Q_HEADS % kAttentionReductionSize == 0);
static_assert(kAttentionReductionSize <= KernelConfig::NUM_WARPS);
static_assert(kAttentionHeadDim % KernelConfig::WARP_SIZE == 0,
              "each lane must own an integral number of head values");
static_assert(kAttentionMaxPartials <= kAttentionMaxReductionList,
              "reduction workspace fanout must fit the serialized instruction");

// Shared memory layout for attention reduction operation
// CuTe views: t_lse[ReductionSize, MaxPartials], t_data[2, ReductionSize, HeadDim]
struct AttnReductionSmem {
    alignas(128) float
        lse_workspace[kAttentionReductionSize * kAttentionMaxPartials];
    alignas(128) float
        data_workspace[2 * kAttentionReductionSize * kAttentionHeadDim];
};

static_assert(sizeof(AttnReductionSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "attention reduction workspace exceeds the selected Ampere CTA budget");

__device__ __forceinline__ void decode_attention_reduction(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    constexpr int HeadDim = kAttentionHeadDim;
    constexpr int ReductionSize = kAttentionReductionSize;
    constexpr int WarpSize = KernelConfig::WARP_SIZE;
    constexpr int MaxPartials = kAttentionMaxPartials;

    // ========================================================================
    // 0. Setup shared memory layout
    // ========================================================================
    AttnReductionSmem& s = *reinterpret_cast<AttnReductionSmem*>(smem.main_workspace);

    // ========================================================================
    // 1. Instruction Parsing
    // ========================================================================
    struct ParsedInstruction {
        int layer_idx;
        int q_head_start_idx;
        int num_partials;
        int reduction_list_length;
        int32_t* reduction_list;

        __device__ __forceinline__ ParsedInstruction(int32_t* smem_inst) {
            // Serialized format: [opcode, layer_idx, head_start_idx, num_partials,
            //                     len(reduction_list), reduction_list[0..n-1], is_terminal, output_partial_idx]
            layer_idx             = smem_inst[1];
            q_head_start_idx      = smem_inst[2];
            num_partials          = smem_inst[3];
            reduction_list_length = smem_inst[4];  // List length
            reduction_list        = &smem_inst[5]; // List elements start here
        }
    };

    __syncthreads();
    const int encoded_layer_idx = smem.instruction[1];
    const int encoded_q_head_start_idx = smem.instruction[2];
    const int encoded_num_partials = smem.instruction[3];
    const int encoded_reduction_list_length = smem.instruction[4];
    if (encoded_layer_idx < 0 || encoded_layer_idx >= QwenConfig::NUM_LAYERS ||
        encoded_q_head_start_idx < 0 ||
        encoded_q_head_start_idx + kAttentionReductionSize >
            QwenConfig::NUM_Q_HEADS ||
        encoded_q_head_start_idx % kAttentionReductionSize != 0 ||
        encoded_num_partials <= 0 ||
        encoded_num_partials > kAttentionMaxPartials ||
        encoded_num_partials > g.max_splits ||
        encoded_reduction_list_length != encoded_num_partials ||
        encoded_reduction_list_length > kAttentionMaxReductionList) {
        __trap();
    }
    ParsedInstruction inst(smem.instruction);
    __syncthreads();

    int tid = threadIdx.x;
    int warp_id = tid / WarpSize;
    int lane_id = tid % WarpSize;

    bool is_active = (warp_id < ReductionSize);
    int my_q_head_abs = inst.q_head_start_idx + warp_id;

    // ========================================================================
    // 2. CuTe shared-memory tensor views (reduce indexing)
    // ========================================================================
    auto t_lse = make_tensor(
        make_smem_ptr(s.lse_workspace),
        make_shape(Int<ReductionSize>{}, Int<MaxPartials>{}),
        LayoutRight{}
    );
    auto t_data = make_tensor(
        make_smem_ptr(s.data_workspace),
        make_shape(Int<2>{}, Int<ReductionSize>{}, Int<HeadDim>{}),
        LayoutRight{}
    );

    // Empty: every reduction operand is produced by attention partials.

    // Wait for all splitkv partials to complete.
    if (tid == 0) {
        int wait_opcode_idx =
            BARRIER_ROW(OPCODE_ATTENTION_PARTIAL);
        auto barriers = g.get_barrier();

        #pragma unroll
        for (int i = 0; i < ReductionSize; ++i) {
            int32_t* bar = &barriers(inst.layer_idx, wait_opcode_idx, inst.q_head_start_idx + i);
            barrier_wait(bar, inst.num_partials);
        }
    }
    __syncthreads();

    // Load LSE values to shared memory.
    if (is_active) {
        auto attn_lse_inter = g.get_attn_lse_intermediates();  // [NUM_HEADS, max_splits]

        #pragma unroll 2
        for (int i = lane_id; i < inst.num_partials; i += WarpSize) {
            int real_partial_idx = inst.reduction_list[i];
            if (real_partial_idx < 0 ||
                real_partial_idx >= inst.num_partials ||
                real_partial_idx >= g.max_splits) {
#ifndef NDEBUG
                printf("ERROR: LSE load: Invalid partial_idx=%d (max=%d), i=%d\n",
                       real_partial_idx, g.max_splits, i);
#endif
                __trap();
            }
            t_lse(warp_id, i) = attn_lse_inter(my_q_head_abs, real_partial_idx);
        }
        __syncwarp();
    }

    // Reduction loop with double-buffered loading.
    if (is_active) {
        float acc_o[kAttentionValuesPerLane] = {};
        float acc_lse = -INFINITY;
        auto attn_out_inter = g.get_attn_out_intermediates();  // [NUM_HEADS, max_splits, HEAD_DIM]

        auto load_stage_async = [&](int step, int stage_idx) {
            int real_partial_idx = inst.reduction_list[step];
            if (real_partial_idx < 0 ||
                real_partial_idx >= inst.num_partials ||
                real_partial_idx >= g.max_splits) {
#ifndef NDEBUG
                if (lane_id == 0)
                    printf("ERROR: Output load: Invalid partial_idx=%d (max=%d), step=%d\n",
                           real_partial_idx, g.max_splits, step);
#endif
                __trap();
            }
            // Per-warp load: each warp loads its own row (t_data(stage_idx, warp_id, :)); cannot use
            // load_tile_async here because that uses block-wide tid 0..N-1 and would only fill row 0.
            for (int vector = lane_id;
                 vector < kAttentionVectorsPerHead;
                 vector += WarpSize) {
                int vec_offset = vector * kFloatsPerAsyncVector;
                cp_async_cg<16>(&t_data(stage_idx, warp_id, vec_offset),
                               &attn_out_inter(my_q_head_abs, real_partial_idx, vec_offset));
            }
        };

        // Prologue: Load Step 0
        if (inst.num_partials > 0) {
            load_stage_async(0, 0); // Load to Stage 0
            cp_async_fence();       // Commit group
        }

        int stage = 0;
        for (int step = 0; step < inst.num_partials; ++step) {
            // Phase 1: Issue Next (Prefetch)
            if (step + 1 < inst.num_partials) {
                load_stage_async(step + 1, stage ^ 1); // Load to Next Stage
                cp_async_fence(); // Commit group
            }

            // Phase 2: Wait Current
            if (step + 1 < inst.num_partials) {
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncwarp();

            // Phase 3: Compute - Numerically stable softmax reduction
            float curr_lse = t_lse(warp_id, step);
            float curr_o[kAttentionValuesPerLane];
            #pragma unroll
            for (int value = 0; value < kAttentionValuesPerLane; ++value) {
                curr_o[value] = t_data(
                    stage, warp_id, lane_id + value * WarpSize);
            }

            float max_lse = fmaxf(acc_lse, curr_lse);
            float acc_exp = exp2f(acc_lse - max_lse);
            float curr_exp = exp2f(curr_lse - max_lse);

            float new_denom = acc_exp + curr_exp;
            float acc_scale = acc_exp / new_denom;
            float curr_scale = curr_exp / new_denom;

            #pragma unroll
            for (int value = 0; value < kAttentionValuesPerLane; ++value) {
                acc_o[value] =
                    acc_o[value] * acc_scale + curr_o[value] * curr_scale;
            }
            acc_lse = max_lse + log2f(new_denom);

            // Flip Stage
            stage ^= 1;
        }

        // Write final output: 2D view [NUM_Q_HEADS, HEAD_DIM] to avoid raw index.
        auto attn_out_1d = g.get_attn_out();
        auto attn_out = make_tensor(
            make_gmem_ptr(attn_out_1d.data().get()),
            make_shape(Int<QwenConfig::NUM_Q_HEADS>{}, Int<QwenConfig::HEAD_DIM>{}),
            LayoutRight{}
        );
        #pragma unroll
        for (int value = 0; value < kAttentionValuesPerLane; ++value) {
            attn_out(my_q_head_abs, lane_id + value * WarpSize) =
                static_cast<bfloat16_t>(acc_o[value]);
        }
    }

    __syncthreads();

    // ========================================================================
    // 6. Update barriers
    // ========================================================================
    if (tid == 0) {
        int my_opcode_idx =
            BARRIER_ROW(OPCODE_ATTENTION_REDUCTION);
        auto barriers = g.get_barrier();
        // Signal completion for every head owned by this reduction CTA
        barrier_publish(&barriers(inst.layer_idx, my_opcode_idx, 0), ReductionSize);
    }
}

} // namespace megakernel
