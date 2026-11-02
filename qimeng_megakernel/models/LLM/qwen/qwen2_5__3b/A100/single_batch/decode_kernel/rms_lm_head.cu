#include "globals.cuh"
#include "qwen_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int kLmHeadTileM = KernelConfig::NUM_WARPS;
constexpr int kLmHeadTileK =
    DecodeScheduleConfig::HIDDEN_GEMV_K_SEGMENT;
constexpr int kLmHeadKSegments =
    QwenConfig::HIDDEN_DIM / kLmHeadTileK;

static_assert(kLmHeadTileK > 0);
static_assert(QwenConfig::HIDDEN_DIM % kLmHeadTileK == 0,
              "LMHead hidden GEMV segment must exactly tile HIDDEN_DIM");

// Shared memory layout for RMS + LM Head GEMV operation
struct RMSLMHeadSmem {
    // RMS normalization materializes the full hidden vector even when the
    // weight pipeline consumes it in smaller K segments.
    alignas(128) bfloat16_t activation[QwenConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t weight_stage_0[kLmHeadTileM * kLmHeadTileK];
    alignas(128) bfloat16_t weight_stage_1[kLmHeadTileM * kLmHeadTileK];
    alignas(128) float gemv_output[kLmHeadTileM];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

static_assert(sizeof(RMSLMHeadSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "LMHead workspace exceeds the selected Ampere CTA budget");

__device__ __forceinline__ void rms_lmhead_gemv(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    constexpr int TileM = kLmHeadTileM;
    constexpr int TileK = kLmHeadTileK;
    constexpr int HiddenDim = QwenConfig::HIDDEN_DIM;
    constexpr int VocabSize = QwenConfig::VOCAB_SIZE;
    constexpr int NumWarps = KernelConfig::NUM_WARPS;

    // ========================================================================
    // 1. Setup shared memory layout
    // ========================================================================
    RMSLMHeadSmem& s = *reinterpret_cast<RMSLMHeadSmem*>(smem.main_workspace);

    // ========================================================================
    // 2. Instruction Parsing
    // ========================================================================
    struct ParsedInstruction {
        int start_block_idx;
        int end_block_idx;

        __device__ __forceinline__ ParsedInstruction(int32_t* smem_inst) {
            // Serialized format from instruction_defs.py RMS_LMHead_GEMV:
            // [opcode, start_output_block_idx, end_output_block_idx]
            start_block_idx = smem_inst[1];
            end_block_idx = smem_inst[2];
        }
    };

    ParsedInstruction inst(smem.instruction);

    int tid = threadIdx.x;
    int total_blocks = inst.end_block_idx - inst.start_block_idx;

    if (total_blocks == 0) return;

    // Weight prefetch before barrier (overlap with DownProj spin).
    auto t_g_lm_head = g.get_lm_head_weight();

    if (total_blocks > 0) {
        int block_idx = inst.start_block_idx;
        for (int row = 0; row < TileM; ++row) {
            const bfloat16_t* row_g = t_g_lm_head.data().get()
                + static_cast<long>(block_idx) * TileM * HiddenDim
                + row * HiddenDim;
            bfloat16_t* row_s = s.weight_stage_0 + row * TileK;
            auto t_g_row = make_tensor(
                make_gmem_ptr(row_g), make_shape(Int<TileK>{}), LayoutRight{});
            auto t_s_row = make_tensor(
                make_smem_ptr(row_s), make_shape(Int<TileK>{}), LayoutRight{});
            load_tile_async(t_g_row, t_s_row);
        }
        cp_async_fence();
    }
    __syncthreads();

    // ========================================================================
    // Barrier Wait (wait for the final layer DownProj).
    // ========================================================================
    if (tid == 0) {
        int last_layer = QwenConfig::NUM_LAYERS - 1;
        int wait_opcode =
            BARRIER_ROW(OPCODE_DOWN_PROJ_RESIDUAL);
        auto barriers = g.get_barrier();
        int32_t* bar =
            &barriers(last_layer, wait_opcode, 0);
        barrier_wait(
            bar,
            (QwenConfig::HIDDEN_DIM / TileM) *
                (QwenConfig::INTERM_DIM /
                 DecodeScheduleConfig::DOWN_PROJ_SPLIT_K_GRANULARITY));
    }
    __syncthreads();

    // ========================================================================
    // RMS Norm (Final Layer Norm).
    // ========================================================================

    // Input (Global)
    auto t_g_input = g.get_hidden_state();

    // Weight (Global)
    auto t_g_rms_w = g.get_final_rms_weight();

    // Output (Shared)
    auto t_s_norm = make_tensor(
        make_smem_ptr(s.activation),
        make_shape(Int<HiddenDim>{}),
        LayoutRight{}
    );

    // Workspace (Shared)
    auto t_reduce_workspace = make_tensor(
        make_smem_ptr(s.reduction_workspace),
        make_shape(Int<NumWarps>{}),
        LayoutRight{}
    );

    // Execute RMS Norm (Global -> Shared)
    rms_norm<NumWarps>(
        t_g_input,
        t_g_rms_w,
        t_s_norm,
        g.rms_norm_eps,
        t_reduce_workspace
    );

    __syncthreads();

    // ========================================================================
    // LM Head GEMV (Pipeline Loop with Double Buffering).
    // ========================================================================

    auto t_g_logits = g.get_logits();

    auto t_s_gemv_out = make_tensor(
        make_smem_ptr(s.gemv_output),
        make_shape(Int<TileM>{}),
        LayoutRight{}
    );

    // --- Main Loop: flattened output block x hidden-K segment. ---
    int total_work = total_blocks * kLmHeadKSegments;
    for (int work = 0; work < total_work; ++work) {
        int stage_curr = work % 2;
        int stage_next = (work + 1) % 2;
        int block_offset = work / kLmHeadKSegments;
        int segment_idx = work % kLmHeadKSegments;
        int block_idx = inst.start_block_idx + block_offset;

        // Phase 1: Issue next load
        if (work + 1 < total_work) {
            int next_work = work + 1;
            int next_block_offset = next_work / kLmHeadKSegments;
            int next_segment_idx = next_work % kLmHeadKSegments;
            int next_block_idx = inst.start_block_idx + next_block_offset;
            bfloat16_t* stage_next_ptr = (stage_next == 0) ? s.weight_stage_0 : s.weight_stage_1;
            for (int row = 0; row < TileM; ++row) {
                const bfloat16_t* row_g = t_g_lm_head.data().get()
                    + static_cast<long>(next_block_idx) * TileM * HiddenDim
                    + row * HiddenDim
                    + next_segment_idx * TileK;
                bfloat16_t* row_s = stage_next_ptr + row * TileK;
                auto t_g_row = make_tensor(
                    make_gmem_ptr(row_g), make_shape(Int<TileK>{}), LayoutRight{});
                auto t_s_row = make_tensor(
                    make_smem_ptr(row_s), make_shape(Int<TileK>{}), LayoutRight{});
                load_tile_async(t_g_row, t_s_row);
            }
            cp_async_fence();
        }

        // Phase 2: Wait for current load
        if (work + 1 < total_work) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // Phase 3: Compute GEMV
        bfloat16_t* s_weight_curr = (stage_curr == 0) ? s.weight_stage_0 : s.weight_stage_1;
        auto t_s_weight = make_tensor(
            make_smem_ptr(s_weight_curr),
            make_shape(Int<TileM>{}, Int<TileK>{}),
            LayoutRight{}
        );

        auto t_s_input_segment = make_tensor(
            make_smem_ptr(s.activation + segment_idx * TileK),
            make_shape(Int<TileK>{}),
            LayoutRight{}
        );
        if (segment_idx == 0) {
            compute_gemv_tile<NumWarps, TileM, TileK>(
                t_s_input_segment, t_s_weight, t_reduce_workspace, t_s_gemv_out);
        } else {
            compute_gemv_tile<NumWarps, TileM, TileK, 1, true>(
                t_s_input_segment, t_s_weight, t_reduce_workspace, t_s_gemv_out);
        }
        __syncthreads();

        if (segment_idx + 1 != kLmHeadKSegments) {
            continue;
        }

        // Phase 4: Epilogue (Store logits).
        if (tid < TileM) {
            float val = s.gemv_output[tid];
            long output_idx = (long)block_idx * TileM + tid;

            if (output_idx < VocabSize) {
                t_g_logits(output_idx) = bfloat16_t(val);
            }
        }
        // Terminal op: logits are the final product, so there is no downstream publication.
    }
}

} // namespace megakernel
