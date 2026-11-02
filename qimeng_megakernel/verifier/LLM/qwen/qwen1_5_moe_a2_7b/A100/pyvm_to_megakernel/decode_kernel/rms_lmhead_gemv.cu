// Qwen1.5-MoE A100: RMSNorm + LM Head GEMV (opcode 7).
// Waits for MoE_Combine_Commit (barriers[last_layer, 13, 0] == 128), then
// computes RMSNorm(hidden_states, lm_head_norm) + GEMV(lm_head_weight) → logits.
// Aligned to Llama 1B A100 rms_lm_head.cu structure.
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

struct RMSLMHeadSmem {
    alignas(128) bfloat16_t activation[QwenMoeConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t weight_stage_0[16 * QwenMoeConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t weight_stage_1[16 * QwenMoeConfig::HIDDEN_DIM];
    alignas(128) float gemv_output[16];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

__device__ __forceinline__ void rms_lmhead_gemv(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    constexpr int TileM     = 16;
    constexpr int TileK     = QwenMoeConfig::HIDDEN_DIM;   // 2048
    constexpr int HiddenDim = QwenMoeConfig::HIDDEN_DIM;   // 2048
    constexpr int VocabSize = QwenMoeConfig::VOCAB_SIZE;
    constexpr int NumWarps  = KernelConfig::NUM_WARPS;     // 16

    RMSLMHeadSmem& s = *reinterpret_cast<RMSLMHeadSmem*>(smem.main_workspace);

    struct ParsedInstruction {
        int start_block_idx;
        int end_block_idx;
        __device__ __forceinline__ ParsedInstruction(const int32_t* smem_inst) {
            // [opcode, start_output_block_idx, end_output_block_idx, tile_M_gran, tile_N_gran]
            start_block_idx = smem_inst[1];
            end_block_idx   = smem_inst[2];
        }
    };

    ParsedInstruction inst(smem.instruction);
    int tid = threadIdx.x;
    int total_blocks = inst.end_block_idx - inst.start_block_idx;

    if (total_blocks == 0) return;

    // Barrier wait: MoE_Combine_Commit of the last layer must be done.
    // barriers[last_layer, OPCODE_MOE_COMBINE_COMMIT-1, 0] == PREV_LAYER_EXPECTED (128)
    {
        constexpr int last_layer = QwenMoeConfig::NUM_LAYERS - 1;
        auto t_barriers = g.get_barrier();
        const int32_t* bar = &t_barriers(last_layer, OPCODE_MOE_COMBINE_COMMIT - 1, 0);
        barrier_wait(bar, QwenMoeConfig::PREV_LAYER_EXPECTED);
    }
    __syncthreads();

    // RMS Norm
    auto t_g_input = g.get_hidden_state();
    auto t_g_rms_w = g.get_final_rms_weight();
    auto t_s_norm  = make_tensor(
        make_smem_ptr(s.activation),
        make_shape(Int<HiddenDim>{}),
        LayoutRight{}
    );
    auto t_reduce_workspace = make_tensor(
        make_smem_ptr(s.reduction_workspace),
        make_shape(Int<NumWarps>{}),
        LayoutRight{}
    );
    rms_norm<NumWarps>(t_g_input, t_g_rms_w, t_s_norm, g.rms_norm_eps, t_reduce_workspace);
    __syncthreads();

    auto t_g_logits  = g.get_logits();
    auto t_g_lm_head = g.get_lm_head_weight();
    auto t_s_input   = make_tensor(make_smem_ptr(s.activation),
                                   make_shape(Int<TileK>{}), LayoutRight{});
    auto t_s_gemv_out = make_tensor(make_smem_ptr(s.gemv_output),
                                    make_shape(Int<TileM>{}), LayoutRight{});

    // Prologue: load first tile into stage_0
    {
        const bfloat16_t* ptr = t_g_lm_head.data().get()
                                + static_cast<long>(inst.start_block_idx) * TileM * TileK;
        auto t_g_tile = make_tensor(make_gmem_ptr(ptr),
                                    make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        auto t_s_tile = make_tensor(make_smem_ptr(s.weight_stage_0),
                                    make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        load_tile_async(t_g_tile, t_s_tile);
        cp_async_fence();
    }

    for (int i = 0; i < total_blocks; ++i) {
        int stage_curr = i % 2;
        int block_idx  = inst.start_block_idx + i;

        // Prefetch next tile
        if (i + 1 < total_blocks) {
            int stage_next   = (i + 1) % 2;
            int next_block   = inst.start_block_idx + (i + 1);
            bfloat16_t* dst  = (stage_next == 0) ? s.weight_stage_0 : s.weight_stage_1;
            const bfloat16_t* ptr = t_g_lm_head.data().get()
                                    + static_cast<long>(next_block) * TileM * TileK;
            auto t_g_next = make_tensor(make_gmem_ptr(ptr),
                                        make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            auto t_s_next = make_tensor(make_smem_ptr(dst),
                                        make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            load_tile_async(t_g_next, t_s_next);
            cp_async_fence();
        }

        if (i + 1 < total_blocks) { cp_async_wait<1>(); } else { cp_async_wait<0>(); }
        __syncthreads();

        bfloat16_t* s_w = (stage_curr == 0) ? s.weight_stage_0 : s.weight_stage_1;
        auto t_s_weight = make_tensor(make_smem_ptr(s_w),
                                      make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        compute_gemv_tile<NumWarps, TileM, TileK, 2>(
            t_s_input, t_s_weight, t_reduce_workspace, t_s_gemv_out);
        __syncthreads();

        if (tid < TileM) {
            long output_idx = static_cast<long>(block_idx) * TileM + tid;
            if (output_idx < VocabSize) {
                t_g_logits(output_idx) = bfloat16_t(s.gemv_output[tid]);
            }
        }
    }
}

}  // namespace megakernel
