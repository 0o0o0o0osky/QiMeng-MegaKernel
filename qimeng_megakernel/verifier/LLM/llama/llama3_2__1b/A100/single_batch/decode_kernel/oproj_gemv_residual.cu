#include "globals.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int OPROJ_TILE_M = KernelConfig::NUM_WARPS;
constexpr int OPROJ_TILE_K =
    DecodeScheduleConfig::HIDDEN_GEMV_K_SEGMENT;
constexpr int OPROJ_K_SEGMENTS =
    LlamaConfig::HIDDEN_DIM / OPROJ_TILE_K;

static_assert(OPROJ_TILE_K > 0);
static_assert(LlamaConfig::HIDDEN_DIM % OPROJ_TILE_K == 0,
              "OProj hidden GEMV segment must exactly tile HIDDEN_DIM");

// Shared memory layout (CuTe views: activation[HIDDEN_DIM], weight_stages[2][TileM][TileK], gemv_output[TileM], reduction_workspace[NumWarps])
struct OProjResidualSmem {
    struct Weights {
        alignas(128) bfloat16_t stage_0[OPROJ_TILE_M * OPROJ_TILE_K];
        alignas(128) bfloat16_t stage_1[OPROJ_TILE_M * OPROJ_TILE_K];
    };
    struct Inputs {
        alignas(128) bfloat16_t activation[LlamaConfig::HIDDEN_DIM];
    };
    struct Misc {
        alignas(128) float gemv_output[OPROJ_TILE_M];
        alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
    };
    Weights weights;
    Inputs  inputs;
    Misc    misc;
};

static_assert(sizeof(OProjResidualSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "OProj workspace exceeds the selected Ampere CTA budget");

__device__ __forceinline__ void oproj_gemv_residual(
    LlamaGlobals& g,
    SmemLayout&   smem
) {
    constexpr int TileM     = OPROJ_TILE_M;
    constexpr int TileK     = OPROJ_TILE_K;
    constexpr int HiddenDim = LlamaConfig::HIDDEN_DIM;
    constexpr int NumWarps  = KernelConfig::NUM_WARPS;

    auto& op_smem = *reinterpret_cast<OProjResidualSmem*>(smem.main_workspace);

    // ========================================================================
    // 1. Instruction Parsing
    // ========================================================================
    struct ParsedInstruction {
        int layer_idx;
        int tile_M_idx;
        int tile_N_start_idx;
        int tile_N_end_idx;
        int reduction_block_idx;
        int num_blocks;

        __device__ __forceinline__ ParsedInstruction(int32_t* smem_inst) {
            layer_idx            = smem_inst[1];
            tile_M_idx           = smem_inst[2];
            tile_N_start_idx     = smem_inst[3];
            tile_N_end_idx       = smem_inst[4];
            reduction_block_idx  = smem_inst[5];
            num_blocks           = tile_N_end_idx - tile_N_start_idx;
        }
    };

    __syncthreads();
    ParsedInstruction inst(smem.instruction);
    __syncthreads();

    int tid = threadIdx.x;

    // ========================================================================
    // Weight prefetch before barrier (overlap with attention-reduction spin).
    // ========================================================================
    auto t_g_layer = g.get_o_weight()(inst.layer_idx, _, _);

    int total_blocks = inst.num_blocks;
    if (total_blocks > 0) {
        int global_block_idx = inst.tile_N_start_idx;
        for (int row = 0; row < TileM; ++row) {
            const bfloat16_t* row_g = t_g_layer.data().get()
                + static_cast<long>(global_block_idx) * TileM * HiddenDim
                + row * HiddenDim;
            bfloat16_t* row_s = op_smem.weights.stage_0 + row * TileK;
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
    // Barrier Wait (Attention Reduction complete).
    // ========================================================================
    if (tid == 0) {
        int wait_opcode_idx =
            BARRIER_ROW(OPCODE_ATTENTION_REDUCTION);
        auto barriers = g.get_barrier();
        int32_t* bar = &barriers(inst.layer_idx, wait_opcode_idx, 0);
        barrier_wait(bar, LlamaConfig::NUM_Q_HEADS);
    }
    __syncthreads();

    // ========================================================================
    // Load Input (attn_out -> smem); wait for weight + activation cp.async groups.
    // ========================================================================
    {
        auto t_g_attn = g.get_attn_out();
        auto t_s_act = make_tensor(
            make_smem_ptr(op_smem.inputs.activation),
            make_shape(Int<LlamaConfig::HIDDEN_DIM>{}),
            LayoutRight{}
        );
        load_tile_async(t_g_attn, t_s_act);
    }
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    auto t_g_hidden = g.get_hidden_state();

    // Smem tensor views reused by the segmented pipeline.
    auto t_s_reduce = make_tensor(
        make_smem_ptr(op_smem.misc.reduction_workspace),
        make_shape(Int<NumWarps>{}),
        LayoutRight{}
    );
    auto t_s_output = make_tensor(
        make_smem_ptr(op_smem.misc.gemv_output),
        make_shape(Int<TileM>{}),
        LayoutRight{}
    );

    // ========================================================================
    // Main Loop: flatten (output block, hidden-K segment).  The residual slot
    // and publication run once, after the final segment of each output block.
    // ========================================================================
    int total_work = total_blocks * OPROJ_K_SEGMENTS;
    for (int work = 0; work < total_work; ++work) {
        int stage_curr = work % 2;
        int stage_next = (work + 1) % 2;
        int block_offset = work / OPROJ_K_SEGMENTS;
        int segment_idx = work % OPROJ_K_SEGMENTS;
        int global_block_idx = inst.tile_N_start_idx + block_offset;
        int row_start = global_block_idx * TileM;

        // --- Phase 1: Issue next row-strided weight segment. ---
        if (work + 1 < total_work) {
            int next_work = work + 1;
            int next_block_offset = next_work / OPROJ_K_SEGMENTS;
            int next_segment_idx = next_work % OPROJ_K_SEGMENTS;
            int next_global_block_idx =
                inst.tile_N_start_idx + next_block_offset;
            bfloat16_t* stage_next_ptr = (stage_next == 0) ? op_smem.weights.stage_0 : op_smem.weights.stage_1;
            for (int row = 0; row < TileM; ++row) {
                const bfloat16_t* row_g = t_g_layer.data().get()
                    + static_cast<long>(next_global_block_idx) * TileM * HiddenDim
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

        // --- Phase 2: Wait current tile ---
        if (work + 1 < total_work) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // --- Phase 3: Compute GEMV (CuTe tensor indexing) ---
        bfloat16_t* smem_weight_curr = (stage_curr == 0) ? op_smem.weights.stage_0 : op_smem.weights.stage_1;
        auto t_s_weight_curr = make_tensor(
            make_smem_ptr(smem_weight_curr),
            make_shape(Int<TileM>{}, Int<TileK>{}),
            LayoutRight{}
        );
        auto t_s_input_segment = make_tensor(
            make_smem_ptr(op_smem.inputs.activation + segment_idx * TileK),
            make_shape(Int<TileK>{}),
            LayoutRight{}
        );
        if (segment_idx == 0) {
            compute_gemv_tile<NumWarps, TileM, TileK>(
                t_s_input_segment, t_s_weight_curr, t_s_reduce, t_s_output);
        } else {
            compute_gemv_tile<NumWarps, TileM, TileK, 1, true>(
                t_s_input_segment, t_s_weight_curr, t_s_reduce, t_s_output);
        }
        __syncthreads();

        if (segment_idx + 1 != OPROJ_K_SEGMENTS) {
            continue;
        }

        // Phase 4: Epilogue - residual add + write (index by row, not raw offset).
        bool row_valid = (tid < TileM);
        int row_in_tile = tid;
        int global_row = row_start + row_in_tile;

        if (row_valid && global_row < HiddenDim) {
            float gemv_val = t_s_output(row_in_tile);
            float res_val = static_cast<float>(t_g_hidden(global_row));
            t_g_hidden(global_row) = static_cast<bfloat16_t>(gemv_val + res_val);
        }

        __syncthreads();

        if (tid == 0) {
            int my_opcode_idx =
                BARRIER_ROW(OPCODE_O_PROJ_RESIDUAL);
            auto barriers = g.get_barrier();
            barrier_publish(&barriers(inst.layer_idx, my_opcode_idx, 0), 1);
        }
    }
}

} // namespace megakernel
