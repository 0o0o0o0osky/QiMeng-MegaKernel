// Qwen1.5-MoE A100: oproj_gemv_residual (opcode 4 only).
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int OPROJ_TILE_M = 16;
constexpr int OPROJ_TILE_K = QwenMoeConfig::HIDDEN_DIM;

struct OProjResidualSmem {
    struct Weights {
        alignas(128) bfloat16_t stage_0[OPROJ_TILE_M * OPROJ_TILE_K];
        alignas(128) bfloat16_t stage_1[OPROJ_TILE_M * OPROJ_TILE_K];
    };
    struct Inputs {
        alignas(128) bfloat16_t activation[QwenMoeConfig::HIDDEN_DIM];
    };
    struct Misc {
        alignas(128) float gemv_output[OPROJ_TILE_M];
        alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
    };
    Weights weights;
    Inputs  inputs;
    Misc    misc;
};

__device__ __forceinline__ void oproj_gemv_residual(
    QwenGlobals& g,
    SmemLayout&  smem
) {
    constexpr int TileM     = OPROJ_TILE_M;
    constexpr int TileK     = OPROJ_TILE_K;
    constexpr int HiddenDim = QwenMoeConfig::HIDDEN_DIM;
    constexpr int NumWarps  = KernelConfig::NUM_WARPS;

    auto& op_smem = *reinterpret_cast<OProjResidualSmem*>(smem.main_workspace);

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

    {
        int wait_opcode_idx = 2;
        auto barriers = g.get_barrier();
        const int32_t* bar = &barriers(inst.layer_idx, wait_opcode_idx, 0);
        barrier_wait(bar, QwenMoeConfig::NUM_Q_HEADS);
    }
    __syncthreads();

    {
        auto t_g_attn = g.get_attn_out();
        auto t_s_act = make_tensor(
            make_smem_ptr(op_smem.inputs.activation),
            make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
            LayoutRight{}
        );
        load_tile_async(t_g_attn, t_s_act);
    }
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    auto t_g_layer = g.get_o_weight()(inst.layer_idx, _, _);
    auto t_g_hidden = g.get_hidden_state();

    auto t_s_input = make_tensor(
        make_smem_ptr(op_smem.inputs.activation),
        make_shape(Int<TileK>{}),
        LayoutRight{}
    );
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

    int total_blocks = inst.num_blocks;
    if (total_blocks > 0) {
        int global_block_idx = inst.tile_N_start_idx;
        const bfloat16_t* d_tile_ptr = t_g_layer.data().get() + static_cast<long>(global_block_idx) * TileM * TileK;
        auto t_g_tile = make_tensor(make_gmem_ptr(d_tile_ptr), make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        auto t_s_tile = make_tensor(make_smem_ptr(op_smem.weights.stage_0), make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        load_tile_async(t_g_tile, t_s_tile);
        cp_async_fence();
    }

    for (int i = 0; i < total_blocks; ++i) {
        int stage_curr  = i % 2;
        int stage_next  = (i + 1) % 2;
        int global_block_idx = inst.tile_N_start_idx + i;
        int row_start = global_block_idx * TileM;

        if (i + 1 < total_blocks) {
            int next_global_block_idx = inst.tile_N_start_idx + (i + 1);
            const bfloat16_t* d_next_ptr = t_g_layer.data().get() + static_cast<long>(next_global_block_idx) * TileM * TileK;
            bfloat16_t* stage_next_ptr = (stage_next == 0) ? op_smem.weights.stage_0 : op_smem.weights.stage_1;
            auto t_g_next = make_tensor(make_gmem_ptr(d_next_ptr), make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            auto t_s_next = make_tensor(make_smem_ptr(stage_next_ptr), make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            load_tile_async(t_g_next, t_s_next);
            cp_async_fence();
        }

        if (i + 1 < total_blocks) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        bfloat16_t* smem_weight_curr = (stage_curr == 0) ? op_smem.weights.stage_0 : op_smem.weights.stage_1;
        auto t_s_weight_curr = make_tensor(
            make_smem_ptr(smem_weight_curr),
            make_shape(Int<TileM>{}, Int<TileK>{}),
            LayoutRight{}
        );
        compute_gemv_tile<NumWarps, TileM, TileK>(
            t_s_input,
            t_s_weight_curr,
            t_s_reduce,
            t_s_output
        );
        __syncthreads();

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
            int my_opcode_idx = 3;
            auto barriers = g.get_barrier();
            barrier_publish(&barriers(inst.layer_idx, my_opcode_idx, 0), 1);
        }
    }
}

}  // namespace megakernel
