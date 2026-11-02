// Qwen1.5-MoE A100: routed expert UpGate + SiLU (opcode 11).
// Processes one expert slice: RMSNorm(h) → GEMV(gate+up) → SiLU → routed_silu_buffer[expert_idx].
// Adapted from shared_expert_upgate_silu.cu with per-expert weight indexing.
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

struct RoutedUpGateSiLUSmem {
    alignas(128) bfloat16_t activation[QwenMoeConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t weight_stage_0[16 * QwenMoeConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t weight_stage_1[16 * QwenMoeConfig::HIDDEN_DIM];
    alignas(128) float gate_output[16];
    alignas(128) float up_output[16];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

__device__ __forceinline__ void routed_expert_upgate_silu(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    constexpr int TileM      = 16;
    constexpr int TileK      = QwenMoeConfig::HIDDEN_DIM;       // 2048
    constexpr int HiddenDim  = QwenMoeConfig::HIDDEN_DIM;       // 2048
    constexpr int MoeIntermDim = QwenMoeConfig::MOE_INTERM_DIM; // 1408
    constexpr int NumWarps   = KernelConfig::NUM_WARPS;         // 16

    RoutedUpGateSiLUSmem& s = *reinterpret_cast<RoutedUpGateSiLUSmem*>(smem.main_workspace);

    struct ParsedInstruction {
        int layer_idx;
        int expert_idx;
        int tile_N_start_idx;
        int tile_N_end_idx;
        int tile_M_granularity;
        int tile_N_granularity;
        int num_blocks;

        __device__ __forceinline__ ParsedInstruction(const int32_t* smem_inst) {
            // serialize: [opcode, layer_idx, expert_idx, tile_N_start, tile_N_end,
            //              tile_M_gran, tile_N_gran]
            layer_idx          = smem_inst[1];
            expert_idx         = smem_inst[2];
            tile_N_start_idx   = smem_inst[3];
            tile_N_end_idx     = smem_inst[4];
            tile_M_granularity = smem_inst[5];
            tile_N_granularity = smem_inst[6];
            num_blocks         = tile_N_end_idx - tile_N_start_idx;
        }
    };

    ParsedInstruction inst(smem.instruction);
    int tid = threadIdx.x;

    if (inst.num_blocks == 0) return;

    // Wait for router softmax+topk (op9) to complete: barriers[layer, 8, 0] == 1
    {
        constexpr int wait_row = OPCODE_ROUTER_SOFTMAX_TOPK - 1;  // 8
        auto t_barriers = g.get_barrier();
        const int32_t* bar = &t_barriers(inst.layer_idx, wait_row, 0);
        barrier_wait(bar, 1);
    }
    __syncthreads();

    auto t_s_norm = make_tensor(
        make_smem_ptr(s.activation),
        make_shape(Int<HiddenDim>{}),
        LayoutRight{}
    );
    auto t_reduce_workspace = make_tensor(
        make_smem_ptr(s.reduction_workspace),
        make_shape(Int<NumWarps>{}),
        LayoutRight{}
    );
    if (g.normed_hidden_states_ptr != nullptr) {
        auto t_g_normed = g.get_normed_hidden_state();
        for (int idx = tid; idx < HiddenDim; idx += blockDim.x) {
            t_s_norm(idx) = t_g_normed(idx);
        }
    } else {
        auto t_g_input  = g.get_hidden_state();
        auto t_g_rms_w  = g.get_mlp_norm_weight()(inst.layer_idx, _);
        rms_norm<NumWarps>(t_g_input, t_g_rms_w, t_s_norm, g.rms_norm_eps, t_reduce_workspace);
    }
    __syncthreads();

    // Look up the actual expert id from expert_indices[expert_idx]
    int eid = static_cast<int>(g.get_expert_indices()(inst.expert_idx));

    // gate+up weights are fused: [NUM_EXPERTS, 2*MOE_INTERM, HIDDEN]
    // gate is rows [0, MOE_INTERM), up is rows [MOE_INTERM, 2*MOE_INTERM)
    const bfloat16_t* gate_up_base = g.get_expert_gate_up_proj(inst.layer_idx, eid);
    const bfloat16_t* gate_base = gate_up_base;                              // [0, MoeIntermDim)
    const bfloat16_t* up_base   = gate_up_base + static_cast<long>(MoeIntermDim) * TileK;  // [MoeIntermDim, 2*MoeIntermDim)

    int total_iters = inst.num_blocks * 2;

    // Prologue: prefetch first tile (gate block 0) into stage_0
    if (total_iters > 0) {
        int block_idx = inst.tile_N_start_idx;
        const bfloat16_t* d_ptr = gate_base + static_cast<long>(block_idx) * TileM * TileK;
        auto t_g_tile = make_tensor(make_gmem_ptr(d_ptr),
                                    make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        auto t_s_tile = make_tensor(make_smem_ptr(s.weight_stage_0),
                                    make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        load_tile_async(t_g_tile, t_s_tile);
        cp_async_fence();
    }

    for (int i = 0; i < total_iters; ++i) {
        bool is_up_step    = (i % 2 != 0);   // even=gate, odd=up
        int  curr_pair_idx = i / 2;
        int  curr_block_idx = inst.tile_N_start_idx + curr_pair_idx;

        // Prefetch next tile
        if (i + 1 < total_iters) {
            int  next_iter      = i + 1;
            bool next_is_up     = (next_iter % 2 != 0);
            int  next_pair_idx  = next_iter / 2;
            int  next_block_idx = inst.tile_N_start_idx + next_pair_idx;

            const bfloat16_t* d_next = next_is_up
                ? up_base   + static_cast<long>(next_block_idx) * TileM * TileK
                : gate_base + static_cast<long>(next_block_idx) * TileM * TileK;
            bfloat16_t* s_next = next_is_up ? s.weight_stage_1 : s.weight_stage_0;

            auto t_g_next = make_tensor(make_gmem_ptr(d_next),
                                        make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            auto t_s_next = make_tensor(make_smem_ptr(s_next),
                                        make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            load_tile_async(t_g_next, t_s_next);
            cp_async_fence();
        }

        if (i + 1 < total_iters) { cp_async_wait<1>(); } else { cp_async_wait<0>(); }
        __syncthreads();

        bfloat16_t* s_weight = is_up_step ? s.weight_stage_1 : s.weight_stage_0;
        float*      s_output = is_up_step ? s.up_output       : s.gate_output;

        auto t_s_weight = make_tensor(make_smem_ptr(s_weight),
                                      make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        auto t_s_output_t = make_tensor(make_smem_ptr(s_output), make_shape(Int<TileM>{}), LayoutRight{});
        compute_gemv_tile<NumWarps, TileM, TileK>(
            t_s_norm, t_s_weight, t_reduce_workspace, t_s_output_t);
        __syncthreads();

        // Epilogue: SiLU(gate) * up → routed_silu_buffer[expert_idx, :]
        if (is_up_step) {
            if (tid < TileM) {
                float gate_val = s.gate_output[tid];
                float up_val   = s.up_output[tid];
                float silu_gate = gate_val / (1.0f + __expf(-gate_val));
                float final_val = silu_gate * up_val;
                int global_idx  = curr_block_idx * TileM + tid;
                if (global_idx < MoeIntermDim) {
                    g.get_routed_silu_buffer()(inst.expert_idx, global_idx) = bfloat16_t(final_val);
                }
            }
            __syncthreads();
            // Increment global barrier slot 0: counts total tile-blocks done across all experts
            if (tid == 0) {
                auto t_barriers = g.get_barrier();
                barrier_publish(
                    &t_barriers(inst.layer_idx, OPCODE_ROUTED_UPGATE_SILU - 1, 0), 1
                );
            }
        }
    }
}

}  // namespace megakernel
