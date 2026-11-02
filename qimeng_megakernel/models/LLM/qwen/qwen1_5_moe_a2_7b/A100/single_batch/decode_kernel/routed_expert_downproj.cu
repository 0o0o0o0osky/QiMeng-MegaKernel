// Routed-expert DownProj (opcode 13), a pure selected-weight GEMV stage.
// GEMV(routed_silu_buffer[expert_idx], expert_down_proj[layer,eid]) →
// deterministic expert-slot accumulation in routed_out_f32.
// Per-tile barrier: barriers[layer, 12, tile_idx] += 1 after each tile.
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

namespace {
constexpr int RD_TileM   = 16;
constexpr int RD_TileK   = QwenMoeConfig::MOE_INTERM_DIM;   // 1408
constexpr int RD_HidDim  = QwenMoeConfig::HIDDEN_DIM;       // 2048
constexpr int RD_IntDim  = QwenMoeConfig::MOE_INTERM_DIM;   // 1408
constexpr int RD_NumWarps = KernelConfig::NUM_WARPS;
constexpr int RD_K_chunks = RD_TileK * sizeof(bfloat16_t) / 16;
}

struct RoutedDownProjSmem {
    alignas(128) bfloat16_t activation[RD_TileK];
    alignas(128) bfloat16_t weight_stage_0[RD_TileM * RD_TileK];
    alignas(128) bfloat16_t weight_stage_1[RD_TileM * RD_TileK];
    alignas(128) float gemv_output[RD_TileM];
    alignas(128) float reduction_workspace[RD_NumWarps];
};

__device__ __forceinline__ void routed_expert_downproj(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    RoutedDownProjSmem& s = *reinterpret_cast<RoutedDownProjSmem*>(smem.main_workspace);

    struct ParsedInstruction {
        int layer_idx;
        int expert_idx;
        int tile_M_idx;
        int tile_N_start_idx;
        int tile_N_end_idx;
        int tile_M_granularity;
        int tile_N_granularity;
        int num_blocks;

        __device__ __forceinline__ ParsedInstruction(const int32_t* smem_inst) {
            layer_idx          = smem_inst[1];
            expert_idx         = smem_inst[2];
            tile_M_idx         = smem_inst[3];
            tile_N_start_idx   = smem_inst[4];
            tile_N_end_idx     = smem_inst[5];
            tile_M_granularity = smem_inst[6];
            tile_N_granularity = smem_inst[7];
            num_blocks         = tile_N_end_idx - tile_N_start_idx;
        }
    };

    ParsedInstruction inst(smem.instruction);

    if (inst.num_blocks == 0) return;

    int tid = threadIdx.x;

    // Wait for all routed upgate tiles to complete: barriers[layer, 10, 0] == NUM_ROUTED_UPGATE_BLOCKS
    {
        constexpr int wait_row = OPCODE_ROUTED_UPGATE_SILU - 1;
        auto t_barriers = g.get_barrier();
        const int32_t* bar = &t_barriers(inst.layer_idx, wait_row, 0);
        barrier_wait(bar, QwenMoeConfig::NUM_ROUTED_UPGATE_BLOCKS);
    }
    __syncthreads();

    if (inst.num_blocks > 0) {
        bfloat16_t* s_input = s.activation;
        {
            const bfloat16_t* g_input_ptr = g.get_routed_silu_buffer().data().get()
                + static_cast<long>(inst.expert_idx) * RD_IntDim;
            auto t_g = make_tensor(make_gmem_ptr(reinterpret_cast<const uint128_t*>(g_input_ptr)),
                                   make_shape(Int<RD_K_chunks>{}), LayoutRight{});
            auto t_s = make_tensor(make_smem_ptr(reinterpret_cast<uint128_t*>(s_input)),
                                   make_shape(Int<RD_K_chunks>{}), LayoutRight{});
            load_tile_async(t_g, t_s);
        }
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        int eid = static_cast<int>(g.get_expert_indices()(inst.expert_idx));
        float expert_w_scale = g.get_expert_weights()(inst.expert_idx);
        const bfloat16_t* weight_base = g.get_expert_down_proj(inst.layer_idx, eid);
        auto g_layer_weight = make_tensor(
            make_gmem_ptr(weight_base),
            make_shape(Int<RD_HidDim>{}, Int<RD_IntDim>{}),
            LayoutRight{}
        );

        {
            int global_block_idx = inst.tile_N_start_idx;
            long row_start = global_block_idx * RD_TileM;
            for (int row = 0; row < RD_TileM; ++row) {
                if (row_start + row < RD_HidDim) {
                    auto t_g_row = make_tensor(
                        make_gmem_ptr(reinterpret_cast<const uint128_t*>(&g_layer_weight(row_start + row, 0))),
                        make_shape(Int<RD_K_chunks>{}), LayoutRight{});
                    auto t_s_row = make_tensor(
                        make_smem_ptr(reinterpret_cast<uint128_t*>(&s.weight_stage_0[row * RD_TileK])),
                        make_shape(Int<RD_K_chunks>{}), LayoutRight{});
                    load_tile_async(t_g_row, t_s_row);
                }
            }
            cp_async_fence();
        }

        for (int i = 0; i < inst.num_blocks; ++i) {
            int stage_curr = i % 2;
            int stage_next = (i + 1) % 2;
            int global_block_idx = inst.tile_N_start_idx + i;
            long row_start = global_block_idx * RD_TileM;

            if (i + 1 < inst.num_blocks) {
                int next_block = inst.tile_N_start_idx + (i + 1);
                long next_row  = next_block * RD_TileM;
                bfloat16_t* s_dst = (stage_next == 0) ? s.weight_stage_0 : s.weight_stage_1;
                for (int row = 0; row < RD_TileM; ++row) {
                    if (next_row + row < RD_HidDim) {
                        auto t_g_row = make_tensor(
                            make_gmem_ptr(reinterpret_cast<const uint128_t*>(&g_layer_weight(next_row + row, 0))),
                            make_shape(Int<RD_K_chunks>{}), LayoutRight{});
                        auto t_s_row = make_tensor(
                            make_smem_ptr(reinterpret_cast<uint128_t*>(&s_dst[row * RD_TileK])),
                            make_shape(Int<RD_K_chunks>{}), LayoutRight{});
                        load_tile_async(t_g_row, t_s_row);
                    }
                }
                cp_async_fence();
            }

            if (i + 1 < inst.num_blocks) { cp_async_wait<1>(); } else { cp_async_wait<0>(); }
            __syncthreads();

            bfloat16_t* s_weight = (stage_curr == 0) ? s.weight_stage_0 : s.weight_stage_1;
            auto t_s_input  = make_tensor(make_smem_ptr(s_input), make_shape(Int<RD_TileK>{}), LayoutRight{});
            auto t_s_weight = make_tensor(make_smem_ptr(s_weight),
                                          make_shape(Int<RD_TileM>{}, Int<RD_TileK>{}), LayoutRight{});
            auto t_s_reduce = make_tensor(make_smem_ptr(s.reduction_workspace),
                                          make_shape(Int<RD_NumWarps>{}), LayoutRight{});
            auto t_s_output = make_tensor(make_smem_ptr(s.gemv_output), make_shape(Int<RD_TileM>{}), LayoutRight{});

            compute_gemv_tile<RD_NumWarps, RD_TileM, RD_TileK>(
                t_s_input, t_s_weight, t_s_reduce, t_s_output);
            __syncthreads();

            // Reuse the per-tile completion counter as the accumulation turn:
            // slot 0, 1, 2, 3 update in that order, matching the PyVM sequence.
            auto t_barriers = g.get_barrier();
            if (inst.expert_idx > 0) {
                barrier_wait(
                    &t_barriers(inst.layer_idx, OPCODE_ROUTED_DOWNPROJ - 1, global_block_idx),
                    inst.expert_idx
                );
            }
            __syncthreads();

            auto t_routed_out_f32 = g.get_routed_out_f32();
            if (tid < RD_TileM) {
                int global_row = row_start + tid;
                if (global_row < RD_HidDim) {
                    float val = s.gemv_output[tid] * expert_w_scale;
                    t_routed_out_f32(global_row) += val;
                }
            }

            __syncthreads();

            if (tid == 0) {
                barrier_publish(
                    &t_barriers(inst.layer_idx, OPCODE_ROUTED_DOWNPROJ - 1, global_block_idx), 1
                );
            }
        }
    }
}

}  // namespace megakernel
