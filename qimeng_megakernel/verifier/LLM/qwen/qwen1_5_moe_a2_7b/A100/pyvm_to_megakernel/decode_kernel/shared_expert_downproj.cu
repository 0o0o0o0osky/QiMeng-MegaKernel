// Qwen1.5-MoE A100: shared expert DownProj (opcode 11), split-K with
// deterministic BF16 accumulation order.
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
constexpr int DP_TileM = 16;
constexpr int DP_TileK = QwenMoeConfig::SPLIT_K_GRANULARITY_SHARED;  // 1408
constexpr int DP_HiddenDim = QwenMoeConfig::HIDDEN_DIM;
constexpr int DP_IntermDim = QwenMoeConfig::INTERM_DIM;
constexpr int DP_NumWarps = KernelConfig::NUM_WARPS;
}

struct SharedDownProjSmem {
    alignas(128) bfloat16_t activation[DP_TileK];
    alignas(128) bfloat16_t weight_stage_0[DP_TileM * DP_TileK];
    alignas(128) bfloat16_t weight_stage_1[DP_TileM * DP_TileK];
    alignas(128) float gemv_output[DP_TileM];
    alignas(128) float reduction_workspace[DP_NumWarps];
};

__device__ __forceinline__ void shared_expert_downproj(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    SharedDownProjSmem& s = *reinterpret_cast<SharedDownProjSmem*>(smem.main_workspace);

    struct ParsedInstruction {
        int layer_idx;
        int tile_M_idx;
        int tile_N_start_idx;
        int tile_N_end_idx;
        int split_K_idx;
        int tile_M_granularity;
        int tile_N_granularity;
        int split_K_granularity;
        int num_blocks;

        __device__ __forceinline__ ParsedInstruction(const int32_t* smem_inst) {
            layer_idx           = smem_inst[1];
            tile_M_idx          = smem_inst[2];
            tile_N_start_idx    = smem_inst[3];
            tile_N_end_idx      = smem_inst[4];
            split_K_idx         = smem_inst[5];
            tile_M_granularity = smem_inst[6];
            tile_N_granularity = smem_inst[7];
            split_K_granularity = smem_inst[8];
            num_blocks          = tile_N_end_idx - tile_N_start_idx;
        }
    };

    ParsedInstruction inst(smem.instruction);

    if (inst.num_blocks == 0) return;

    int tid = threadIdx.x;

    {
        constexpr int wait_opcode_row = OPCODE_SHARED_UPGATE_SILU - 1;  // 9
        auto t_barriers = g.get_barrier();
        const int32_t* bar = &t_barriers(inst.layer_idx, wait_opcode_row, 0);
        barrier_wait(bar, QwenMoeConfig::NUM_SHARED_UPGATE_TILES);
    }
    __syncthreads();

    constexpr int K_chunks = DP_TileK * sizeof(bfloat16_t) / 16;
    auto t_g_silu_out = g.get_shared_expert_silu_out();
    bfloat16_t* s_input = s.activation;
    {
        long input_offset = inst.split_K_idx * inst.split_K_granularity;
        const bfloat16_t* g_input_ptr = t_g_silu_out.data().get() + input_offset;
        auto t_g = make_tensor(
            make_gmem_ptr(reinterpret_cast<const uint128_t*>(g_input_ptr)),
            make_shape(Int<K_chunks>{}),
            LayoutRight{}
        );
        auto t_s = make_tensor(
            make_smem_ptr(reinterpret_cast<uint128_t*>(s_input)),
            make_shape(Int<K_chunks>{}),
            LayoutRight{}
        );
        load_tile_async(t_g, t_s);
    }
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    auto g_layer_weight = g.get_shared_expert_down_proj_weight()(inst.layer_idx, _, _);
    long col_start = inst.split_K_idx * inst.split_K_granularity;

    float* s_gemv_out = s.gemv_output;
    float* s_reduce_workspace = s.reduction_workspace;

    if (inst.num_blocks > 0) {
        int global_block_idx = inst.tile_N_start_idx;
        long row_start = global_block_idx * inst.tile_N_granularity;
        for (int row = 0; row < DP_TileM; ++row) {
            if (row_start + row < DP_HiddenDim) {
                auto t_g_row = make_tensor(
                    make_gmem_ptr(reinterpret_cast<const uint128_t*>(&g_layer_weight(row_start + row, col_start))),
                    make_shape(Int<K_chunks>{}),
                    LayoutRight{}
                );
                auto t_s_row = make_tensor(
                    make_smem_ptr(reinterpret_cast<uint128_t*>(&s.weight_stage_0[row * DP_TileK])),
                    make_shape(Int<K_chunks>{}),
                    LayoutRight{}
                );
                load_tile_async(t_g_row, t_s_row);
            }
        }
        cp_async_fence();
    }

    int total_blocks = inst.num_blocks;
    for (int i = 0; i < total_blocks; ++i) {
        int stage_curr = i % 2;
        int stage_next = (i + 1) % 2;
        int global_block_idx = inst.tile_N_start_idx + i;
        long row_start = global_block_idx * inst.tile_N_granularity;

        if (i + 1 < total_blocks) {
            int next_global_block_idx = inst.tile_N_start_idx + (i + 1);
            long next_row_start = next_global_block_idx * inst.tile_N_granularity;
            bfloat16_t* s_dst = (stage_next == 0) ? s.weight_stage_0 : s.weight_stage_1;
            for (int row = 0; row < DP_TileM; ++row) {
                if (next_row_start + row < DP_HiddenDim) {
                    auto t_g_row = make_tensor(
                        make_gmem_ptr(reinterpret_cast<const uint128_t*>(&g_layer_weight(next_row_start + row, col_start))),
                        make_shape(Int<K_chunks>{}),
                        LayoutRight{}
                    );
                    auto t_s_row = make_tensor(
                        make_smem_ptr(reinterpret_cast<uint128_t*>(&s_dst[row * DP_TileK])),
                        make_shape(Int<K_chunks>{}),
                        LayoutRight{}
                    );
                    load_tile_async(t_g_row, t_s_row);
                }
            }
            cp_async_fence();
        }

        if (i + 1 < total_blocks) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        bfloat16_t* s_weight_curr = (stage_curr == 0) ? s.weight_stage_0 : s.weight_stage_1;

        auto t_s_input = make_tensor(make_smem_ptr(s_input), make_shape(Int<DP_TileK>{}), LayoutRight{});
        auto t_s_weight = make_tensor(make_smem_ptr(s_weight_curr), make_shape(Int<DP_TileM>{}, Int<DP_TileK>{}), LayoutRight{});
        auto t_s_reduce = make_tensor(make_smem_ptr(s_reduce_workspace), make_shape(Int<DP_NumWarps>{}), LayoutRight{});
        auto t_s_output = make_tensor(make_smem_ptr(s_gemv_out), make_shape(Int<DP_TileM>{}), LayoutRight{});

        compute_gemv_tile<DP_NumWarps, DP_TileM, DP_TileK>(t_s_input, t_s_weight, t_s_reduce, t_s_output);
        __syncthreads();

        // The same barrier slot is both the split-order state and the final
        // completion count. This preserves the PyVM stage boundary exactly:
        // shared_out is rounded to BF16 after split 0, 1, 2, then 3.
        auto t_barriers = g.get_barrier();
        if (inst.split_K_idx > 0) {
            barrier_wait(
                &t_barriers(inst.layer_idx, OPCODE_SHARED_DOWNPROJ - 1, global_block_idx),
                inst.split_K_idx
            );
        }
        __syncthreads();

        auto t_g_shared_out = g.get_shared_out();
        if (tid < DP_TileM) {
            int global_row = row_start + tid;
            if (global_row < DP_HiddenDim) {
                float gemv_val = s_gemv_out[tid];
                t_g_shared_out(global_row) = bfloat16_t(
                    float(t_g_shared_out(global_row)) + gemv_val
                );
            }
        }

        __syncthreads();

        // Per-tile barrier: slot = global_block_idx, counts split-K segments landed on this tile.
        // Opcode 14 waits for the shared-downproj row to reach NUM_DOWNPROJ_SPLIT_K_SHARED.
        if (tid == 0) {
            barrier_publish(
                &t_barriers(inst.layer_idx, OPCODE_SHARED_DOWNPROJ - 1, global_block_idx), 1
            );
        }
    }
}

}  // namespace megakernel
