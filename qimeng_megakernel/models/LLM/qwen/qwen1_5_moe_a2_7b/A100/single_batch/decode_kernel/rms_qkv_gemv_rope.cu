#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int RMS_QKV_TILE_M = 16;
constexpr int RMS_QKV_TILE_K = QwenMoeConfig::HIDDEN_DIM;

struct RmsQkvRopeSmem {
    struct Weights {
        alignas(128) bfloat16_t stage_0[RMS_QKV_TILE_M * QwenMoeConfig::HIDDEN_DIM];
        alignas(128) bfloat16_t stage_1[RMS_QKV_TILE_M * QwenMoeConfig::HIDDEN_DIM];
    };
    struct Inputs {
        alignas(128) bfloat16_t activation[QwenMoeConfig::HIDDEN_DIM];
    };
    struct Misc {
        alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
        alignas(128) float gemv_output[RMS_QKV_TILE_M];
    };
    Weights weights;
    Inputs  inputs;
    Misc    misc;
};

__device__ __forceinline__ void rms_qkv_gemv_rope(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    constexpr int SlicedK = 1;

    struct ParsedInstruction {
        int layer_idx;
        int tile_M_idx;
        int start_block_idx;
        int end_block_idx;
        int iters;
        __device__ __forceinline__ explicit ParsedInstruction(const int32_t* smem_inst) {
            layer_idx       = smem_inst[1];
            tile_M_idx      = smem_inst[2];
            start_block_idx = smem_inst[3];
            end_block_idx   = smem_inst[4];
            iters           = end_block_idx - start_block_idx;
        }
    };

    ParsedInstruction inst(smem.instruction);
    constexpr int TileM        = RMS_QKV_TILE_M;
    constexpr int TileK       = RMS_QKV_TILE_K;
    constexpr int HeadDim      = QwenMoeConfig::HEAD_DIM;
    constexpr int TilesPerHead = HeadDim / TileM;

    constexpr int Q_End = QwenMoeConfig::NUM_Q_HEADS * HeadDim;
    constexpr int K_End = Q_End + QwenMoeConfig::NUM_KV_HEADS * HeadDim;
    constexpr int V_End = K_End + QwenMoeConfig::NUM_KV_HEADS * HeadDim;

    auto& op_smem = *reinterpret_cast<RmsQkvRopeSmem*>(smem.main_workspace);

    if (inst.layer_idx > 0) {
        int prev_layer  = inst.layer_idx - 1;
        int wait_opcode = QwenMoeConfig::PREV_LAYER_BARRIER_ROW;
        int wait_slot   = 0;
        auto t_barriers = g.get_barrier();
        const int32_t* bar_ptr = &t_barriers(prev_layer, wait_opcode, wait_slot);
        barrier_wait(bar_ptr, QwenMoeConfig::PREV_LAYER_EXPECTED);
    }

    auto t_g_qkv_w    = g.get_qkv_weight()(inst.layer_idx, _, _);
    auto t_g_qkv_bias = g.get_qkv_bias()(inst.layer_idx, _);
    auto t_g_input    = g.get_hidden_state();
    auto t_g_rms_w    = g.get_attn_norm()(inst.layer_idx, _);
    auto t_g_q_out    = g.get_q_post_rope();
    auto t_g_k_cache  = g.get_k_cache()(inst.layer_idx, 0, _, _, _);
    auto t_g_v_cache  = g.get_v_cache()(inst.layer_idx, 0, _, _, _);
    auto t_g_rope_cos = g.get_rope_cos()(g.pos_id, _);
    auto t_g_rope_sin = g.get_rope_sin()(g.pos_id, _);

    auto t_s_norm = make_tensor(
        make_smem_ptr(op_smem.inputs.activation),
        make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
        LayoutRight{}
    );
    auto t_reduce_workspace = make_tensor(
        make_smem_ptr(op_smem.misc.reduction_workspace),
        make_shape(Int<KernelConfig::NUM_WARPS>{}),
        LayoutRight{}
    );
    auto t_s_gemv_out = make_tensor(
        make_smem_ptr(op_smem.misc.gemv_output),
        make_shape(Int<RMS_QKV_TILE_M>{}),
        LayoutRight{}
    );

    auto t_barriers = g.get_barrier();

    int total_tiles = inst.iters;

    if (total_tiles > 0) {
        int global_tile_idx = inst.start_block_idx;
        for (int row = 0; row < TileM; ++row) {
            const bfloat16_t* row_g = t_g_qkv_w.data().get()
                + static_cast<long>(global_tile_idx) * TileM * QwenMoeConfig::HIDDEN_DIM + row * QwenMoeConfig::HIDDEN_DIM;
            bfloat16_t* row_s = op_smem.weights.stage_0 + row * TileK;
            auto t_g_row = make_tensor(make_gmem_ptr(row_g), make_shape(Int<TileK>{}), LayoutRight{});
            auto t_s_row = make_tensor(make_smem_ptr(row_s), make_shape(Int<TileK>{}), LayoutRight{});
            load_tile_async(t_g_row, t_s_row);
        }
        cp_async_fence();
    }

    rms_norm<KernelConfig::NUM_WARPS>(
        t_g_input,
        t_g_rms_w,
        t_s_norm,
        g.rms_norm_eps,
        t_reduce_workspace
    );
    __syncthreads();

    for (int i = 0; i < total_tiles; ++i) {
        int stage_curr      = i % 2;
        int stage_next      = (i + 1) % 2;
        int global_tile_idx = inst.start_block_idx + i;

        if (i + 1 < total_tiles) {
            int next_global_tile_idx = inst.start_block_idx + (i + 1);
            bfloat16_t* stage_next_ptr = (stage_next == 0) ? op_smem.weights.stage_0 : op_smem.weights.stage_1;
            for (int row = 0; row < TileM; ++row) {
                const bfloat16_t* row_g = t_g_qkv_w.data().get()
                    + static_cast<long>(next_global_tile_idx) * TileM * QwenMoeConfig::HIDDEN_DIM + row * QwenMoeConfig::HIDDEN_DIM;
                bfloat16_t* row_s = stage_next_ptr + row * TileK;
                auto t_g_row = make_tensor(make_gmem_ptr(row_g), make_shape(Int<TileK>{}), LayoutRight{});
                auto t_s_row = make_tensor(make_smem_ptr(row_s), make_shape(Int<TileK>{}), LayoutRight{});
                load_tile_async(t_g_row, t_s_row);
            }
            cp_async_fence();
        }

        if (i + 1 < total_tiles) cp_async_wait<1>();
        else                      cp_async_wait<0>();
        __syncthreads();

        bfloat16_t* smem_ptr_curr = (stage_curr == 0) ? op_smem.weights.stage_0 : op_smem.weights.stage_1;
        auto t_s_tile_curr = make_tensor(
            make_smem_ptr(smem_ptr_curr),
            make_shape(Int<TileM>{}, Int<TileK>{}),
            LayoutRight{}
        );
        compute_gemv_tile<KernelConfig::NUM_WARPS, TileM, TileK, SlicedK>(
            t_s_norm,
            t_s_tile_curr,
            t_reduce_workspace,
            t_s_gemv_out
        );
        __syncthreads();

        int tid_inner    = threadIdx.x;
        bool row_valid   = (tid_inner < TileM);
        int row_in_tile  = tid_inner;
        int global_row   = global_tile_idx * TileM + row_in_tile;
        int head_dim_idx = global_row % HeadDim;

        if (row_valid) {
            float val = t_s_gemv_out(row_in_tile);
            val += static_cast<float>(t_g_qkv_bias(global_row));

            constexpr unsigned mask = (TileM == 32) ? 0xffffffffu : (1u << TileM) - 1;
            __syncwarp(mask);

            if (global_row < Q_End) {
                float c = t_g_rope_cos(head_dim_idx);
                float s = t_g_rope_sin(head_dim_idx);
                val = apply_rope_interleaved<TileM>(val, c, s, tid_inner);
                t_g_q_out(global_row) = static_cast<bfloat16_t>(val);
            }
            else if (global_row < K_End) {
                float c = t_g_rope_cos(head_dim_idx);
                float s = t_g_rope_sin(head_dim_idx);
                val = apply_rope_interleaved<TileM>(val, c, s, tid_inner);
                int k_head_idx = (global_row - Q_End) / HeadDim;
                t_g_k_cache(g.pos_id, k_head_idx, head_dim_idx) = static_cast<bfloat16_t>(val);
            }
            else if (global_row < V_End) {
                int v_head_idx = (global_row - K_End) / HeadDim;
                t_g_v_cache(g.pos_id, v_head_idx, head_dim_idx) = static_cast<bfloat16_t>(val);
            }
        }
        __syncthreads();

        __syncthreads();
        if (tid_inner == 0) {
            int head_idx = global_tile_idx / TilesPerHead;
            barrier_publish(
                &t_barriers(inst.layer_idx, OPCODE_RMS_QKV_GEMV_ROPE - 1, head_idx),
                1
            );
        }
    }
}
}  // namespace megakernel
