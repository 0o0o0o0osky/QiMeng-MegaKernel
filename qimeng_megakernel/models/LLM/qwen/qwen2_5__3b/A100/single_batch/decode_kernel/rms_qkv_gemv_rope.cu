#include "globals.cuh"
#include "qwen_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

// Tile configuration for the QKV GEMV.
constexpr int RMS_QKV_TILE_M = KernelConfig::NUM_WARPS;
constexpr int RMS_QKV_TILE_K =
    DecodeScheduleConfig::HIDDEN_GEMV_K_SEGMENT;
constexpr int RMS_QKV_K_SEGMENTS =
    QwenConfig::HIDDEN_DIM / RMS_QKV_TILE_K;

static_assert(RMS_QKV_TILE_K > 0);
static_assert(QwenConfig::HIDDEN_DIM % RMS_QKV_TILE_K == 0,
              "QKV hidden GEMV segment must exactly tile HIDDEN_DIM");

// Shared-memory layout used by the RMS+QKV GEMV+RoPE op,
// carved out of SmemLayout::main_workspace.
struct RmsQkvRopeSmem {
    struct Weights {
        alignas(128) bfloat16_t
            stage_0[RMS_QKV_TILE_M * RMS_QKV_TILE_K];
        alignas(128) bfloat16_t
            stage_1[RMS_QKV_TILE_M * RMS_QKV_TILE_K];
    };

    struct Inputs {
        alignas(128) bfloat16_t
            activation[QwenConfig::HIDDEN_DIM];
    };

    struct Misc {
        // One slot per warp is enough for reductions.
        alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
        alignas(128) float gemv_output[RMS_QKV_TILE_M];
    };

    Weights weights;
    Inputs  inputs;
    Misc    misc;
};

static_assert(sizeof(RmsQkvRopeSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "QKV workspace exceeds the selected Ampere CTA budget");

// ========================================================================
// 1. Kernel Implementation (1x16 pipeline: 16 warps, double-buffer G2S,
//    S2R + register GEMV; fusion points: after G2S [RMS], after R2S [RoPE+write])
//    SlicedK is fixed inside (no template at call site).
// ========================================================================
__device__ __forceinline__ void rms_qkv_gemv_rope(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    struct ParsedInstruction {
        int layer_idx;
        int tile_M_idx;
        int start_block_idx;
        int end_block_idx;
        int iters;
        // Python serializes RMS_QKV_GEMV_RoPE as:
        // [opcode, layer_idx, tile_M_idx, tile_N_start_idx, tile_N_end_idx, tile_M_granularity, tile_N_granularity]
        // Indices:  0       1           2            3                4             5                    6
        __device__ __forceinline__ explicit ParsedInstruction(
            const int32_t* smem_inst
        ) {
            layer_idx       = smem_inst[1];
            tile_M_idx      = smem_inst[2];  // Not used in this kernel (always processing full tiles)
            start_block_idx = smem_inst[3];  // tile_N_start_idx
            end_block_idx   = smem_inst[4];  // tile_N_end_idx
            iters           = end_block_idx - start_block_idx;
        }
    };

    ParsedInstruction inst(smem.instruction);

    int tid = threadIdx.x;

    constexpr int TileM        = RMS_QKV_TILE_M;
    constexpr int TileK        = RMS_QKV_TILE_K;
    constexpr int HeadDim      = QwenConfig::HEAD_DIM;
    constexpr int TilesPerHead = HeadDim / TileM;

    constexpr int Q_End = QwenConfig::NUM_Q_HEADS * HeadDim;
    constexpr int K_End = Q_End + QwenConfig::NUM_KV_HEADS * HeadDim;
    constexpr int V_End = K_End + QwenConfig::NUM_KV_HEADS * HeadDim;
    constexpr int DownProjContributions =
        (QwenConfig::HIDDEN_DIM / TileM) *
        (QwenConfig::INTERM_DIM /
         DecodeScheduleConfig::DOWN_PROJ_SPLIT_K_GRANULARITY);

    // View of the op-specific shared memory.
    auto& op_smem = *reinterpret_cast<RmsQkvRopeSmem*>(smem.main_workspace);

    // ========================================================================
    // Weight prefetch before barrier (overlap HBM with upstream spin)
    // ========================================================================
    auto t_g_qkv_w = g.get_qkv_weight()(inst.layer_idx, _, _);
    auto t_g_qkv_bias = g.get_qkv_bias()(inst.layer_idx, _);

    int total_tiles = inst.iters;
    int total_work = total_tiles * RMS_QKV_K_SEGMENTS;

    if (total_tiles > 0) {
        int global_tile_idx = inst.start_block_idx;
        for (int row = 0; row < TileM; ++row) {
            const bfloat16_t* row_g = t_g_qkv_w.data().get()
                + static_cast<long>(global_tile_idx) * TileM * QwenConfig::HIDDEN_DIM + row * QwenConfig::HIDDEN_DIM;
            bfloat16_t* row_s = op_smem.weights.stage_0 + row * TileK;
            auto t_g_row = make_tensor(make_gmem_ptr(row_g), make_shape(Int<TileK>{}), LayoutRight{});
            auto t_s_row = make_tensor(make_smem_ptr(row_s), make_shape(Int<TileK>{}), LayoutRight{});
            load_tile_async(t_g_row, t_s_row);
        }
        cp_async_fence();
    }
    __syncthreads();

    // ========================================================================
    // Barrier Wait (prior layer DownProj complete)
    // ========================================================================
    if (inst.layer_idx > 0 && tid == 0) {
        int prev_layer  = inst.layer_idx - 1;
        int wait_opcode = BARRIER_ROW(OPCODE_DOWN_PROJ_RESIDUAL);
        int wait_slot   = 0;

        auto t_barriers_wait = g.get_barrier();
        int32_t* bar_ptr = &t_barriers_wait(prev_layer, wait_opcode, wait_slot);

        barrier_wait(bar_ptr, DownProjContributions);
    }
    __syncthreads();

    auto t_g_input = g.get_hidden_state();
    auto t_g_rms_w = g.get_attn_norm()(inst.layer_idx, _);
    auto t_g_q_out = g.get_q_post_rope();
    auto t_g_k_cache = g.get_k_cache()(inst.layer_idx, 0, _, _, _);
    auto t_g_v_cache = g.get_v_cache()(inst.layer_idx, 0, _, _, _);
    auto t_g_rope_cos = g.get_rope_cos()(g.pos_id, _);
    auto t_g_rope_sin = g.get_rope_sin()(g.pos_id, _);

    auto t_s_norm = make_tensor(
        make_smem_ptr(op_smem.inputs.activation),
        make_shape(Int<QwenConfig::HIDDEN_DIM>{}),
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

    // Fusion 1 (after G2S): RMS norm (full activation read, then normalize).
    rms_norm<KernelConfig::NUM_WARPS>(
        t_g_input,
        t_g_rms_w,
        t_s_norm,
        g.rms_norm_eps,
        t_reduce_workspace
    );

    __syncthreads();

    // ========================================================================
    // Main Loop: flatten (N tile, hidden-K segment) into one double-buffered
    // stream.  Segment zero initializes the FP32 row result; later segments
    // accumulate into it.  The RoPE/write slot runs only after the final
    // segment of each output tile.
    // ========================================================================
    for (int work = 0; work < total_work; ++work) {
        int stage_curr = work % 2;
        int stage_next = (work + 1) % 2;
        int tile_offset = work / RMS_QKV_K_SEGMENTS;
        int segment_idx = work % RMS_QKV_K_SEGMENTS;
        int global_tile_idx = inst.start_block_idx + tile_offset;

        // --- Phase 1: Issue next row-strided weight segment. ---
        if (work + 1 < total_work) {
            int next_work = work + 1;
            int next_tile_offset = next_work / RMS_QKV_K_SEGMENTS;
            int next_segment_idx = next_work % RMS_QKV_K_SEGMENTS;
            int next_global_tile_idx =
                inst.start_block_idx + next_tile_offset;
            bfloat16_t* stage_next_ptr = (stage_next == 0) ? op_smem.weights.stage_0 : op_smem.weights.stage_1;
            for (int row = 0; row < TileM; ++row) {
                const bfloat16_t* row_g = t_g_qkv_w.data().get()
                    + static_cast<long>(next_global_tile_idx) * TileM * QwenConfig::HIDDEN_DIM
                    + row * QwenConfig::HIDDEN_DIM
                    + next_segment_idx * TileK;
                bfloat16_t* row_s = stage_next_ptr + row * TileK;
                auto t_g_row = make_tensor(make_gmem_ptr(row_g), make_shape(Int<TileK>{}), LayoutRight{});
                auto t_s_row = make_tensor(make_smem_ptr(row_s), make_shape(Int<TileK>{}), LayoutRight{});
                load_tile_async(t_g_row, t_s_row);
            }
            cp_async_fence();
        }

        // --- Phase 2: Wait ---
        if (work + 1 < total_work) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // --- Phase 3: Compute (S2R + register GEMV + R2S) ---
        bfloat16_t* smem_ptr_curr = (stage_curr == 0)
            ? op_smem.weights.stage_0
            : op_smem.weights.stage_1;

        auto t_s_input_segment = make_tensor(
            make_smem_ptr(op_smem.inputs.activation + segment_idx * TileK),
            make_shape(Int<TileK>{}),
            LayoutRight{}
        );
        auto t_s_tile_curr = make_tensor(
            make_smem_ptr(smem_ptr_curr),
            make_shape(Int<TileM>{}, Int<TileK>{}),
            LayoutRight{}
        );

        if (segment_idx == 0) {
            compute_gemv_tile<KernelConfig::NUM_WARPS, TileM, TileK>(
                t_s_input_segment,
                t_s_tile_curr,
                t_reduce_workspace,
                t_s_gemv_out
            );
        } else {
            compute_gemv_tile<KernelConfig::NUM_WARPS, TileM, TileK, 1, true>(
                t_s_input_segment,
                t_s_tile_curr,
                t_reduce_workspace,
                t_s_gemv_out
            );
        }
        __syncthreads();

        if (segment_idx + 1 != RMS_QKV_K_SEGMENTS) {
            continue;
        }

        // Phase 4: Epilogue (R2S) + Fusion 2: RoPE + global write.
        int tid_inner = threadIdx.x;
        bool row_valid = (tid_inner < TileM);
        int row_in_tile    = tid_inner;
        int global_row_idx = global_tile_idx * TileM + row_in_tile;
        int head_dim_idx   = global_row_idx % HeadDim;

        if (row_valid) {
            float val = t_s_gemv_out(row_in_tile);
            val += static_cast<float>(t_g_qkv_bias(global_row_idx));

            constexpr unsigned mask =
                (TileM == KernelConfig::WARP_SIZE)
                    ? 0xffffffffu
                    : (1u << TileM) - 1;
            __syncwarp(mask);

            if (global_row_idx < Q_End) {
                float c = t_g_rope_cos(head_dim_idx);
                float s = t_g_rope_sin(head_dim_idx);
                val = apply_rope_interleaved<TileM>(
                    val, c, s, tid_inner
                );
                t_g_q_out(global_row_idx) =
                    static_cast<bfloat16_t>(val);
            }
            else if (global_row_idx < K_End) {
                float c = t_g_rope_cos(head_dim_idx);
                float s = t_g_rope_sin(head_dim_idx);
                val = apply_rope_interleaved<TileM>(
                    val, c, s, tid_inner
                );
                int k_head_idx =
                    (global_row_idx - Q_End) / HeadDim;

                t_g_k_cache(g.pos_id, k_head_idx, head_dim_idx) =
                    static_cast<bfloat16_t>(val);
            }
            else if (global_row_idx < V_End) {
                int v_head_idx =
                    (global_row_idx - K_End) / HeadDim;

                t_g_v_cache(g.pos_id, v_head_idx, head_dim_idx) =
                    static_cast<bfloat16_t>(val);
            }
        }
        __syncthreads();

        // Release-publish after all CTA writers have reached the barrier.
        if (tid_inner == 0) {
             int head_idx = global_tile_idx / TilesPerHead;
             barrier_publish(
                 &t_barriers(inst.layer_idx,
                             BARRIER_ROW(OPCODE_RMS_QKV_GEMV_ROPE),
                             head_idx),
                 1
             );
        }
    }
}
} // namespace megakernel
