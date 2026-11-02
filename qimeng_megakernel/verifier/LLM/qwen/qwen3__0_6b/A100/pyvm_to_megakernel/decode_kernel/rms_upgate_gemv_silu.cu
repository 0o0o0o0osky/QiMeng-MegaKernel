#include "globals.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int kUpGateOutputTileN = KernelConfig::NUM_WARPS;
constexpr int kUpGateInputTileK =
    DecodeScheduleConfig::HIDDEN_GEMV_K_SEGMENT;
constexpr int kUpGateKSegments =
    QwenConfig::HIDDEN_DIM / kUpGateInputTileK;
using UpGateDownProjBuckets = SplitKBucketContract<
    QwenConfig::INTERM_DIM,
    kUpGateOutputTileN,
    DecodeScheduleConfig::DOWN_PROJ_SPLIT_K_GRANULARITY>;

// Shared memory layout for RMS + UpGate GEMV + SiLU operation
struct RMSUpGateSiLUSmem {
    alignas(128) bfloat16_t activation[QwenConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t
        weight_stage_0[kUpGateOutputTileN * kUpGateInputTileK];
    alignas(128) bfloat16_t
        weight_stage_1[kUpGateOutputTileN * kUpGateInputTileK];
    alignas(128) float up_output[kUpGateOutputTileN];
    alignas(128) float gate_output[kUpGateOutputTileN];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

static_assert(sizeof(RMSUpGateSiLUSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "UpGate workspace exceeds the selected Ampere CTA budget");

__device__ __forceinline__ void rms_upgate_gemv_silu(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    constexpr int TileM = kUpGateOutputTileN;
    constexpr int TileK = kUpGateInputTileK;
    constexpr int HiddenDim = QwenConfig::HIDDEN_DIM;
    constexpr int IntermDim = QwenConfig::INTERM_DIM;
    constexpr int NumWarps = KernelConfig::NUM_WARPS;
    static_assert(TileM == UpGateDownProjBuckets::PRODUCER_TILE_N);
    static_assert(TileK > 0);
    static_assert(HiddenDim % TileK == 0,
                  "UpGate hidden GEMV segment must exactly tile HIDDEN_DIM");

    // 1. Setup shared memory layout
    RMSUpGateSiLUSmem& s = *reinterpret_cast<RMSUpGateSiLUSmem*>(smem.main_workspace);

    // 2. Instruction Parsing
    struct ParsedInstruction {
        int layer_idx;
        int tile_M_idx;
        int num_blocks;
        const int32_t* block_idxs;

        __device__ __forceinline__ ParsedInstruction(int32_t* smem_inst) {
            // Serialized format from instruction_defs.py RMS_UpGateProj_GEMV_SiLU:
            // [opcode, layer_idx, tile_M_idx, num_blocks, block_idx_0, block_idx_1, ...]
            layer_idx = smem_inst[1];
            tile_M_idx = smem_inst[2];
            num_blocks = smem_inst[3];
            block_idxs = smem_inst + 4;
        }
    };

    ParsedInstruction inst(smem.instruction);

    if (inst.num_blocks == 0) return;

    int tid = threadIdx.x;

    // Weight prefetch before barrier (overlap with OProj spin).
    auto t_g_up = g.get_up_weight()(inst.layer_idx, _, _);

    // Each output block executes all Up segments followed by all Gate segments.
    int total_iters = inst.num_blocks * 2 * kUpGateKSegments;

    if (total_iters > 0) {
        int block_idx = inst.block_idxs[0];
        for (int row = 0; row < TileM; ++row) {
            const bfloat16_t* row_g = t_g_up.data().get()
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

    // Barrier Wait (wait for this layer's OProj GEMV residual).
    if (tid == 0) {
        int wait_opcode =
            BARRIER_ROW(OPCODE_O_PROJ_RESIDUAL);
        auto barriers = g.get_barrier();
        int32_t* bar =
            &barriers(inst.layer_idx, wait_opcode, 0);
        barrier_wait(bar, (QwenConfig::HIDDEN_DIM / TileM));
    }
    __syncthreads();

    // RMS Norm (one lane group per configured warp).

    // Input (Global)
    auto t_g_input = g.get_hidden_state();

    // Weight (Global): one layer
    auto t_g_rms_w = g.get_mlp_norm_weight()(inst.layer_idx, _);

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

    // Pipeline Loop (Up & Gate GEMV).

    auto t_g_gate = g.get_gate_weight()(inst.layer_idx, _, _);

    for (int i = 0; i < total_iters; ++i) {
        int projection_span = kUpGateKSegments;
        int block_span = 2 * projection_span;
        int curr_pair_idx = i / block_span;
        int within_block = i % block_span;
        bool is_gate_step = within_block >= projection_span;
        int segment_idx = within_block % projection_span;
        int curr_block_idx = inst.block_idxs[curr_pair_idx];
        int stage_curr = i % 2;
        int stage_next = (i + 1) % 2;

        // --- Phase 1: Prefetch Next ---
        if (i + 1 < total_iters) {
            int next_iter = i + 1;
            int next_pair_idx = next_iter / block_span;
            int next_within_block = next_iter % block_span;
            bool next_is_gate = next_within_block >= projection_span;
            int next_segment_idx = next_within_block % projection_span;
            int next_block_idx = inst.block_idxs[next_pair_idx];
            auto next_weights = next_is_gate ? t_g_gate : t_g_up;
            bfloat16_t* next_stage =
                (stage_next == 0) ? s.weight_stage_0 : s.weight_stage_1;
            for (int row = 0; row < TileM; ++row) {
                const bfloat16_t* row_g = next_weights.data().get()
                    + static_cast<long>(next_block_idx) * TileM * HiddenDim
                    + row * HiddenDim
                    + next_segment_idx * TileK;
                bfloat16_t* row_s = next_stage + row * TileK;
                auto t_g_row = make_tensor(
                    make_gmem_ptr(row_g), make_shape(Int<TileK>{}), LayoutRight{});
                auto t_s_row = make_tensor(
                    make_smem_ptr(row_s), make_shape(Int<TileK>{}), LayoutRight{});
                load_tile_async(t_g_row, t_s_row);
            }
            cp_async_fence();
        }

        // --- Phase 2: Wait ---
        if (i + 1 < total_iters) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // --- Phase 3: Compute paired GEMV using the configured warp profile ---
        bfloat16_t* current_stage =
            (stage_curr == 0) ? s.weight_stage_0 : s.weight_stage_1;
        auto t_s_weight = make_tensor(
            make_smem_ptr(current_stage),
            make_shape(Int<TileM>{}, Int<TileK>{}),
            LayoutRight{}
        );
        auto t_s_input_segment = make_tensor(
            make_smem_ptr(s.activation + segment_idx * TileK),
            make_shape(Int<TileK>{}),
            LayoutRight{}
        );
        auto t_s_output = make_tensor(
            make_smem_ptr(is_gate_step ? s.gate_output : s.up_output),
            make_shape(Int<TileM>{}),
            LayoutRight{}
        );
        if (segment_idx == 0) {
            compute_gemv_tile<NumWarps, TileM, TileK>(
                t_s_input_segment, t_s_weight, t_reduce_workspace, t_s_output);
        } else {
            compute_gemv_tile<NumWarps, TileM, TileK, 1, true>(
                t_s_input_segment, t_s_weight, t_reduce_workspace, t_s_output);
        }
        __syncthreads();

        // Phase 4: Epilogue (Gate step only: SiLU + Write back).
        if (is_gate_step && segment_idx + 1 == kUpGateKSegments) {

            // One writer per output row in this tile.
            if (tid < TileM) {
                float up_val = s.up_output[tid];
                float gate_val = s.gate_output[tid];

                // SiLU: x / (1 + exp(-x))
                float silu_gate = gate_val / (1.0f + __expf(-gate_val));
                float final_val = silu_gate * up_val;

                // Write back to Global Memory
                // Target: ptr_silu_out [IntermDim]
                // Offset: block_idx * TileM + tid
                int global_idx = curr_block_idx * TileM + tid;
                if (global_idx < IntermDim) {
                    g.get_silu_out()(global_idx) = bfloat16_t(final_val);
                }
            }

            // Publish the split-K bucket consumed by the matching DownProj.
            __syncthreads();
            if (tid == 0) {
                int signal_bucket =
                    UpGateDownProjBuckets::producer_bucket(curr_block_idx);
                auto barriers = g.get_barrier();
                barrier_publish(
                    &barriers(
                        inst.layer_idx,
                        BARRIER_ROW(OPCODE_RMS_UPGATE_SILU),
                        signal_bucket),
                    1);
            }
        }
    }
}

} // namespace megakernel
