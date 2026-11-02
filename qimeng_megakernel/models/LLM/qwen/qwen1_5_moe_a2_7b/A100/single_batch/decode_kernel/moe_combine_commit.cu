// GEMV-owned direct post-op (opcode 14): wait for the shared/routed fan-in,
// combine one-owner hidden tiles, clear routed accumulation, and publish.
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>

namespace megakernel {

using namespace cute;

namespace {
constexpr int CombineTileM = QwenMoeConfig::TILE_N_GRANULARITY;
constexpr int CombineHiddenDim = QwenMoeConfig::HIDDEN_DIM;
constexpr int CombineNumWarps = KernelConfig::NUM_WARPS;
}

struct MoECombineSmem {
    alignas(128) float reduction_workspace[CombineNumWarps];
};

__device__ __forceinline__ void moe_combine_commit(
    QwenGlobals& g,
    SmemLayout& smem
) {
    struct ParsedInstruction {
        int layer_idx;
        int tile_N_start_idx;
        int tile_N_end_idx;
        int tile_N_granularity;

        __device__ __forceinline__ explicit ParsedInstruction(
            const int32_t* inst
        ) {
            layer_idx = inst[1];
            tile_N_start_idx = inst[2];
            tile_N_end_idx = inst[3];
            tile_N_granularity = inst[4];
        }
    };

    ParsedInstruction inst(smem.instruction);
    if (inst.tile_N_start_idx >= inst.tile_N_end_idx) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    auto& s = *reinterpret_cast<MoECombineSmem*>(smem.main_workspace);
    auto t_reduce = make_tensor(
        make_smem_ptr(s.reduction_workspace),
        make_shape(Int<CombineNumWarps>{}),
        LayoutRight{}
    );

    // Independent compute slot: the scalar gate only consumes opcode-8-owned data.
    auto t_normed = g.get_normed_hidden_state();
    auto t_gate_weight = g.get_shared_expert_gate_weight()(inst.layer_idx, _);
    // Match the route-independent reduction shape: each warp owns one
    // contiguous 128-element slice, then warp 0 combines the 16 warp sums in
    // increasing slice order.
    constexpr int GateColsPerWarp = CombineHiddenDim / CombineNumWarps;
    float partial = 0.0f;
    int gate_start = warp_id * GateColsPerWarp;
    for (int d = gate_start + lane_id;
         d < gate_start + GateColsPerWarp;
         d += 32) {
        partial += float(t_gate_weight(d)) * float(t_normed(d));
    }
    partial = warp_reduce_sum(partial);
    if (lane_id == 0) t_reduce(warp_id) = partial;
    __syncthreads();
    if (tid == 0) {
        float gate_logit = 0.0f;
        #pragma unroll
        for (int warp = 0; warp < CombineNumWarps; ++warp) {
            gate_logit = __fadd_rn(gate_logit, t_reduce(warp));
        }
        t_reduce(0) = gate_logit;
    }
    __syncthreads();
    float gate_logit = t_reduce(0);
    float gate_scale = 1.0f / (1.0f + __expf(-gate_logit));

    auto t_barriers = g.get_barrier();
    auto t_hidden = g.get_hidden_state();
    auto t_shared = g.get_shared_out();
    auto t_routed_f32 = g.get_routed_out_f32();
    auto t_routed_bf16 = g.get_routed_out();

    for (int tile = inst.tile_N_start_idx; tile < inst.tile_N_end_idx; ++tile) {
        {
            barrier_wait(
                &t_barriers(inst.layer_idx, OPCODE_SHARED_DOWNPROJ - 1, tile),
                QwenMoeConfig::NUM_DOWNPROJ_SPLIT_K_SHARED
            );
            barrier_wait(
                &t_barriers(inst.layer_idx, OPCODE_ROUTED_DOWNPROJ - 1, tile),
                QwenMoeConfig::NUM_EXPERTS_PER_TOK
            );
        }
        __syncthreads();

        if (tid < CombineTileM) {
            int row = tile * inst.tile_N_granularity + tid;
            if (row < CombineHiddenDim) {
                float value = __fadd_rn(
                    __fadd_rn(float(t_hidden(row)), t_routed_f32(row)),
                    __fmul_rn(gate_scale, float(t_shared(row)))
                );
                t_hidden(row) = bfloat16_t(value);
                t_routed_f32(row) = 0.0f;
                t_routed_bf16(row) = bfloat16_t(0.0f);
            }
        }
        __syncthreads();

        if (tid == 0) {
            barrier_publish(
                &t_barriers(inst.layer_idx, OPCODE_MOE_COMBINE_COMMIT - 1, 0), 1
            );
        }
        __syncthreads();
    }
}

}  // namespace megakernel
