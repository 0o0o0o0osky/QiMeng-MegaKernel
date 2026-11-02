// Router GEMV (opcode 8): one output row per scheduled expert.
// Slot 0 publishes completed logits; one designated instruction also stores the
// normalized hidden vector and publishes slot 1 for both expert branches.
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
namespace megakernel {

using namespace cute;

constexpr int ROUTER_EXPECTED_O_PROJ_BLOCKS = QwenMoeConfig::HIDDEN_DIM / 16;

struct RouterGemvSmem {
    alignas(128) bfloat16_t normed[QwenMoeConfig::HIDDEN_DIM];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

__device__ __forceinline__ void router_gemv(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    struct ParsedInstruction {
        int layer_idx;
        int tile_N_start_idx;
        int tile_N_end_idx;
        int write_normed_hidden;
        int tile_N_granularity;
        __device__ __forceinline__ explicit ParsedInstruction(const int32_t* inst) {
            layer_idx        = inst[1];
            tile_N_start_idx = inst[2];
            tile_N_end_idx   = inst[3];
            write_normed_hidden = inst[4];
            tile_N_granularity = inst[5];
        }
    };

    ParsedInstruction inst(smem.instruction);
    int tid       = threadIdx.x;
    constexpr int NumWarps = KernelConfig::NUM_WARPS;
    constexpr int HiddenDim = QwenMoeConfig::HIDDEN_DIM;
    constexpr int NumExperts = QwenMoeConfig::NUM_EXPERTS;

    int start_e = inst.tile_N_start_idx * inst.tile_N_granularity;
    int end_e   = min(inst.tile_N_end_idx * inst.tile_N_granularity, NumExperts);

    auto& op_smem = *reinterpret_cast<RouterGemvSmem*>(smem.main_workspace);

    {
        constexpr int OPROJ_BARRIER_ROW = 3;  // opcode 4 - 1
        auto barriers = g.get_barrier();
        const int32_t* bar = &barriers(inst.layer_idx, OPROJ_BARRIER_ROW, 0);
        barrier_wait(bar, ROUTER_EXPECTED_O_PROJ_BLOCKS);
    }
    __syncthreads();

    auto t_g_hidden = g.get_hidden_state();
    auto t_g_mlp_norm = g.get_mlp_norm_weight()(inst.layer_idx, _);
    auto t_s_norm = make_tensor(
        make_smem_ptr(op_smem.normed),
        make_shape(Int<HiddenDim>{}),
        LayoutRight{}
    );
    auto t_s_reduce = make_tensor(
        make_smem_ptr(op_smem.reduction_workspace),
        make_shape(Int<NumWarps>{}),
        LayoutRight{}
    );
    rms_norm<NumWarps>(t_g_hidden, t_g_mlp_norm, t_s_norm, g.rms_norm_eps, t_s_reduce);
    __syncthreads();

    if (inst.write_normed_hidden) {
        for (int d = tid; d < HiddenDim; d += blockDim.x) {
            g.normed_hidden_states_ptr[d] = op_smem.normed[d];
        }
    }
    __syncthreads();

    float* logits_out = g.router_logits_buf_ptr;  // [num_experts], shared across layers
    const bfloat16_t* router_base = g.get_router_weight().data().get()
        + static_cast<long>(inst.layer_idx) * NumExperts * HiddenDim;

    // The latency schedule assigns one expert per instruction; the loop keeps
    // the serialized instruction valid if a later schedule coarsens adjacent rows.
    for (int e = start_e; e < end_e; ++e) {
        const bfloat16_t* w_row = router_base + static_cast<long>(e) * HiddenDim;
        float partial = 0.0f;
        constexpr int VecElems = 4;  // 4 * bf16 = 8B = uint2
        constexpr int NumVec = HiddenDim / VecElems;
        if (tid < NumVec) {
            const uint2 w_pack = reinterpret_cast<const uint2*>(w_row)[tid];
            const uint2 x_pack = reinterpret_cast<const uint2*>(op_smem.normed)[tid];
            const bfloat16_t* wv = reinterpret_cast<const bfloat16_t*>(&w_pack);
            const bfloat16_t* xv = reinterpret_cast<const bfloat16_t*>(&x_pack);
            #pragma unroll
            for (int i = 0; i < VecElems; ++i) {
                partial += float(wv[i]) * float(xv[i]);
            }
        }
        float dot = block_reduce_sum<NumWarps>(partial, t_s_reduce);
        if (tid == 0) {
            logits_out[e] = dot;
        }
        __syncthreads();
    }

    if (tid == 0 && start_e < end_e) {
        constexpr int ROUTER_GEMV_BARRIER_ROW = 7;  // opcode 8 - 1
        auto barriers = g.get_barrier();
        barrier_publish(
            &barriers(inst.layer_idx, ROUTER_GEMV_BARRIER_ROW, 0),
            end_e - start_e
        );
        if (inst.write_normed_hidden) {
            barrier_publish(
                &barriers(inst.layer_idx, ROUTER_GEMV_BARRIER_ROW, 1), 1
            );
        }
    }
}

}  // namespace megakernel
