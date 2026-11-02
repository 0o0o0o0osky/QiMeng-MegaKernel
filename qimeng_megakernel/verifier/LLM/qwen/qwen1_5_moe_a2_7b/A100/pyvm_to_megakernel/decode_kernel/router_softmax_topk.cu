// Qwen1.5-MoE A100: router softmax + topk (opcode 9). Waits for all Router_GEMV (op8), then
// softmax(router_logits_buf) → topk(4) → expert_indices/weights. Sets barrier row 8 = 1.
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>

namespace megakernel {

using namespace cute;

constexpr int ROUTER_NUM_EXPERT_BLOCKS = 60;  // fixed 60 router workers

struct RouterSoftmaxTopKSmem {
    alignas(128) float logits[QwenMoeConfig::NUM_EXPERTS];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

__device__ __forceinline__ void router_softmax_topk(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    struct ParsedInstruction {
        int layer_idx;
        __device__ __forceinline__ explicit ParsedInstruction(const int32_t* inst) {
            layer_idx = inst[1];
        }
    };

    ParsedInstruction inst(smem.instruction);
    int tid       = threadIdx.x;
    int warp_id   = tid / 32;
    int lane_id   = tid % 32;
    constexpr int NumWarps = KernelConfig::NUM_WARPS;
    constexpr int NumExperts = QwenMoeConfig::NUM_EXPERTS;
    constexpr int TopK = QwenMoeConfig::NUM_EXPERTS_PER_TOK;

    auto& op_smem = *reinterpret_cast<RouterSoftmaxTopKSmem*>(smem.main_workspace);

    {
        constexpr int ROUTER_GEMV_BARRIER_ROW = 7;  // opcode 8 - 1
        auto barriers = g.get_barrier();
        const int32_t* bar = &barriers(inst.layer_idx, ROUTER_GEMV_BARRIER_ROW, 0);
        barrier_wait(bar, ROUTER_NUM_EXPERT_BLOCKS);
    }
    __syncthreads();

    float* logits_in = g.router_logits_buf_ptr;
    auto t_s_logits = make_tensor(
        make_smem_ptr(op_smem.logits),
        make_shape(Int<NumExperts>{}),
        LayoutRight{}
    );
    auto t_s_reduce = make_tensor(
        make_smem_ptr(op_smem.reduction_workspace),
        make_shape(Int<NumWarps>{}),
        LayoutRight{}
    );

    for (int i = tid; i < NumExperts; i += NumWarps * 32) {
        t_s_logits(i) = logits_in[i];
    }
    __syncthreads();

    float max_logit = -1e30f;
    if (tid < NumExperts) {
        max_logit = op_smem.logits[tid];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float v = __shfl_xor_sync(0xffffffff, max_logit, offset);
        max_logit = fmaxf(max_logit, v);
    }
    if (lane_id == 0) {
        op_smem.reduction_workspace[warp_id] = max_logit;
    }
    __syncthreads();
    if (tid < NumWarps) {
        max_logit = op_smem.reduction_workspace[tid];
    } else {
        max_logit = -1e30f;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float v = __shfl_xor_sync(0xffffffff, max_logit, offset);
        max_logit = fmaxf(max_logit, v);
    }
    if (tid == 0) {
        op_smem.reduction_workspace[0] = max_logit;
    }
    __syncthreads();
    max_logit = op_smem.reduction_workspace[0];

    float sum_exp = 0.0f;
    if (tid < NumExperts) {
        float v = expf(op_smem.logits[tid] - max_logit);
        op_smem.logits[tid] = v;
        sum_exp = v;
    }
    sum_exp = block_reduce_sum<NumWarps>(sum_exp, t_s_reduce);
    if (tid == 0) {
        op_smem.reduction_workspace[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / op_smem.reduction_workspace[0];

    if (tid < NumExperts) {
        op_smem.logits[tid] *= inv_sum;
    }
    __syncthreads();

    auto t_expert_indices = g.get_expert_indices();
    auto t_expert_weights = g.get_expert_weights();

    if (warp_id == 0) {
        #pragma unroll
        for (int k = 0; k < TopK; ++k) {
            float best_val = -1.0f;
            int best_idx = 0;
            for (int i = lane_id; i < NumExperts; i += 32) {
                float v = op_smem.logits[i];
                if (v > best_val) {
                    best_val = v;
                    best_idx = i;
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float v = __shfl_xor_sync(0xffffffff, best_val, offset);
                int vi = __shfl_xor_sync(0xffffffff, best_idx, offset);
                if (v > best_val) {
                    best_val = v;
                    best_idx = vi;
                }
            }
            if (lane_id == 0) {
                t_expert_indices(k) = static_cast<int64_t>(best_idx);
                t_expert_weights(k) = best_val;
                op_smem.logits[best_idx] = -1.0f;
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        constexpr int ROUTER_SOFTMAX_BARRIER_ROW = 8;  // opcode 9 - 1
        auto barriers = g.get_barrier();
        barrier_publish(
            &barriers(inst.layer_idx, ROUTER_SOFTMAX_BARRIER_ROW, 0), 1
        );
    }
}

}  // namespace megakernel
