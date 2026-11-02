// Qwen1.5-MoE-A2.7B A100 decode MegaKernel entrypoint (op 1-4, 7, 8-14).

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "globals.cuh"
#include "instruction_list.cuh"
#include "kernel_config.cuh"
#include "rms_qkv_gemv_rope.cu"
#include "decode_attention_splitkv.cu"
#include "decode_attention_reduction.cu"
#include "oproj_gemv_residual.cu"
#include "router_gemv.cu"
#include "router_softmax_topk.cu"
#include "shared_expert_upgate_silu.cu"
#include "shared_expert_downproj.cu"
#include "routed_expert_upgate_silu.cu"
#include "routed_expert_downproj.cu"
#include "moe_combine_commit.cu"
#include "rms_lmhead_gemv.cu"

#include "mk.h"

using namespace megakernel;
using namespace cute;

namespace {
constexpr int INSTRUCTION_WIDTH = 32;
}

__global__ __launch_bounds__(512, 1) void mk_kernel(QwenGlobals g) {
    int sm_id = blockIdx.x;
    int tid   = threadIdx.x;

    extern __shared__ uint8_t raw_smem[];
    SmemLayout& smem = *reinterpret_cast<SmemLayout*>(raw_smem);

    int32_t* inst_base = g.instruction_ptr + sm_id * g.num_insts * INSTRUCTION_WIDTH;
    auto g_insts_sm = make_tensor(
        make_gmem_ptr(inst_base),
        make_shape(g.num_insts, Int<INSTRUCTION_WIDTH>{}),
        LayoutRight{}
    );

    for (int i = 0; i < g.num_insts; ++i) {
        if (tid < INSTRUCTION_WIDTH) {
            smem.instruction[tid] = g_insts_sm(i, tid);
        }
        __syncthreads();

        int opcode = smem.instruction[0];

        switch (opcode) {
            case OPCODE_RMS_QKV_GEMV_ROPE:
                rms_qkv_gemv_rope(g, smem);
                break;
            case OPCODE_ATTENTION_PARTIAL:
                decode_attention_splitkv(g, smem);
                break;
            case OPCODE_ATTENTION_REDUCTION:
                decode_attention_reduction(g, smem);
                break;
            case OPCODE_O_PROJ_RESIDUAL:
                oproj_gemv_residual(g, smem);
                break;
            case OPCODE_ROUTER_GEMV:
                router_gemv(g, smem);
                break;
            case OPCODE_ROUTER_SOFTMAX_TOPK:
                router_softmax_topk(g, smem);
                break;
            case OPCODE_SHARED_UPGATE_SILU:
                shared_expert_upgate_silu(g, smem);
                break;
            case OPCODE_SHARED_DOWNPROJ:
                shared_expert_downproj(g, smem);
                break;
            case OPCODE_ROUTED_UPGATE_SILU:
                routed_expert_upgate_silu(g, smem);
                break;
            case OPCODE_ROUTED_DOWNPROJ:
                routed_expert_downproj(g, smem);
                break;
            case OPCODE_MOE_COMBINE_COMMIT:
                moe_combine_commit(g, smem);
                break;
            case OPCODE_RMS_LMHEAD_GEMV:
                rms_lmhead_gemv(g, smem);
                break;
            default:
                break;
        }
        __syncthreads();
    }
}

void mk(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
    const torch::Tensor& normed_hidden_states,
    const torch::Tensor& q_post_rope,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& attn_out,
    const torch::Tensor& attn_lse_intermediates,
    const torch::Tensor& attn_out_intermediates,
    const torch::Tensor& silu_out,
    const torch::Tensor& logits,
    const torch::Tensor& attn_norm_weights,
    const torch::Tensor& qkv_weights,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin,
    const torch::Tensor& o_proj_weights,
    const torch::Tensor& mlp_norm_weights,
    const torch::Tensor& up_proj_weights,
    const torch::Tensor& gate_proj_weights,
    const torch::Tensor& down_proj_weights,
    const torch::Tensor& lm_head_norm_weights,
    const torch::Tensor& lm_head_weights,
    const torch::Tensor& router_weights,
    const torch::Tensor& router_logits_buf,
    const torch::Tensor& expert_indices,
    const torch::Tensor& expert_weights,
    const torch::Tensor& shared_expert_gate_proj_weights,
    const torch::Tensor& shared_expert_up_proj_weights,
    const torch::Tensor& shared_expert_down_proj_weights,
    const torch::Tensor& shared_expert_silu_out,
    const torch::Tensor& shared_out,
    const torch::Tensor& shared_expert_gate_weights,
    const torch::Tensor& expert_gate_up_proj_weights,
    const torch::Tensor& expert_down_proj_weights,
    const torch::Tensor& routed_silu_buffer,
    const torch::Tensor& routed_out_f32,
    const torch::Tensor& routed_out,
    const torch::Tensor& barriers,
    int pos_id,
    float attn_scale,
    float rms_norm_eps
) {
    QwenGlobals g{};

    g.instruction_ptr = instructions.data_ptr<int32_t>();
    g.num_insts       = static_cast<int32_t>(instructions.size(1));

    g.pos_id           = static_cast<uint32_t>(pos_id);
    g.hidden_state_ptr = reinterpret_cast<bfloat16_t*>(hidden_states.data_ptr<torch::BFloat16>());
    g.normed_hidden_states_ptr = (normed_hidden_states.defined() && normed_hidden_states.numel() > 0)
        ? reinterpret_cast<bfloat16_t*>(normed_hidden_states.data_ptr<torch::BFloat16>())
        : nullptr;
    g.q_post_rope_ptr  = reinterpret_cast<bfloat16_t*>(q_post_rope.data_ptr<torch::BFloat16>());
    g.attn_out_ptr     = reinterpret_cast<bfloat16_t*>(attn_out.data_ptr<torch::BFloat16>());

    g.attn_lse_ptr       = attn_lse_intermediates.data_ptr<float>();
    g.attn_out_inter_ptr = attn_out_intermediates.data_ptr<float>();

    g.silu_out_ptr = reinterpret_cast<bfloat16_t*>(silu_out.data_ptr<torch::BFloat16>());

    g.k_cache_ptr = reinterpret_cast<bfloat16_t*>(k_cache.data_ptr<torch::BFloat16>());
    g.v_cache_ptr = reinterpret_cast<bfloat16_t*>(v_cache.data_ptr<torch::BFloat16>());

    g.attn_norm_ptr  = reinterpret_cast<bfloat16_t*>(attn_norm_weights.data_ptr<torch::BFloat16>());
    g.qkv_weight_ptr = reinterpret_cast<bfloat16_t*>(qkv_weights.data_ptr<torch::BFloat16>());
    g.qkv_bias_ptr   = reinterpret_cast<bfloat16_t*>(qkv_bias.data_ptr<torch::BFloat16>());

    g.rope_cos_ptr = rope_cos.data_ptr<float>();
    g.rope_sin_ptr = rope_sin.data_ptr<float>();
    g.max_seq_len  = static_cast<int32_t>(k_cache.size(2));

    g.o_weight_ptr    = reinterpret_cast<bfloat16_t*>(o_proj_weights.data_ptr<torch::BFloat16>());
    g.mlp_norm_ptr    = reinterpret_cast<bfloat16_t*>(mlp_norm_weights.data_ptr<torch::BFloat16>());
    g.up_weight_ptr   = reinterpret_cast<bfloat16_t*>(up_proj_weights.data_ptr<torch::BFloat16>());
    g.gate_weight_ptr = reinterpret_cast<bfloat16_t*>(gate_proj_weights.data_ptr<torch::BFloat16>());
    g.down_weight_ptr = reinterpret_cast<bfloat16_t*>(down_proj_weights.data_ptr<torch::BFloat16>());

    g.lm_head_norm_ptr   = reinterpret_cast<bfloat16_t*>(lm_head_norm_weights.data_ptr<torch::BFloat16>());
    g.lm_head_weight_ptr = reinterpret_cast<bfloat16_t*>(lm_head_weights.data_ptr<torch::BFloat16>());
    g.logits_ptr         = reinterpret_cast<bfloat16_t*>(logits.data_ptr<torch::BFloat16>());

    g.router_weight_ptr     = reinterpret_cast<bfloat16_t*>(router_weights.data_ptr<torch::BFloat16>());
    g.router_logits_buf_ptr = router_logits_buf.data_ptr<float>();
    g.expert_indices_ptr    = expert_indices.data_ptr<int64_t>();
    g.expert_weights_ptr = expert_weights.data_ptr<float>();

    g.shared_expert_gate_proj_ptr  = reinterpret_cast<bfloat16_t*>(shared_expert_gate_proj_weights.data_ptr<torch::BFloat16>());
    g.shared_expert_up_proj_ptr    = reinterpret_cast<bfloat16_t*>(shared_expert_up_proj_weights.data_ptr<torch::BFloat16>());
    g.shared_expert_down_proj_ptr  = reinterpret_cast<bfloat16_t*>(shared_expert_down_proj_weights.data_ptr<torch::BFloat16>());
    g.shared_expert_silu_out_ptr       = reinterpret_cast<bfloat16_t*>(shared_expert_silu_out.data_ptr<torch::BFloat16>());
    g.shared_out_ptr                   = reinterpret_cast<bfloat16_t*>(shared_out.data_ptr<torch::BFloat16>());
    g.shared_expert_gate_weight_ptr    = reinterpret_cast<bfloat16_t*>(shared_expert_gate_weights.data_ptr<torch::BFloat16>());

    g.expert_gate_up_proj_ptr  = reinterpret_cast<bfloat16_t*>(expert_gate_up_proj_weights.data_ptr<torch::BFloat16>());
    g.expert_down_proj_ptr     = reinterpret_cast<bfloat16_t*>(expert_down_proj_weights.data_ptr<torch::BFloat16>());
    g.routed_silu_buffer_ptr   = reinterpret_cast<bfloat16_t*>(routed_silu_buffer.data_ptr<torch::BFloat16>());
    g.routed_out_f32_ptr       = routed_out_f32.data_ptr<float>();
    g.routed_out_ptr           = reinterpret_cast<bfloat16_t*>(routed_out.data_ptr<torch::BFloat16>());

    g.barrier_ptr        = barriers.data_ptr<int32_t>();
    g.num_layer_barriers = static_cast<int32_t>(barriers.size(1));

    g.attn_scale   = attn_scale;
    g.rms_norm_eps = rms_norm_eps;
    g.max_splits   = static_cast<int32_t>(attn_lse_intermediates.size(1));

    size_t smem_size = sizeof(SmemLayout);

    cudaError_t err_attr = cudaFuncSetAttribute(
        mk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)
    );
    if (err_attr != cudaSuccess) {
        return;
    }

    int grid_size = static_cast<int>(instructions.size(0));
    mk_kernel<<<grid_size, KernelConfig::NUM_THREADS, smem_size>>>(g);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return;
    }
}
