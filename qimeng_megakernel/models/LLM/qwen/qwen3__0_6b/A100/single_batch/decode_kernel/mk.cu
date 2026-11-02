// Qwen3-0.6B A100 decode MegaKernel entrypoint.

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "globals.cuh"
#include "instruction_list.cuh"
#include "kernel_config.cuh"
#include "rms_qkv_gemv.cu"
#include "decode_attention_splitkv.cu"
#include "decode_attention_reduction.cu"
#include "oproj_gemv_residual.cu"
#include "rms_upgate_gemv_silu.cu"
#include "down_proj_gemv_residual.cu"
#include "rms_lm_head.cu"

#include "mk.h"

#ifndef NDEBUG
#include <cstdio>
#endif

using namespace megakernel;
using namespace cute;

__global__ __launch_bounds__(KernelConfig::NUM_THREADS, 1)
void mk_kernel(QwenGlobals g) {
    int sm_id = blockIdx.x;
    int tid   = threadIdx.x;

    extern __shared__ uint8_t raw_smem[];
    SmemLayout& smem = *reinterpret_cast<SmemLayout*>(raw_smem);

    int32_t* inst_base = g.instruction_ptr +
        sm_id * g.num_insts * KernelConfig::INSTRUCTION_WIDTH;
    auto g_insts_sm = make_tensor(
        make_gmem_ptr(inst_base),
        make_shape(g.num_insts, Int<KernelConfig::INSTRUCTION_WIDTH>{}),
        LayoutRight{}
    );

    for (int i = 0; i < g.num_insts; ++i) {
        if (tid < KernelConfig::INSTRUCTION_WIDTH) {
            smem.instruction[tid] = g_insts_sm(i, tid);
        }
        __syncthreads();

        int opcode = smem.instruction[0];

        switch (opcode) {
            case 0:
                // Scheduler padding.
                break;
            case OPCODE_RMS_QKV_GEMV:
                rms_qkv_gemv(g, smem);
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
            case OPCODE_RMS_UPGATE_SILU:
                rms_upgate_gemv_silu(g, smem);
                break;
            case OPCODE_DOWN_PROJ_RESIDUAL:
                down_proj_gemv_residual(g, smem);
                break;
            case OPCODE_RMS_LM_HEAD:
                rms_lmhead_gemv(g, smem);
                break;
            default:
#ifndef NDEBUG
                if (tid == 0) {
                    printf("[mk] unsupported opcode=%d on sm=%d inst=%d\n",
                           opcode, sm_id, i);
                }
#endif
                __trap();
                break;
        }
        __syncthreads();
    }
}

void mk(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
    const torch::Tensor& qkv_raw,
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
    const torch::Tensor& q_norm_weights,
    const torch::Tensor& k_norm_weights,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin,
    const torch::Tensor& o_proj_weights,
    const torch::Tensor& mlp_norm_weights,
    const torch::Tensor& up_proj_weights,
    const torch::Tensor& gate_proj_weights,
    const torch::Tensor& down_proj_weights,
    const torch::Tensor& lm_head_norm_weights,
    const torch::Tensor& lm_head_weights,
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
    g.qkv_raw_ptr      = reinterpret_cast<bfloat16_t*>(qkv_raw.data_ptr<torch::BFloat16>());
    g.q_post_rope_ptr  = reinterpret_cast<bfloat16_t*>(q_post_rope.data_ptr<torch::BFloat16>());
    g.attn_out_ptr     = reinterpret_cast<bfloat16_t*>(attn_out.data_ptr<torch::BFloat16>());

    g.attn_lse_ptr       = attn_lse_intermediates.data_ptr<float>();
    g.attn_out_inter_ptr = attn_out_intermediates.data_ptr<float>();

    g.silu_out_ptr = reinterpret_cast<bfloat16_t*>(silu_out.data_ptr<torch::BFloat16>());

    g.k_cache_ptr = reinterpret_cast<bfloat16_t*>(k_cache.data_ptr<torch::BFloat16>());
    g.v_cache_ptr = reinterpret_cast<bfloat16_t*>(v_cache.data_ptr<torch::BFloat16>());

    g.attn_norm_ptr  = reinterpret_cast<bfloat16_t*>(attn_norm_weights.data_ptr<torch::BFloat16>());
    g.qkv_weight_ptr = reinterpret_cast<bfloat16_t*>(qkv_weights.data_ptr<torch::BFloat16>());
    // Qwen3: q_norm and k_norm instead of qkv_bias
    g.q_norm_ptr     = reinterpret_cast<bfloat16_t*>(q_norm_weights.data_ptr<torch::BFloat16>());
    g.k_norm_ptr     = reinterpret_cast<bfloat16_t*>(k_norm_weights.data_ptr<torch::BFloat16>());

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
