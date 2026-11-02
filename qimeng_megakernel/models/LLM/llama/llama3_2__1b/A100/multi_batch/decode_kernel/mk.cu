// Multi-batch decode MegaKernel entrypoint.

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <ATen/cuda/CUDAContext.h>

#include "globals.cuh"
#include "instruction_list.cuh"
#include "kernel_config.cuh"
#include "profiling.cuh"
#include "decode_attn_rms.cu"
#include "decode_mlp_rms.cu"
#include "decode_lmhead_rms.cu"
#include "qkv_gemm_rope.cu"
// Active splitkv path is CuTe-only; do not switch this include to attn_* PTX PV.
#include "decode_attention_splitkv.cu"
#include "decode_attention_reduction.cu"
#include "decode_oproj_gemm_residual.cu"
#include "decode_upgate_gemm_silu.cu"
#include "decode_downproj_gemm_residual.cu"
#include "decode_lmhead_gemm.cu"

#include "mk.h"

#include <optional>

using namespace megakernel;
using namespace cute;

// Instruction word count (must be visible to __global__ kernels below).
constexpr int kMkInstructionWidth = 32;

__global__ __launch_bounds__(
    KernelConfig::NUM_THREADS,
    KernelConfig::TARGET_CTAS_PER_SM)
void mk_kernel(LlamaGlobals g);
__global__ __launch_bounds__(
    KernelConfig::NUM_THREADS,
    KernelConfig::TARGET_CTAS_PER_SM)
void mk_step_kernel(LlamaGlobals g, int inst_idx);

namespace {

template <typename Kernel>
void validate_residency(Kernel kernel, size_t smem_size, const char* kernel_name) {
    int active_ctas = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_ctas,
        kernel,
        KernelConfig::NUM_THREADS,
        smem_size);
    TORCH_CHECK(
        err == cudaSuccess,
        kernel_name,
        " occupancy query failed: ",
        cudaGetErrorString(err));
    TORCH_CHECK(
        active_ctas >= KernelConfig::TARGET_CTAS_PER_SM,
        kernel_name,
        " supports only ",
        active_ctas,
        " resident CTA(s)/SM; requested ",
        KernelConfig::TARGET_CTAS_PER_SM);
}

static void populate_llama_globals(
    LlamaGlobals& g,
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    const torch::Tensor* prof_buf,
    int pos_id,
    int batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out)
{
    g.instruction_ptr   = instructions.data_ptr<int32_t>();
    g.num_insts         = static_cast<int32_t>(instructions.size(1));

    g.batch_size        = batch_size;
    g.pos_id            = static_cast<uint32_t>(pos_id);
    g.position_ids_ptr  = nullptr;
    g.active_mask_ptr   = nullptr;
    g.attn_partial_valid_ptr = nullptr;

    if (position_ids.has_value() && position_ids->defined()) {
        TORCH_CHECK(position_ids->device().is_cuda(), "position_ids must be a CUDA tensor");
        TORCH_CHECK(position_ids->is_contiguous(), "position_ids must be contiguous");
        TORCH_CHECK(position_ids->dim() == 1, "position_ids must be 1-D");
        TORCH_CHECK(position_ids->size(0) == batch_size, "position_ids numel must equal batch_size");
        TORCH_CHECK(position_ids->scalar_type() == torch::kInt32, "position_ids must be int32");
        g.position_ids_ptr = position_ids->data_ptr<int32_t>();
    }

    if (active_mask.has_value() && active_mask->defined()) {
        TORCH_CHECK(active_mask->device().is_cuda(), "active_mask must be a CUDA tensor");
        TORCH_CHECK(active_mask->is_contiguous(), "active_mask must be contiguous");
        TORCH_CHECK(active_mask->dim() == 1, "active_mask must be 1-D");
        TORCH_CHECK(active_mask->size(0) == batch_size, "active_mask numel must equal batch_size");
        auto st = active_mask->scalar_type();
        if (st == torch::kUInt8) {
            g.active_mask_ptr = active_mask->data_ptr<uint8_t>();
        } else if (st == torch::kBool) {
            g.active_mask_ptr = reinterpret_cast<uint8_t*>(active_mask->data_ptr<bool>());
        } else {
            TORCH_CHECK(false, "active_mask must be torch.uint8 or torch.bool");
        }
    }

    if (attn_partial_valid.has_value() && attn_partial_valid->defined()) {
        TORCH_CHECK(attn_partial_valid->device().is_cuda(), "attn_partial_valid must be a CUDA tensor");
        TORCH_CHECK(attn_partial_valid->is_contiguous(), "attn_partial_valid must be contiguous");
        TORCH_CHECK(attn_partial_valid->dim() == 3, "attn_partial_valid must be 3-D");
        TORCH_CHECK(attn_partial_valid->size(0) == batch_size, "attn_partial_valid batch dim must equal batch_size");
        TORCH_CHECK(attn_partial_valid->size(1) == LlamaConfig::NUM_Q_HEADS, "attn_partial_valid head dim must equal NUM_Q_HEADS");
        const int32_t expected_splits = (attn_lse_intermediates.dim() >= 3)
            ? static_cast<int32_t>(attn_lse_intermediates.size(2))
            : static_cast<int32_t>(attn_lse_intermediates.size(1));
        TORCH_CHECK(attn_partial_valid->size(2) == expected_splits, "attn_partial_valid split dim must equal attn_lse_intermediates.size(-1)");
        auto st = attn_partial_valid->scalar_type();
        if (st == torch::kUInt8) {
            g.attn_partial_valid_ptr = attn_partial_valid->data_ptr<uint8_t>();
        } else if (st == torch::kBool) {
            g.attn_partial_valid_ptr = reinterpret_cast<uint8_t*>(attn_partial_valid->data_ptr<bool>());
        } else {
            TORCH_CHECK(false, "attn_partial_valid must be torch.uint8 or torch.bool");
        }
    }

    g.hidden_state_ptr  = reinterpret_cast<bfloat16_t*>(hidden_states.data_ptr<torch::BFloat16>());
    g.q_post_rope_ptr   = reinterpret_cast<bfloat16_t*>(q_post_rope.data_ptr<torch::BFloat16>());
    g.attn_out_ptr      = reinterpret_cast<bfloat16_t*>(attn_out.data_ptr<torch::BFloat16>());
    g.silu_out_ptr      = reinterpret_cast<bfloat16_t*>(silu_out.data_ptr<torch::BFloat16>());
    g.logits_ptr        = reinterpret_cast<bfloat16_t*>(logits.data_ptr<torch::BFloat16>());

    g.attn_lse_ptr       = attn_lse_intermediates.data_ptr<float>();
    g.attn_out_inter_ptr = attn_out_intermediates.data_ptr<float>();

    g.k_cache_ptr = reinterpret_cast<bfloat16_t*>(k_cache.data_ptr<torch::BFloat16>());
    g.v_cache_ptr = reinterpret_cast<bfloat16_t*>(v_cache.data_ptr<torch::BFloat16>());

    g.attn_norm_ptr  = reinterpret_cast<bfloat16_t*>(attn_norm_weights.data_ptr<torch::BFloat16>());
    g.qkv_weight_ptr = reinterpret_cast<bfloat16_t*>(qkv_weights.data_ptr<torch::BFloat16>());

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

    g.barrier_ptr        = barriers.data_ptr<int32_t>();
    g.num_layer_barriers = static_cast<int32_t>(barriers.size(1));

    g.attn_scale   = attn_scale;
    g.rms_norm_eps = rms_norm_eps;
    g.max_splits = (attn_lse_intermediates.dim() >= 3)
        ? static_cast<int32_t>(attn_lse_intermediates.size(2))
        : static_cast<int32_t>(attn_lse_intermediates.size(1));

    {
        auto last_dim = static_cast<int32_t>(barriers.size(2));
        g.barrier_stride = (batch_size > 0) ? (last_dim / batch_size) : last_dim;
    }

    const torch::Tensor& lm_head_rms_scratch =
        lm_head_rms_out.defined() ? lm_head_rms_out : mlp_rms_out;

    TORCH_CHECK(attn_rms_out.defined(), "attn_rms_out must be provided");
    TORCH_CHECK(mlp_rms_out.defined(), "mlp_rms_out must be provided");
    TORCH_CHECK(lm_head_rms_scratch.defined(), "lm_head_rms_out must be provided");
    TORCH_CHECK(attn_rms_out.is_cuda(), "attn_rms_out must be CUDA");
    TORCH_CHECK(mlp_rms_out.is_cuda(), "mlp_rms_out must be CUDA");
    TORCH_CHECK(lm_head_rms_scratch.is_cuda(), "lm_head_rms_out must be CUDA");
    TORCH_CHECK(attn_rms_out.scalar_type() == torch::kBFloat16, "attn_rms_out must be bf16");
    TORCH_CHECK(mlp_rms_out.scalar_type() == torch::kBFloat16, "mlp_rms_out must be bf16");
    TORCH_CHECK(lm_head_rms_scratch.scalar_type() == torch::kBFloat16, "lm_head_rms_out must be bf16");
    TORCH_CHECK(attn_rms_out.dim() == 2, "attn_rms_out must be [B, H]");
    TORCH_CHECK(mlp_rms_out.dim() == 2, "mlp_rms_out must be [B, H]");
    TORCH_CHECK(lm_head_rms_scratch.dim() == 2, "lm_head_rms_out must be [B, H]");
    TORCH_CHECK(attn_rms_out.size(1) == LlamaConfig::HIDDEN_DIM, "attn_rms_out H mismatch");
    TORCH_CHECK(mlp_rms_out.size(1) == LlamaConfig::HIDDEN_DIM, "mlp_rms_out H mismatch");
    TORCH_CHECK(lm_head_rms_scratch.size(1) == LlamaConfig::HIDDEN_DIM, "lm_head_rms_out H mismatch");
    TORCH_CHECK(attn_rms_out.size(0) == batch_size, "attn_rms_out batch dim must match batch_size");
    TORCH_CHECK(mlp_rms_out.size(0) == batch_size, "mlp_rms_out batch dim must match batch_size");
    TORCH_CHECK(lm_head_rms_scratch.size(0) == batch_size, "lm_head_rms_out batch dim must match batch_size");
    g.attn_rms_out_ptr = reinterpret_cast<bfloat16_t*>(attn_rms_out.data_ptr<torch::BFloat16>());
    g.mlp_rms_out_ptr = reinterpret_cast<bfloat16_t*>(mlp_rms_out.data_ptr<torch::BFloat16>());
    g.lm_head_rms_out_ptr = reinterpret_cast<bfloat16_t*>(lm_head_rms_scratch.data_ptr<torch::BFloat16>());

    g.debug_buf  = nullptr;
#if defined(ENABLE_PROFILING) || defined(ENABLE_BOUNDARY_PROFILING)
    g.prof_ptr = nullptr;
    g.prof_max_insts = 0;
    if (prof_buf != nullptr) {
        g.prof_ptr = const_cast<int64_t*>(prof_buf->data_ptr<int64_t>());
        g.prof_max_insts = static_cast<int32_t>(prof_buf->size(1));
    }
#endif
}

static void launch_mk_kernel(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    const torch::Tensor* prof_buf,
    int pos_id,
    int batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out)
{
    LlamaGlobals g{};
    populate_llama_globals(
        g,
        instructions,
        hidden_states,
        q_post_rope,
        k_cache,
        v_cache,
        attn_out,
        attn_lse_intermediates,
        attn_out_intermediates,
        silu_out,
        logits,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        o_proj_weights,
        mlp_norm_weights,
        up_proj_weights,
        gate_proj_weights,
        down_proj_weights,
        lm_head_norm_weights,
        lm_head_weights,
        barriers,
        prof_buf,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids,
        active_mask,
        attn_partial_valid,
        attn_rms_out,
        mlp_rms_out,
        lm_head_rms_out);

    size_t smem_size = sizeof(SmemLayout);

    cudaError_t err = cudaFuncSetAttribute(
        mk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    TORCH_CHECK(
        err == cudaSuccess,
        "mk cudaFuncSetAttribute failed: ",
        cudaGetErrorString(err));
    validate_residency(mk_kernel, smem_size, "mk_kernel");

    const int num_worker_grid = static_cast<int>(instructions.size(0));
    TORCH_CHECK(
        num_worker_grid > 0 && num_worker_grid <= KernelConfig::NUM_WORKERS,
        "instructions.size(0) must be in [1, ",
        KernelConfig::NUM_WORKERS,
        "] for this kernel configuration, got ",
        num_worker_grid);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mk_kernel<<<num_worker_grid, KernelConfig::NUM_THREADS, smem_size, stream>>>(g);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mk launch failed: ", cudaGetErrorString(err));
}

static void launch_mk_step_kernel(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    const torch::Tensor* prof_buf,
    int pos_id,
    int batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out,
    int inst_idx)
{
    TORCH_CHECK(
        inst_idx >= 0 && inst_idx < static_cast<int>(instructions.size(1)),
        "mk_step: inst_idx out of range");

    LlamaGlobals g{};
    populate_llama_globals(
        g,
        instructions,
        hidden_states,
        q_post_rope,
        k_cache,
        v_cache,
        attn_out,
        attn_lse_intermediates,
        attn_out_intermediates,
        silu_out,
        logits,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        o_proj_weights,
        mlp_norm_weights,
        up_proj_weights,
        gate_proj_weights,
        down_proj_weights,
        lm_head_norm_weights,
        lm_head_weights,
        barriers,
        prof_buf,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids,
        active_mask,
        attn_partial_valid,
        attn_rms_out,
        mlp_rms_out,
        lm_head_rms_out);

    size_t smem_size = sizeof(SmemLayout);

    cudaError_t err = cudaFuncSetAttribute(
        mk_step_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    TORCH_CHECK(
        err == cudaSuccess,
        "mk_step cudaFuncSetAttribute failed: ",
        cudaGetErrorString(err));
    validate_residency(mk_step_kernel, smem_size, "mk_step_kernel");

    const int num_worker_grid = static_cast<int>(instructions.size(0));
    TORCH_CHECK(
        num_worker_grid > 0 && num_worker_grid <= KernelConfig::NUM_WORKERS,
        "instructions.size(0) exceeds this kernel configuration's worker count");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mk_step_kernel<<<num_worker_grid, KernelConfig::NUM_THREADS, smem_size, stream>>>(g, inst_idx);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mk_step launch failed: ", cudaGetErrorString(err));
}

static void launch_mk_step_loop(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    const torch::Tensor* prof_buf,
    int pos_id,
    int batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out)
{
    LlamaGlobals g{};
    populate_llama_globals(
        g,
        instructions,
        hidden_states,
        q_post_rope,
        k_cache,
        v_cache,
        attn_out,
        attn_lse_intermediates,
        attn_out_intermediates,
        silu_out,
        logits,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        o_proj_weights,
        mlp_norm_weights,
        up_proj_weights,
        gate_proj_weights,
        down_proj_weights,
        lm_head_norm_weights,
        lm_head_weights,
        barriers,
        prof_buf,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids,
        active_mask,
        attn_partial_valid,
        attn_rms_out,
        mlp_rms_out,
        lm_head_rms_out);

    const int num_insts = static_cast<int>(instructions.size(1));
    const int num_worker_grid = static_cast<int>(instructions.size(0));
    TORCH_CHECK(
        num_worker_grid > 0 && num_worker_grid <= KernelConfig::NUM_WORKERS,
        "instructions.size(0) exceeds this kernel configuration's worker count");

    size_t smem_size = sizeof(SmemLayout);
    cudaError_t err = cudaFuncSetAttribute(
        mk_step_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    TORCH_CHECK(
        err == cudaSuccess,
        "mk_step_loop cudaFuncSetAttribute failed: ",
        cudaGetErrorString(err));
    validate_residency(mk_step_kernel, smem_size, "mk_step_kernel");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < num_insts; ++i) {
        mk_step_kernel<<<num_worker_grid, KernelConfig::NUM_THREADS, smem_size, stream>>>(g, i);
        err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "mk_step_loop launch failed at instruction ",
            i,
            ": ",
            cudaGetErrorString(err));
    }
}
}  // namespace

// =============================================================================
// Device: dispatch one instruction slot (instruction words already in smem)
// =============================================================================

__device__ __forceinline__ void mk_dispatch_instruction(
    LlamaGlobals& g,
    SmemLayout& smem,
    int sm_id,
    int inst_idx)
{
    int opcode = smem.instruction[0];
    PROF_RECORD_START(g, sm_id, inst_idx, opcode);
    switch (opcode) {
        case OPCODE_QKV_GEMM_ROPE:
            qkv_gemm_rope(g, smem, inst_idx);
            break;
        case OPCODE_ATTN_RMS:
            decode_attn_rms(g, smem, inst_idx);
            break;
        case OPCODE_MLP_RMS:
            decode_mlp_rms(g, smem, inst_idx);
            break;
        case OPCODE_LMHEAD_RMS:
            decode_lmhead_rms(g, smem, inst_idx);
            break;
        case OPCODE_ATTENTION_PARTIAL:
            decode_attention_splitkv(g, smem, inst_idx);
            break;
        case OPCODE_ATTENTION_REDUCTION:
            decode_attention_reduction(g, smem, inst_idx);
            break;
        case OPCODE_O_PROJ_RESIDUAL:
            decode_oproj_gemm_residual(g, smem, inst_idx);
            break;
        case OPCODE_UPGATE_SILU:
            decode_upgate_gemm_silu(g, smem, inst_idx);
            break;
        case OPCODE_DOWN_PROJ_RESIDUAL:
            decode_downproj_gemm_residual(g, smem, inst_idx);
            break;
        case OPCODE_LMHEAD_GEMM:
            decode_lmhead_gemm(g, smem, inst_idx);
            break;
        default:
            break;
    }
    PROF_RECORD_DONE(g, sm_id, inst_idx);
}

// =============================================================================
// Kernels
// =============================================================================

__global__ __launch_bounds__(
    KernelConfig::NUM_THREADS,
    KernelConfig::TARGET_CTAS_PER_SM)
void mk_kernel(LlamaGlobals g) {
    int sm_id = blockIdx.x;
    int tid   = threadIdx.x;

    extern __shared__ uint8_t raw_smem[];
    SmemLayout& smem = *reinterpret_cast<SmemLayout*>(raw_smem);

    int32_t* inst_base = g.instruction_ptr
        + sm_id * g.num_insts * kMkInstructionWidth;

    for (int i = 0; i < g.num_insts; ++i) {
        if (tid < kMkInstructionWidth)
            smem.instruction[tid] = inst_base[i * kMkInstructionWidth + tid];
        __syncthreads();

        mk_dispatch_instruction(g, smem, sm_id, i);
        __syncthreads();
    }
}

__global__ __launch_bounds__(
    KernelConfig::NUM_THREADS,
    KernelConfig::TARGET_CTAS_PER_SM)
void mk_step_kernel(LlamaGlobals g, int inst_idx) {
    int sm_id = blockIdx.x;
    int tid   = threadIdx.x;

    extern __shared__ uint8_t raw_smem[];
    SmemLayout& smem = *reinterpret_cast<SmemLayout*>(raw_smem);

    int32_t* inst_base = g.instruction_ptr
        + sm_id * g.num_insts * kMkInstructionWidth;

    if (tid < kMkInstructionWidth)
        smem.instruction[tid] = inst_base[inst_idx * kMkInstructionWidth + tid];
    __syncthreads();

    mk_dispatch_instruction(g, smem, sm_id, inst_idx);
    __syncthreads();
}

// =============================================================================
// Host entry points
// =============================================================================

void mk(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    int   pos_id,
    int   batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out)
{
    launch_mk_kernel(
        instructions,
        hidden_states,
        q_post_rope,
        k_cache,
        v_cache,
        attn_out,
        attn_lse_intermediates,
        attn_out_intermediates,
        silu_out,
        logits,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        o_proj_weights,
        mlp_norm_weights,
        up_proj_weights,
        gate_proj_weights,
        down_proj_weights,
        lm_head_norm_weights,
        lm_head_weights,
        barriers,
        nullptr,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids,
        active_mask,
        attn_partial_valid,
        attn_rms_out,
        mlp_rms_out,
        lm_head_rms_out);
}

void mk_step(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    int   inst_idx,
    int   pos_id,
    int   batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out)
{
    launch_mk_step_kernel(
        instructions,
        hidden_states,
        q_post_rope,
        k_cache,
        v_cache,
        attn_out,
        attn_lse_intermediates,
        attn_out_intermediates,
        silu_out,
        logits,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        o_proj_weights,
        mlp_norm_weights,
        up_proj_weights,
        gate_proj_weights,
        down_proj_weights,
        lm_head_norm_weights,
        lm_head_weights,
        barriers,
        nullptr,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids,
        active_mask,
        attn_partial_valid,
        attn_rms_out,
        mlp_rms_out,
        lm_head_rms_out,
        inst_idx);
}

void mk_step_loop(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    int   pos_id,
    int   batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out)
{
    launch_mk_step_loop(
        instructions,
        hidden_states,
        q_post_rope,
        k_cache,
        v_cache,
        attn_out,
        attn_lse_intermediates,
        attn_out_intermediates,
        silu_out,
        logits,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        o_proj_weights,
        mlp_norm_weights,
        up_proj_weights,
        gate_proj_weights,
        down_proj_weights,
        lm_head_norm_weights,
        lm_head_weights,
        barriers,
        nullptr,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids,
        active_mask,
        attn_partial_valid,
        attn_rms_out,
        mlp_rms_out,
        lm_head_rms_out);
}

void mk_profile(
    const torch::Tensor& instructions,
    const torch::Tensor& hidden_states,
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
    const torch::Tensor& prof_buf,
    int   pos_id,
    int   batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids,
    const std::optional<torch::Tensor>& active_mask,
    const std::optional<torch::Tensor>& attn_partial_valid,
    const torch::Tensor& attn_rms_out,
    const torch::Tensor& mlp_rms_out,
    const torch::Tensor& lm_head_rms_out)
{
    TORCH_CHECK(prof_buf.is_cuda(), "prof_buf must be CUDA tensor");
    TORCH_CHECK(prof_buf.scalar_type() == torch::kInt64, "prof_buf dtype must be int64");
    TORCH_CHECK(
        prof_buf.dim() == 3,
        "prof_buf must have shape [num_sms, max_insts, PROF_FIELDS_PER_INST]");
    TORCH_CHECK(
        prof_buf.size(2) == PROF_FIELDS_PER_INST,
        "prof_buf last dim must equal PROF_FIELDS_PER_INST");
    TORCH_CHECK(prof_buf.is_contiguous(), "prof_buf must be contiguous");

    launch_mk_kernel(
        instructions,
        hidden_states,
        q_post_rope,
        k_cache,
        v_cache,
        attn_out,
        attn_lse_intermediates,
        attn_out_intermediates,
        silu_out,
        logits,
        attn_norm_weights,
        qkv_weights,
        rope_cos,
        rope_sin,
        o_proj_weights,
        mlp_norm_weights,
        up_proj_weights,
        gate_proj_weights,
        down_proj_weights,
        lm_head_norm_weights,
        lm_head_weights,
        barriers,
        &prof_buf,
        pos_id,
        batch_size,
        attn_scale,
        rms_norm_eps,
        position_ids,
        active_mask,
        attn_partial_valid,
        attn_rms_out,
        mlp_rms_out,
        lm_head_rms_out);
}
