#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "../../single_batch/decode_kernel/include/megakernel.cuh"
#include "../../single_batch/decode_kernel/include/barrier.cuh"
#include "globals.cuh"
#include "instruction_list.cuh"

#include "rms_qkv_gemm_rope.cu"
#include "decode_attention.cu"
#include "oproj_gemm_residual.cu"
#include "rms_upgate_gemm_silu.cu"
#include "down_proj_gemm_residual.cu"
#include "rms_lmhead_gemm.cu"

using config = llama_mb::h100_mb16_config;
using globals = llama_mb::globals;
using qkv_op = llama_mb::qkv_gemm<config, globals>;
using attention_op = llama_mb::decode_attention<config, globals>;
using oproj_op = llama_mb::oproj_gemm<config, globals>;
using upgate_op = llama_mb::upgate_gemm<config, globals>;
using down_op = llama_mb::down_proj_gemm<config, globals>;
using down_reduce_op = llama_mb::down_proj_reduction<config, globals>;
using lmhead_op = llama_mb::lmhead_gemm<config, globals>;

#ifndef MK_MODULE_NAME
#define MK_MODULE_NAME qimeng_llama3_2_1b_h100_multi_batch_kernel
#endif

PYBIND11_MODULE(MK_MODULE_NAME, m) {
    m.doc() = "Llama-3.2-1B H100 instruction-queue multi-batch decode";
    kittens::py::bind_kernel<
        megakernel::mk<config, globals, lmhead_op, qkv_op, attention_op,
                       oproj_op, upgate_op, down_op, down_reduce_op>>(
        m, "mk", &globals::instructions, &globals::hidden_states,
        &globals::q_post_rope,
        &globals::k_cache, &globals::v_cache, &globals::k_cache_decode,
        &globals::v_cache_decode, &globals::attn_out,
        &globals::silu_out, &globals::logits, &globals::attn_rms_out,
        &globals::mlp_rms_out, &globals::lm_head_rms_out,
        &globals::attn_rms_out_low, &globals::attn_rms_out_high,
        &globals::attn_out_low, &globals::attn_out_high,
        &globals::down_partials, &globals::attn_norm_weights,
        &globals::qkv_weights,
        &globals::rope_cos, &globals::rope_sin, &globals::o_proj_weights,
        &globals::mlp_norm_weights, &globals::up_proj_weights,
        &globals::gate_proj_weights, &globals::down_proj_weights,
        &globals::lm_head_norm_weights, &globals::lm_head_weights,
        &globals::Bar, &globals::position_ids, &globals::active_mask,
        &globals::batch_size, &globals::max_seq_len,
        &globals::attn_scale, &globals::rms_norm_eps);
}
