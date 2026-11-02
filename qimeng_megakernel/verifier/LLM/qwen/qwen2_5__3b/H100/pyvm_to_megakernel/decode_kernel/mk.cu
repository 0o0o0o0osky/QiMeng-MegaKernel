#include "instruction_list.cuh"

#include "rms_qkv_gemv_rope.cu"
#include "decode_attention_splitkv.cu"
#include "decode_attention_reduction.cu"
#include "oproj_gemv_residual.cu"
#include "down_proj_gemv_residual.cu"
#include "rms_upgate_gemv_silu.cu"
#include "rms_lmhead_gemv.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkv_rope_append_op = rms_qkv_rope_append<config, qwen_2_5_3b_globals>;
using attention_partial_op = attention_partial<config, qwen_2_5_3b_globals>;
using attention_reduction_op = attention_reduction<config, qwen_2_5_3b_globals>;
using o_proj_op = o_proj<config, qwen_2_5_3b_globals>;
using rms_upgate_silu_op = rms_upgate_silu<config, qwen_2_5_3b_globals>;
using downproj_op = downproj<config, qwen_2_5_3b_globals>;
using rms_lm_head_op = rms_lm_head<config, qwen_2_5_3b_globals>;

#ifndef MK_MODULE_NAME
#define MK_MODULE_NAME mk
#endif

PYBIND11_MODULE(MK_MODULE_NAME, m) {
    m.doc() = "Qwen2.5 3B H100 decode MegaKernel (all ops: RMS+RoPE, splitkv, reduction, oproj, rms_upgate, down_proj, rms_lmhead)";
    kittens::py::bind_kernel<
        mk<config, qwen_2_5_3b_globals, attention_partial_op, attention_reduction_op,
           rms_qkv_rope_append_op, downproj_op, o_proj_op, rms_upgate_silu_op,
           rms_lm_head_op>>(
        m, "mk",
        &qwen_2_5_3b_globals::instructions,
        &qwen_2_5_3b_globals::hidden_states,
        &qwen_2_5_3b_globals::q_post_rope,
        &qwen_2_5_3b_globals::k_cache,
        &qwen_2_5_3b_globals::v_cache,
        &qwen_2_5_3b_globals::attn_out,
        &qwen_2_5_3b_globals::attn_lse_intermediates,
        &qwen_2_5_3b_globals::attn_out_intermediates,
        &qwen_2_5_3b_globals::silu_out,
        &qwen_2_5_3b_globals::logits,
        &qwen_2_5_3b_globals::attn_norm_weights,
        &qwen_2_5_3b_globals::qkv_weights,
        &qwen_2_5_3b_globals::rope_cos,
        &qwen_2_5_3b_globals::rope_sin,
        &qwen_2_5_3b_globals::qkv_bias,
        &qwen_2_5_3b_globals::o_weights,
        &qwen_2_5_3b_globals::mlp_norm_weights,
        &qwen_2_5_3b_globals::up_weights,
        &qwen_2_5_3b_globals::gate_weights,
        &qwen_2_5_3b_globals::down_weights,
        &qwen_2_5_3b_globals::lm_head_norm_weights,
        &qwen_2_5_3b_globals::lm_head_weights,
        &qwen_2_5_3b_globals::Bar,
        &qwen_2_5_3b_globals::pos_id,
        &qwen_2_5_3b_globals::attn_scale,
        &qwen_2_5_3b_globals::rms_norm_eps);
}
