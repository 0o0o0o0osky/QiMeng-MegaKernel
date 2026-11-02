#include "instruction_list.cuh"

#include "rms_qkv_gemv_raw.cu"
#include "decode_attention_splitkv.cu"
#include "oproj_gemv_residual.cu"
#include "down_proj_gemv_residual.cu"
#include "rms_upgate_gemv_silu.cu"
#include "rms_lmhead_gemv.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkv_raw_op = rms_qkv_raw<config, qwen3_0_6b_globals>;
using attention_partial_op = attention_partial<config, qwen3_0_6b_globals>;
using o_proj_op = o_proj<config, qwen3_0_6b_globals>;
using rms_upgate_silu_op = rms_upgate_silu<config, qwen3_0_6b_globals>;
using downproj_op = downproj<config, qwen3_0_6b_globals>;
using rms_lm_head_op = rms_lm_head<config, qwen3_0_6b_globals>;

#ifndef MK_MODULE_NAME
#define MK_MODULE_NAME mk
#endif

PYBIND11_MODULE(MK_MODULE_NAME, m) {
    m.doc() = "Qwen3 0.6B H100 production decode MegaKernel. "
              "Bind order includes qkv_raw and q_norm/k_norm weights (no qkv_bias).";
    kittens::py::bind_kernel<
        mk<config, qwen3_0_6b_globals, attention_partial_op,
           rms_qkv_raw_op, downproj_op, o_proj_op, rms_upgate_silu_op,
           rms_lm_head_op>>(
        m, "mk",
        &qwen3_0_6b_globals::instructions,
        &qwen3_0_6b_globals::hidden_states,
        &qwen3_0_6b_globals::qkv_raw,
        &qwen3_0_6b_globals::q_post_rope,
        &qwen3_0_6b_globals::k_cache,
        &qwen3_0_6b_globals::v_cache,
        &qwen3_0_6b_globals::attn_out,
        &qwen3_0_6b_globals::attn_lse_intermediates,
        &qwen3_0_6b_globals::attn_out_intermediates,
        &qwen3_0_6b_globals::silu_out,
        &qwen3_0_6b_globals::logits,
        &qwen3_0_6b_globals::attn_norm_weights,
        &qwen3_0_6b_globals::q_norm_weights,
        &qwen3_0_6b_globals::k_norm_weights,
        &qwen3_0_6b_globals::qkv_weights,
        &qwen3_0_6b_globals::rope_cos,
        &qwen3_0_6b_globals::rope_sin,
        &qwen3_0_6b_globals::o_weights,
        &qwen3_0_6b_globals::mlp_norm_weights,
        &qwen3_0_6b_globals::up_weights,
        &qwen3_0_6b_globals::gate_weights,
        &qwen3_0_6b_globals::down_weights,
        &qwen3_0_6b_globals::lm_head_norm_weights,
        &qwen3_0_6b_globals::lm_head_weights,
        &qwen3_0_6b_globals::Bar,
        &qwen3_0_6b_globals::pos_id,
        &qwen3_0_6b_globals::attn_scale,
        &qwen3_0_6b_globals::rms_norm_eps);
}
