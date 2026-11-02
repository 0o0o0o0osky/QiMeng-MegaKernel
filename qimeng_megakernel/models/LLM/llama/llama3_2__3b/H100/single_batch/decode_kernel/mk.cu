#include "instruction_list.cuh"

#include "rms_qkv_gemv_rope.cu"
#include "decode_attention_splitkv.cu"
#include "decode_attention_reduction.cu"
#include "oproj_gemv_residual.cu"
#include "rms_upgate_gemv_silu.cu"
#include "down_proj_gemv_residual.cu"
#include "rms_lmhead_gemv.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkv_rope_append_op =
    rms_qkv_rope_append<default_config, llama_3b_globals>;
using attention_partial_op =
    attention_partial<default_config, llama_3b_globals>;
using attention_reduction_op =
    attention_reduction<default_config, llama_3b_globals>;
using o_proj_op = o_proj<default_config, llama_3b_globals>;
using rms_upgate_silu_op = rms_upgate_silu<default_config, llama_3b_globals>;
using downproj_op = downproj<default_config, llama_3b_globals>;
using rms_lm_head_op = rms_lm_head<default_config, llama_3b_globals>;

#ifndef MK_MODULE_NAME
#define MK_MODULE_NAME mk
#endif

PYBIND11_MODULE(MK_MODULE_NAME, m) {
    m.doc() = "Llama 3.2 3B decode MegaKernel (RoPE, SplitKV, Reduction, OProj, UpGate, DownProj, LMHead)";
    kittens::py::bind_kernel<
        mk<default_config, llama_3b_globals, rms_qkv_rope_append_op,
            attention_partial_op, attention_reduction_op, o_proj_op,
            rms_upgate_silu_op, downproj_op, rms_lm_head_op>>(
        m, "mk",
        &llama_3b_globals::instructions,
        &llama_3b_globals::hidden_states,
        &llama_3b_globals::q_post_rope,
        &llama_3b_globals::k_cache,
        &llama_3b_globals::v_cache,
        &llama_3b_globals::attn_out,
        &llama_3b_globals::attn_lse_intermediates,
        &llama_3b_globals::attn_out_intermediates,
        &llama_3b_globals::silu_out,
        &llama_3b_globals::logits,
        &llama_3b_globals::attn_norm_weights,
        &llama_3b_globals::qkv_weights,
        &llama_3b_globals::rope_cos,
        &llama_3b_globals::rope_sin,
        &llama_3b_globals::o_weights,
        &llama_3b_globals::mlp_norm_weights,
        &llama_3b_globals::up_weights,
        &llama_3b_globals::gate_weights,
        &llama_3b_globals::down_weights,
        &llama_3b_globals::lm_head_norm_weights,
        &llama_3b_globals::lm_head_weights,
        &llama_3b_globals::Bar,
        &llama_3b_globals::pos_id,
        &llama_3b_globals::attn_scale,
        &llama_3b_globals::rms_norm_eps);
}
