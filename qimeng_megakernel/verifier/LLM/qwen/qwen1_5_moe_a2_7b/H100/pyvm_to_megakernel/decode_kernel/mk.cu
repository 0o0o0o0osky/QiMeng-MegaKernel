// Qwen1.5-MoE-A2.7B H100 decode MegaKernel translation unit.
// bind_kernel argument order follows globals.cuh and decode_schedule/mk_kernel.py.

#include "instruction_list.cuh"

#include "rms_qkv_gemv_rope.cu"           // opcode 1
#include "decode_attention_splitkv.cu"    // opcode 2
#include "decode_attention_reduction.cu"  // opcode 3
#include "oproj_gemv_residual.cu"         // opcode 4
#include "rms_lmhead_gemv.cu"             // opcode 7
#include "router_gemv.cu"                 // opcode 8
#include "router_softmax_topk.cu"         // opcode 9
#include "shared_expert_upgate_silu.cu"   // opcode 10
#include "shared_expert_downproj.cu"      // opcode 11
#include "routed_expert_upgate_silu.cu"   // opcode 12
#include "routed_expert_downproj.cu"      // opcode 13
#include "moe_combine_commit.cu"          // opcode 14

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkvproj_gemv_rope_op =
    rms_qkvproj_gemv_rope<default_config, qwen_moe_globals>;
using decode_attention_splitkv_op =
    decode_attention_splitkv<default_config, qwen_moe_globals>;
using decode_attention_reduction_op =
    decode_attention_reduction<default_config, qwen_moe_globals>;
using oproj_gemv_residual_op =
    oproj_gemv_residual<default_config, qwen_moe_globals>;
using rms_lmhead_gemv_op = rms_lmhead_gemv<default_config, qwen_moe_globals>;
using router_gemv_op = router_gemv<default_config, qwen_moe_globals>;
using router_softmax_topk_op =
    router_softmax_topk<default_config, qwen_moe_globals>;
using shared_expert_upgate_silu_op =
    shared_expert_upgate_silu<default_config, qwen_moe_globals>;
using shared_expert_downproj_op =
    shared_expert_downproj<default_config, qwen_moe_globals>;
using routed_expert_upgate_silu_op =
    routed_expert_upgate_silu<default_config, qwen_moe_globals>;
using routed_expert_downproj_op =
    routed_expert_downproj<default_config, qwen_moe_globals>;
using moe_combine_commit_op =
    moe_combine_commit<default_config, qwen_moe_globals>;

#ifndef MK_MODULE_NAME
#define MK_MODULE_NAME mk
#endif

PYBIND11_MODULE(MK_MODULE_NAME, m) {
    m.doc() = "Qwen1.5-MoE-A2.7B H100 single-batch decode MegaKernel";
    kittens::py::bind_kernel<
        mk<default_config, qwen_moe_globals,
           rms_qkvproj_gemv_rope_op,          // 1
           decode_attention_splitkv_op,       // 2
           decode_attention_reduction_op,     // 3
           oproj_gemv_residual_op,            // 4
           rms_lmhead_gemv_op,                // 7
           router_gemv_op,                    // 8
           router_softmax_topk_op,            // 9
           shared_expert_upgate_silu_op,      // 10
           shared_expert_downproj_op,         // 11
           routed_expert_upgate_silu_op,      // 12
           routed_expert_downproj_op,         // 13
           moe_combine_commit_op>>(           // 14
        m, "mk",
        // --- instruction stream -------------------------------------- #1
        &qwen_moe_globals::instructions,
        // --- activation buffers ----------------------------------- #2..#17
        &qwen_moe_globals::hidden_states,
        &qwen_moe_globals::normed_hidden_states,
        &qwen_moe_globals::q_post_rope,
        &qwen_moe_globals::k_cache,
        &qwen_moe_globals::v_cache,
        &qwen_moe_globals::attn_out,
        &qwen_moe_globals::attn_lse_intermediates,
        &qwen_moe_globals::attn_out_intermediates,
        &qwen_moe_globals::shared_expert_silu_out,
        &qwen_moe_globals::shared_out,
        &qwen_moe_globals::routed_silu_buffer,
        &qwen_moe_globals::routed_out_f32,
        &qwen_moe_globals::router_logits_buf,
        &qwen_moe_globals::expert_indices,
        &qwen_moe_globals::expert_weights,
        &qwen_moe_globals::logits,
        // --- weights ---------------------------------------------- #18..#33
        &qwen_moe_globals::attn_norm_weights,
        &qwen_moe_globals::qkv_weights,
        &qwen_moe_globals::qkv_bias,
        &qwen_moe_globals::rope_cos,
        &qwen_moe_globals::rope_sin,
        &qwen_moe_globals::o_proj_weights,
        &qwen_moe_globals::mlp_norm_weights,
        &qwen_moe_globals::shared_expert_gate_proj_weights,
        &qwen_moe_globals::shared_expert_up_proj_weights,
        &qwen_moe_globals::shared_expert_down_proj_weights,
        &qwen_moe_globals::shared_expert_gate_weights,
        &qwen_moe_globals::router_weights,
        &qwen_moe_globals::expert_gate_up_proj_weights_flat,
        &qwen_moe_globals::expert_down_proj_weights_flat,
        &qwen_moe_globals::lm_head_norm_weights,
        &qwen_moe_globals::lm_head_weights,
        // --- sync + scalars --------------------------------------- #34..#37
        &qwen_moe_globals::Bar,
        &qwen_moe_globals::pos_id,
        &qwen_moe_globals::attn_scale,
        &qwen_moe_globals::rms_norm_eps);
}
