#include <torch/extension.h>
#include "mk.h"

#ifndef MK_MODULE_NAME
#error "MK_MODULE_NAME must be provided by the route Makefile"
#endif

PYBIND11_MODULE(MK_MODULE_NAME, m) {
    m.doc() = "MegaKernel for Qwen1.5-MoE-A2.7B (A100) full decode (op 1-4, 7-14)";

    m.def("mk",
          &mk,
          "Run Qwen1.5-MoE decode MegaKernel (attention + MoE + lmhead)",
          py::arg("instructions"),
          py::arg("hidden_states"),
          py::arg("normed_hidden_states"),
          py::arg("q_post_rope"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("attn_out"),
          py::arg("attn_lse_intermediates"),
          py::arg("attn_out_intermediates"),
          py::arg("silu_out"),
          py::arg("logits"),
          py::arg("attn_norm_weights"),
          py::arg("qkv_weights"),
          py::arg("qkv_bias"),
          py::arg("rope_cos"),
          py::arg("rope_sin"),
          py::arg("o_proj_weights"),
          py::arg("mlp_norm_weights"),
          py::arg("up_proj_weights"),
          py::arg("gate_proj_weights"),
          py::arg("down_proj_weights"),
          py::arg("lm_head_norm_weights"),
          py::arg("lm_head_weights"),
          py::arg("router_weights"),
          py::arg("router_logits_buf"),
          py::arg("expert_indices"),
          py::arg("expert_weights"),
          py::arg("shared_expert_gate_proj_weights"),
          py::arg("shared_expert_up_proj_weights"),
          py::arg("shared_expert_down_proj_weights"),
          py::arg("shared_expert_silu_out"),
          py::arg("shared_out"),
          py::arg("shared_expert_gate_weights"),
          py::arg("expert_gate_up_proj_weights"),
          py::arg("expert_down_proj_weights"),
          py::arg("routed_silu_buffer"),
          py::arg("routed_out_f32"),
          py::arg("routed_out"),
          py::arg("barriers"),
          py::arg("pos_id"),
          py::arg("attn_scale"),
          py::arg("rms_norm_eps")
    );
}
