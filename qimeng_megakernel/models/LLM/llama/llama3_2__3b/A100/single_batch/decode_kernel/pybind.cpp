#include <torch/extension.h>
#include "mk.h"

#ifndef MK_MODULE_NAME
#error "MK_MODULE_NAME must be provided by the route Makefile"
#endif

PYBIND11_MODULE(MK_MODULE_NAME, m) {
    m.doc() = "MegaKernel for Llama3.2-1B";

    m.def("mk",
          &mk,
          "Run Llama MegaKernel",
          py::arg("instructions"),
          py::arg("hidden_states"),
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
          py::arg("rope_cos"),
          py::arg("rope_sin"),
          py::arg("o_proj_weights"),
          py::arg("mlp_norm_weights"),
          py::arg("up_proj_weights"),
          py::arg("gate_proj_weights"),
          py::arg("down_proj_weights"),
          py::arg("head_norm_weights"),
          py::arg("head_weights"),
          py::arg("barriers"),
          py::arg("pos_id"),
          py::arg("attn_scale"),
          py::arg("rms_norm_eps")
    );
}
