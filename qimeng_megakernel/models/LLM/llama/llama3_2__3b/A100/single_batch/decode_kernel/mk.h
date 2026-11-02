#pragma once
#include <torch/extension.h>
#
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
    int pos_id,
    float attn_scale,
    float rms_norm_eps
);
