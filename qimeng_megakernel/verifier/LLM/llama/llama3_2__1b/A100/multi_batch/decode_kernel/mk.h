#pragma once
#include <optional>
#include <torch/extension.h>

#include "profiling.cuh"

// Host entry point for the multi-batch decode MegaKernel.
// All tensor arguments are contiguous on the GPU.
void mk(
    const torch::Tensor& instructions,           // [num_sm_scheduled, num_insts, 32] int32
    const torch::Tensor& hidden_states,          // [batch_size, HIDDEN_DIM] bf16
    const torch::Tensor& q_post_rope,            // [batch_size, HIDDEN_DIM] bf16
    const torch::Tensor& k_cache,               // [L, batch_size, Seq, KV_H, D] bf16
    const torch::Tensor& v_cache,               // [L, batch_size, Seq, KV_H, D] bf16
    const torch::Tensor& attn_out,              // [batch_size, HIDDEN_DIM] bf16
    const torch::Tensor& attn_lse_intermediates, // [NUM_HEADS, max_splits] float
    const torch::Tensor& attn_out_intermediates, // [NUM_HEADS, max_splits, HEAD_DIM] float
    const torch::Tensor& silu_out,              // [batch_size, INTERM_DIM] bf16
    const torch::Tensor& logits,                // [VOCAB_SIZE] bf16
    const torch::Tensor& attn_norm_weights,     // [L, HIDDEN_DIM] bf16
    const torch::Tensor& qkv_weights,           // [L, QKV_OUT_DIM, HIDDEN_DIM] bf16
    const torch::Tensor& rope_cos,              // [max_seq_len, HEAD_DIM] float
    const torch::Tensor& rope_sin,              // [max_seq_len, HEAD_DIM] float
    const torch::Tensor& o_proj_weights,        // [L, HIDDEN_DIM, HIDDEN_DIM] bf16
    const torch::Tensor& mlp_norm_weights,      // [L, HIDDEN_DIM] bf16
    const torch::Tensor& up_proj_weights,       // [L, INTERM_DIM, HIDDEN_DIM] bf16
    const torch::Tensor& gate_proj_weights,     // [L, INTERM_DIM, HIDDEN_DIM] bf16
    const torch::Tensor& down_proj_weights,     // [L, HIDDEN_DIM, INTERM_DIM] bf16
    const torch::Tensor& lm_head_norm_weights,  // [HIDDEN_DIM] bf16
    const torch::Tensor& lm_head_weights,       // [VOCAB_SIZE, HIDDEN_DIM] bf16
    const torch::Tensor& barriers,              // [L, num_layer_barriers, NUM_HEADS] int32
    int   pos_id,
    int   batch_size,
    float attn_scale,
    float rms_norm_eps,
    const std::optional<torch::Tensor>& position_ids = std::nullopt,  // [batch_size] int32 cuda, contiguous
    const std::optional<torch::Tensor>& active_mask = std::nullopt,   // [batch_size] uint8 or bool cuda
    const std::optional<torch::Tensor>& attn_partial_valid = std::nullopt,
    const torch::Tensor& attn_rms_out = torch::Tensor(),
    const torch::Tensor& mlp_rms_out = torch::Tensor(),
    const torch::Tensor& lm_head_rms_out = torch::Tensor()
);

// One MegaKernel column uses the same `instructions` tensor as mk ([num_sm, num_insts, 32]); `inst_idx` selects the column.
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
    const std::optional<torch::Tensor>& position_ids = std::nullopt,
    const std::optional<torch::Tensor>& active_mask = std::nullopt,
    const std::optional<torch::Tensor>& attn_partial_valid = std::nullopt,
    const torch::Tensor& attn_rms_out = torch::Tensor(),
    const torch::Tensor& mlp_rms_out = torch::Tensor(),
    const torch::Tensor& lm_head_rms_out = torch::Tensor()
);

// `num_insts` sequential launches of mk_step_kernel (columns 0..num_insts-1).
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
    const std::optional<torch::Tensor>& position_ids = std::nullopt,
    const std::optional<torch::Tensor>& active_mask = std::nullopt,
    const std::optional<torch::Tensor>& attn_partial_valid = std::nullopt,
    const torch::Tensor& attn_rms_out = torch::Tensor(),
    const torch::Tensor& mlp_rms_out = torch::Tensor(),
    const torch::Tensor& lm_head_rms_out = torch::Tensor()
);

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
    const std::optional<torch::Tensor>& position_ids = std::nullopt,
    const std::optional<torch::Tensor>& active_mask = std::nullopt,
    const std::optional<torch::Tensor>& attn_partial_valid = std::nullopt,
    const torch::Tensor& attn_rms_out = torch::Tensor(),
    const torch::Tensor& mlp_rms_out = torch::Tensor(),
    const torch::Tensor& lm_head_rms_out = torch::Tensor()
);
