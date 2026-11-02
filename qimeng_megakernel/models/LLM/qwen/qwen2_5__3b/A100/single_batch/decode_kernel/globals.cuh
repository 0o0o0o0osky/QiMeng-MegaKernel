#pragma once
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cstdint>

#include "qwen_config.cuh"

namespace megakernel {
    using namespace cute;
    using bfloat16_t = cutlass::bfloat16_t;

    struct QwenGlobals {
        int32_t*    instruction_ptr;
        int32_t*    barrier_ptr;

        bfloat16_t* hidden_state_ptr;
        bfloat16_t* q_post_rope_ptr;
        bfloat16_t* attn_out_ptr;
        bfloat16_t* silu_out_ptr;
        float*      attn_lse_ptr;
        float*      attn_out_inter_ptr;

        bfloat16_t* attn_norm_ptr;
        bfloat16_t* qkv_weight_ptr;
        bfloat16_t* qkv_bias_ptr;
        bfloat16_t* o_weight_ptr;
        bfloat16_t* mlp_norm_ptr;
        bfloat16_t* up_weight_ptr;
        bfloat16_t* gate_weight_ptr;
        bfloat16_t* down_weight_ptr;
        bfloat16_t* lm_head_norm_ptr;
        bfloat16_t* lm_head_weight_ptr;
        bfloat16_t* logits_ptr;

        float*      rope_cos_ptr;
        float*      rope_sin_ptr;

        bfloat16_t* k_cache_ptr;
        bfloat16_t* v_cache_ptr;

        int32_t num_insts;
        int32_t max_seq_len;
        int32_t num_layer_barriers;
        int32_t max_splits;
        uint32_t pos_id;

        float attn_scale;
        float rms_norm_eps;

        __device__ __forceinline__ auto get_hidden_state() const {
            return make_tensor(
                make_gmem_ptr(hidden_state_ptr),
                make_shape(Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_q_post_rope() const {
            return make_tensor(
                make_gmem_ptr(q_post_rope_ptr),
                make_shape(Int<QwenConfig::NUM_Q_HEADS * QwenConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_norm_weight() const {
            return make_tensor(
                make_gmem_ptr(attn_norm_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_mlp_norm_weight() const {
            return make_tensor(
                make_gmem_ptr(mlp_norm_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_silu_out() const {
            return make_tensor(
                make_gmem_ptr(silu_out_ptr),
                make_shape(Int<QwenConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_qkv_weight() const {
            return make_tensor(
                make_gmem_ptr(qkv_weight_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::QKV_OUT_DIM>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_qkv_bias() const {
            return make_tensor(
                make_gmem_ptr(qkv_bias_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::QKV_OUT_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_o_weight() const {
            return make_tensor(
                make_gmem_ptr(o_weight_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::HIDDEN_DIM>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_up_weight() const {
            return make_tensor(
                make_gmem_ptr(up_weight_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::INTERM_DIM>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_gate_weight() const {
            return make_tensor(
                make_gmem_ptr(gate_weight_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::INTERM_DIM>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_down_weight() const {
            return make_tensor(
                make_gmem_ptr(down_weight_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::HIDDEN_DIM>{}, Int<QwenConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_lm_head_weight() const {
            return make_tensor(
                make_gmem_ptr(lm_head_weight_ptr),
                make_shape(Int<QwenConfig::VOCAB_SIZE>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_final_rms_weight() const {
            return make_tensor(
                make_gmem_ptr(lm_head_norm_ptr),
                make_shape(Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_logits() const {
            return make_tensor(
                make_gmem_ptr(logits_ptr),
                make_shape(Int<QwenConfig::VOCAB_SIZE>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_norm() const {
            return make_tensor(
                make_gmem_ptr(attn_norm_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_mlp_norm() const {
            return make_tensor(
                make_gmem_ptr(mlp_norm_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<QwenConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_rope_cos() const {
            return make_tensor(
                make_gmem_ptr(rope_cos_ptr),
                make_shape(max_seq_len, Int<QwenConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_rope_sin() const {
            return make_tensor(
                make_gmem_ptr(rope_sin_ptr),
                make_shape(max_seq_len, Int<QwenConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_k_cache() const {
            return make_tensor(
                make_gmem_ptr(k_cache_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<1>{}, max_seq_len,
                           Int<QwenConfig::NUM_KV_HEADS>{}, Int<QwenConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_v_cache() const {
            return make_tensor(
                make_gmem_ptr(v_cache_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, Int<1>{}, max_seq_len,
                           Int<QwenConfig::NUM_KV_HEADS>{}, Int<QwenConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_barrier() const {
            return make_tensor(
                make_gmem_ptr(barrier_ptr),
                make_shape(Int<QwenConfig::NUM_LAYERS>{}, num_layer_barriers, Int<QwenConfig::NUM_HEADS>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_lse_intermediates() const {
            return make_tensor(
                make_gmem_ptr(attn_lse_ptr),
                make_shape(Int<QwenConfig::NUM_Q_HEADS>{}, max_splits),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_out() const {
            return make_tensor(
                make_gmem_ptr(attn_out_ptr),
                make_shape(Int<QwenConfig::NUM_Q_HEADS * QwenConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_out_intermediates() const {
            return make_tensor(
                make_gmem_ptr(attn_out_inter_ptr),
                make_shape(Int<QwenConfig::NUM_Q_HEADS>{}, max_splits, Int<QwenConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }
    };
}
