#pragma once
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cstdint>

#include "qwen_moe_config.cuh"

namespace megakernel {
    using namespace cute;
    using bfloat16_t = cutlass::bfloat16_t;

    struct QwenGlobals {
        int32_t*    instruction_ptr;
        int32_t*    barrier_ptr;

        bfloat16_t* hidden_state_ptr;
        bfloat16_t* normed_hidden_states_ptr;  // [HIDDEN_DIM], written by opcode 8
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

        bfloat16_t* router_weight_ptr;
        float*      router_logits_buf_ptr;  // [num_experts] float32, written by Router_GEMV
        int64_t*    expert_indices_ptr;
        float*      expert_weights_ptr;

        bfloat16_t* shared_expert_gate_proj_ptr;
        bfloat16_t* shared_expert_up_proj_ptr;
        bfloat16_t* shared_expert_down_proj_ptr;
        bfloat16_t* shared_expert_silu_out_ptr;
        bfloat16_t* shared_out_ptr;
        bfloat16_t* shared_expert_gate_weight_ptr;  // [L, H, 1]: scalar gate for shared expert

        // Routed expert weights: [L, NUM_EXPERTS, 2*MOE_INTERM_DIM, HIDDEN_DIM] (gate+up fused)
        // and [L, NUM_EXPERTS, HIDDEN_DIM, MOE_INTERM_DIM] (down proj)
        bfloat16_t* expert_gate_up_proj_ptr;   // [L, NUM_EXPERTS, 2*MOE_INTERM, H]
        bfloat16_t* expert_down_proj_ptr;      // [L, NUM_EXPERTS, H, MOE_INTERM]
        // Per-expert SiLU intermediate: [NUM_EXPERTS_PER_TOK, MOE_INTERM_DIM]
        bfloat16_t* routed_silu_buffer_ptr;
        // Routed expert accumulation output: [HIDDEN_DIM]
        float*      routed_out_f32_ptr;
        bfloat16_t* routed_out_ptr;

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
                make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_normed_hidden_state() const {
            return make_tensor(
                make_gmem_ptr(normed_hidden_states_ptr),
                make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_q_post_rope() const {
            return make_tensor(
                make_gmem_ptr(q_post_rope_ptr),
                make_shape(Int<QwenMoeConfig::NUM_Q_HEADS * QwenMoeConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_norm_weight() const {
            return make_tensor(
                make_gmem_ptr(attn_norm_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_mlp_norm_weight() const {
            return make_tensor(
                make_gmem_ptr(mlp_norm_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_silu_out() const {
            return make_tensor(
                make_gmem_ptr(silu_out_ptr),
                make_shape(Int<QwenMoeConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_qkv_weight() const {
            return make_tensor(
                make_gmem_ptr(qkv_weight_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::QKV_OUT_DIM>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_qkv_bias() const {
            return make_tensor(
                make_gmem_ptr(qkv_bias_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::QKV_OUT_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_o_weight() const {
            return make_tensor(
                make_gmem_ptr(o_weight_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_up_weight() const {
            return make_tensor(
                make_gmem_ptr(up_weight_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::INTERM_DIM>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_gate_weight() const {
            return make_tensor(
                make_gmem_ptr(gate_weight_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::INTERM_DIM>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_down_weight() const {
            return make_tensor(
                make_gmem_ptr(down_weight_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}, Int<QwenMoeConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_lm_head_weight() const {
            return make_tensor(
                make_gmem_ptr(lm_head_weight_ptr),
                make_shape(Int<QwenMoeConfig::VOCAB_SIZE>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_final_rms_weight() const {
            return make_tensor(
                make_gmem_ptr(lm_head_norm_ptr),
                make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_logits() const {
            return make_tensor(
                make_gmem_ptr(logits_ptr),
                make_shape(Int<QwenMoeConfig::VOCAB_SIZE>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_norm() const {
            return make_tensor(
                make_gmem_ptr(attn_norm_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_mlp_norm() const {
            return make_tensor(
                make_gmem_ptr(mlp_norm_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_rope_cos() const {
            return make_tensor(
                make_gmem_ptr(rope_cos_ptr),
                make_shape(max_seq_len, Int<QwenMoeConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_rope_sin() const {
            return make_tensor(
                make_gmem_ptr(rope_sin_ptr),
                make_shape(max_seq_len, Int<QwenMoeConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_k_cache() const {
            return make_tensor(
                make_gmem_ptr(k_cache_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<1>{}, max_seq_len,
                           Int<QwenMoeConfig::NUM_KV_HEADS>{}, Int<QwenMoeConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_v_cache() const {
            return make_tensor(
                make_gmem_ptr(v_cache_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<1>{}, max_seq_len,
                           Int<QwenMoeConfig::NUM_KV_HEADS>{}, Int<QwenMoeConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_barrier() const {
            return make_tensor(
                make_gmem_ptr(barrier_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, num_layer_barriers, Int<QwenMoeConfig::BARRIER_SLOTS>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_lse_intermediates() const {
            return make_tensor(
                make_gmem_ptr(attn_lse_ptr),
                make_shape(Int<QwenMoeConfig::NUM_Q_HEADS>{}, max_splits),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_out() const {
            return make_tensor(
                make_gmem_ptr(attn_out_ptr),
                make_shape(Int<QwenMoeConfig::NUM_Q_HEADS * QwenMoeConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_attn_out_intermediates() const {
            return make_tensor(
                make_gmem_ptr(attn_out_inter_ptr),
                make_shape(Int<QwenMoeConfig::NUM_Q_HEADS>{}, max_splits, Int<QwenMoeConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_router_weight() const {
            return make_tensor(
                make_gmem_ptr(router_weight_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::NUM_EXPERTS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_expert_indices() const {
            return make_tensor(
                make_gmem_ptr(expert_indices_ptr),
                make_shape(Int<QwenMoeConfig::NUM_EXPERTS_PER_TOK>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_expert_weights() const {
            return make_tensor(
                make_gmem_ptr(expert_weights_ptr),
                make_shape(Int<QwenMoeConfig::NUM_EXPERTS_PER_TOK>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_shared_expert_gate_proj_weight() const {
            return make_tensor(
                make_gmem_ptr(shared_expert_gate_proj_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::INTERM_DIM>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_shared_expert_up_proj_weight() const {
            return make_tensor(
                make_gmem_ptr(shared_expert_up_proj_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::INTERM_DIM>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_shared_expert_down_proj_weight() const {
            return make_tensor(
                make_gmem_ptr(shared_expert_down_proj_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}, Int<QwenMoeConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_shared_expert_silu_out() const {
            return make_tensor(
                make_gmem_ptr(shared_expert_silu_out_ptr),
                make_shape(Int<QwenMoeConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_shared_out() const {
            return make_tensor(
                make_gmem_ptr(shared_out_ptr),
                make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // shared expert scalar gate weight: [L, H] (original shape is [L, H, 1], squeezed)
        __device__ __forceinline__ auto get_shared_expert_gate_weight() const {
            return make_tensor(
                make_gmem_ptr(shared_expert_gate_weight_ptr),
                make_shape(Int<QwenMoeConfig::NUM_LAYERS>{}, Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // --- Routed expert accessors ---

        // gate+up weights fused: [L, NUM_EXPERTS, 2*MOE_INTERM, H]
        // Total size > int32_t range; use raw pointer arithmetic in kernels.
        __device__ __forceinline__ const bfloat16_t* get_expert_gate_up_proj(int layer, int eid) const {
            constexpr long stride_l = static_cast<long>(QwenMoeConfig::NUM_EXPERTS)
                                    * (2 * QwenMoeConfig::MOE_INTERM_DIM)
                                    * QwenMoeConfig::HIDDEN_DIM;
            constexpr long stride_e = static_cast<long>(2 * QwenMoeConfig::MOE_INTERM_DIM)
                                    * QwenMoeConfig::HIDDEN_DIM;
            return expert_gate_up_proj_ptr + layer * stride_l + eid * stride_e;
        }

        __device__ __forceinline__ auto get_expert_gate_up_proj_weight() const {
            constexpr long total = static_cast<long>(QwenMoeConfig::NUM_LAYERS)
                                 * QwenMoeConfig::NUM_EXPERTS
                                 * 2 * QwenMoeConfig::MOE_INTERM_DIM
                                 * QwenMoeConfig::HIDDEN_DIM;
            return make_tensor(make_gmem_ptr(expert_gate_up_proj_ptr), make_shape(total), LayoutRight{});
        }

        // down proj weights: [L, NUM_EXPERTS, H, MOE_INTERM]
        // Returns flat ptr (total size > int32_t range, index with raw arithmetic in kernel)
        __device__ __forceinline__ const bfloat16_t* get_expert_down_proj_weight_ptr() const {
            return expert_down_proj_ptr;
        }

        // Convenience: pointer to weights for a given (layer, expert)
        __device__ __forceinline__ const bfloat16_t* get_expert_down_proj(int layer, int eid) const {
            constexpr long stride_l = static_cast<long>(QwenMoeConfig::NUM_EXPERTS)
                                    * QwenMoeConfig::HIDDEN_DIM
                                    * QwenMoeConfig::MOE_INTERM_DIM;
            constexpr long stride_e = static_cast<long>(QwenMoeConfig::HIDDEN_DIM)
                                    * QwenMoeConfig::MOE_INTERM_DIM;
            return expert_down_proj_ptr + layer * stride_l + eid * stride_e;
        }

        // 2D accessor for a specific (layer, expert): [H, MOE_INTERM]
        __device__ __forceinline__ auto get_expert_down_proj_weight() const {
            constexpr long total = static_cast<long>(QwenMoeConfig::NUM_LAYERS)
                                 * QwenMoeConfig::NUM_EXPERTS
                                 * QwenMoeConfig::HIDDEN_DIM
                                 * QwenMoeConfig::MOE_INTERM_DIM;
            return make_tensor(
                make_gmem_ptr(expert_down_proj_ptr),
                make_shape(total),
                LayoutRight{}
            );
        }

        // per-expert SiLU buffer: [NUM_EXPERTS_PER_TOK, MOE_INTERM_DIM]
        __device__ __forceinline__ auto get_routed_silu_buffer() const {
            return make_tensor(
                make_gmem_ptr(routed_silu_buffer_ptr),
                make_shape(Int<QwenMoeConfig::NUM_EXPERTS_PER_TOK>{},
                           Int<QwenMoeConfig::MOE_INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        // routed_out accumulation (f32): [HIDDEN_DIM]
        __device__ __forceinline__ auto get_routed_out_f32() const {
            return make_tensor(
                make_gmem_ptr(routed_out_f32_ptr),
                make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // routed_out (bf16): [HIDDEN_DIM]
        __device__ __forceinline__ auto get_routed_out() const {
            return make_tensor(
                make_gmem_ptr(routed_out_ptr),
                make_shape(Int<QwenMoeConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }
    };
}
