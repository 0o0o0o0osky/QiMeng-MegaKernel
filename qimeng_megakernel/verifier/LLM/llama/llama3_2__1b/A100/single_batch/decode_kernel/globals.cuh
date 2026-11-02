#pragma once
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cstdint>

#include "llama_config.cuh"

namespace megakernel {
    using namespace cute;
    using bfloat16_t = cutlass::bfloat16_t;

    struct LlamaGlobals {
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
    int32_t max_splits;  // Maximum number of attention partials (SM count)
    uint32_t pos_id;

    float attn_scale;
    float rms_norm_eps;
    
        // Hidden State: [HIDDEN_DIM]
        __device__ __forceinline__ auto get_hidden_state() const {
            return make_tensor(
                make_gmem_ptr(hidden_state_ptr),
                make_shape(Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Q after RoPE: [HIDDEN_DIM]
        __device__ __forceinline__ auto get_q_post_rope() const {
            return make_tensor(
                make_gmem_ptr(q_post_rope_ptr),
                make_shape(Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Attention Norm: [Layers, HIDDEN_DIM] (use (layer_idx, _) for one layer)
        __device__ __forceinline__ auto get_attn_norm_weight() const {
            return make_tensor(
                make_gmem_ptr(attn_norm_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // MLP Norm: [Layers, HIDDEN_DIM] (use (layer_idx, _) for one layer)
        __device__ __forceinline__ auto get_mlp_norm_weight() const {
            return make_tensor(
                make_gmem_ptr(mlp_norm_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // SiLU Output: [INTERM_DIM]
        __device__ __forceinline__ auto get_silu_out() const {
            return make_tensor(
                make_gmem_ptr(silu_out_ptr),
                make_shape(Int<LlamaConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        // Barriers: flat pointer
        __device__ __forceinline__ int32_t* get_barriers() const {
            return barrier_ptr;
        }
    
        // QKV Proj: [Layers, QKV_OUT_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_qkv_weight() const {
            return make_tensor(
                make_gmem_ptr(qkv_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::QKV_OUT_DIM>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{} 
            );
        }
    
        // O Proj: [Layers, HIDDEN_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_o_weight() const {
            return make_tensor(
                make_gmem_ptr(o_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }
    
        // MLP Up: [Layers, INTERM_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_up_weight() const {
            return make_tensor(
                make_gmem_ptr(up_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::INTERM_DIM>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // MLP Gate: [Layers, INTERM_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_gate_weight() const {
            return make_tensor(
                make_gmem_ptr(gate_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::INTERM_DIM>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // MLP Down: [Layers, HIDDEN_DIM, INTERM_DIM]
        __device__ __forceinline__ auto get_down_weight() const {
            return make_tensor(
                make_gmem_ptr(down_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}, Int<LlamaConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        // LM Head Weights: [VOCAB_SIZE, HIDDEN_DIM]
        __device__ __forceinline__ auto get_lm_head_weight() const {
            return make_tensor(
                make_gmem_ptr(lm_head_weight_ptr),
                make_shape(Int<LlamaConfig::VOCAB_SIZE>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Final RMS Norm (for LM Head): [HIDDEN_DIM]
        __device__ __forceinline__ auto get_final_rms_weight() const {
            return make_tensor(
                make_gmem_ptr(lm_head_norm_ptr),
                make_shape(Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Logits Output: [VOCAB_SIZE]
        __device__ __forceinline__ auto get_logits() const {
            return make_tensor(
                make_gmem_ptr(logits_ptr),
                make_shape(Int<LlamaConfig::VOCAB_SIZE>{}),
                LayoutRight{}
            );
        }
    
        // RMS Norm: [Layers, HIDDEN_DIM]
        __device__ __forceinline__ auto get_attn_norm() const {
            return make_tensor(
                make_gmem_ptr(attn_norm_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // MLP Norm: [Layers, HIDDEN_DIM]
        __device__ __forceinline__ auto get_mlp_norm() const {
            return make_tensor(
                make_gmem_ptr(mlp_norm_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }
    
        // LM Head Norm: [HIDDEN_DIM]
        __device__ __forceinline__ auto get_lm_head_norm() const {
            return make_tensor(
                make_gmem_ptr(lm_head_norm_ptr),
                make_shape(Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // RoPE: [Seq, HEAD_DIM]
        __device__ __forceinline__ auto get_rope_cos() const {
            return make_tensor(
                make_gmem_ptr(rope_cos_ptr),
                make_shape(max_seq_len, Int<LlamaConfig::HEAD_DIM>{}), 
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_rope_sin() const {
            return make_tensor(
                make_gmem_ptr(rope_sin_ptr),
                make_shape(max_seq_len, Int<LlamaConfig::HEAD_DIM>{}), 
                LayoutRight{}
            );
        }
    
        // KV Cache: [Layers, Batch=1, Seq, Head, Dim]
        __device__ __forceinline__ auto get_k_cache() const {
            return make_tensor(
                make_gmem_ptr(k_cache_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<1>{}, max_seq_len, Int<LlamaConfig::NUM_KV_HEADS>{}, Int<LlamaConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        __device__ __forceinline__ auto get_v_cache() const {
            return make_tensor(
                make_gmem_ptr(v_cache_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<1>{}, max_seq_len, Int<LlamaConfig::NUM_KV_HEADS>{}, Int<LlamaConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }
        
        // Instruction: [NumInsts, 32]
        __device__ __forceinline__ auto get_instruction() const {
            return make_tensor(
                make_gmem_ptr(instruction_ptr),
                make_shape(num_insts, Int<32>{}),
                LayoutRight{}
            );
        }

        // Barrier: [NUM_LAYERS, LayerBarriers, NUM_HEADS] — must match Python (L, 10, 48)
        __device__ __forceinline__ auto get_barrier() const {
            return make_tensor(
                make_gmem_ptr(barrier_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, num_layer_barriers, Int<LlamaConfig::NUM_HEADS>{}),
                LayoutRight{}
            );
        }
        
        // Attention LSE intermediates: [NUM_HEADS, max_splits]
        __device__ __forceinline__ auto get_attn_lse_intermediates() const {
            return make_tensor(
                make_gmem_ptr(attn_lse_ptr),
                make_shape(Int<LlamaConfig::NUM_HEADS>{}, max_splits),
                LayoutRight{}
            );
        }
        
        // Final attention output: [NUM_HEADS, HEAD_DIM] = [HIDDEN_DIM]
        __device__ __forceinline__ auto get_attn_out() const {
            return make_tensor(
                make_gmem_ptr(attn_out_ptr),
                make_shape(Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }
        
        // Attention output intermediates: [NUM_HEADS, max_splits, HEAD_DIM]
        __device__ __forceinline__ auto get_attn_out_intermediates() const {
            return make_tensor(
                make_gmem_ptr(attn_out_inter_ptr),
                make_shape(Int<LlamaConfig::NUM_HEADS>{}, max_splits, Int<LlamaConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }
    };
}
