#pragma once
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cstdint>

#include "llama_config.cuh"

namespace megakernel {
    using namespace cute;
    using bfloat16_t = cutlass::bfloat16_t;

    struct LlamaGlobals {
        // -----------------------------------------------------------------------
        // Instruction / sync
        // -----------------------------------------------------------------------
        int32_t*    instruction_ptr;
        int32_t*    barrier_ptr;

        // -----------------------------------------------------------------------
        // Runtime state
        // -----------------------------------------------------------------------
        int32_t     batch_size;     // actual batch rows (2..16)
        // Scalar decode index for all rows when position_ids_ptr == nullptr (tests, lockstep).
        uint32_t    pos_id;
        // Per-row decode indices [batch_size], int32 contiguous on device; nullptr => use pos_id.
        int32_t*    position_ids_ptr;
        // Per-row predication: nullptr => all active; else nonzero => active (writes q_post_rope/KV).
        uint8_t*    active_mask_ptr;
        int32_t     num_insts;
        int32_t     max_seq_len;
        int32_t     num_layer_barriers;
        int32_t     max_splits;
        // Per-batch barrier stride: NUM_Q_HEADS + 2*NUM_KV_HEADS = 48.
        // barriers last dim = barrier_stride * batch_size.
        int32_t     barrier_stride;

        float       attn_scale;
        float       rms_norm_eps;

        // -----------------------------------------------------------------------
        // Activation / output buffers  (multi-batch: 2-D pointers)
        // -----------------------------------------------------------------------
        bfloat16_t* hidden_state_ptr;   // [batch_size, HIDDEN_DIM]
        // Instance-selected RMS staging before QKV, Up/Gate, and LM-head GEMM.
        bfloat16_t* attn_rms_out_ptr;   // [batch_size, HIDDEN_DIM]
        bfloat16_t* mlp_rms_out_ptr;    // [batch_size, HIDDEN_DIM]
        bfloat16_t* lm_head_rms_out_ptr; // [batch_size, HIDDEN_DIM]
        bfloat16_t* q_post_rope_ptr;    // [batch_size, HIDDEN_DIM]
        bfloat16_t* attn_out_ptr;       // [batch_size, HIDDEN_DIM]
        bfloat16_t* silu_out_ptr;       // [batch_size, INTERM_DIM]
        float*      attn_lse_ptr;       // [NUM_HEADS, max_splits]  (per-head per-split)
        float*      attn_out_inter_ptr; // [NUM_HEADS, max_splits, HEAD_DIM]
        uint8_t*    attn_partial_valid_ptr; // [batch_size, NUM_Q_HEADS, max_splits]
        bfloat16_t* logits_ptr;         // [VOCAB_SIZE]

        // -----------------------------------------------------------------------
        // Weights
        // -----------------------------------------------------------------------
        bfloat16_t* attn_norm_ptr;      // [NUM_LAYERS, HIDDEN_DIM]
        bfloat16_t* qkv_weight_ptr;     // [NUM_LAYERS, QKV_OUT_DIM, HIDDEN_DIM]
        bfloat16_t* o_weight_ptr;       // [NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM]
        bfloat16_t* mlp_norm_ptr;       // [NUM_LAYERS, HIDDEN_DIM]
        bfloat16_t* up_weight_ptr;      // [NUM_LAYERS, INTERM_DIM, HIDDEN_DIM]
        bfloat16_t* gate_weight_ptr;    // [NUM_LAYERS, INTERM_DIM, HIDDEN_DIM]
        bfloat16_t* down_weight_ptr;    // [NUM_LAYERS, HIDDEN_DIM, INTERM_DIM]
        bfloat16_t* lm_head_norm_ptr;   // [HIDDEN_DIM]
        bfloat16_t* lm_head_weight_ptr; // [VOCAB_SIZE, HIDDEN_DIM]

        float*      rope_cos_ptr;       // [max_seq_len, HEAD_DIM]
        float*      rope_sin_ptr;       // [max_seq_len, HEAD_DIM]

        // -----------------------------------------------------------------------
        // KV cache  [NUM_LAYERS, batch_size, max_seq_len, KV_HEADS, HEAD_DIM]
        // -----------------------------------------------------------------------
        bfloat16_t* k_cache_ptr;
        bfloat16_t* v_cache_ptr;

        // -----------------------------------------------------------------------
        // Optional debug buffer (null in production; allocated by test when needed)
        // -----------------------------------------------------------------------
        float*      debug_buf;

#if defined(ENABLE_PROFILING) || defined(ENABLE_BOUNDARY_PROFILING)
        int64_t*    prof_ptr;
        int32_t     prof_max_insts;
#endif

        // =======================================================================
        // Tensor accessors
        // =======================================================================

        // Hidden state: [batch_size, HIDDEN_DIM]
        __device__ __forceinline__ auto get_hidden_state() const {
            return make_tensor(
                make_gmem_ptr(hidden_state_ptr),
                make_shape(batch_size, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Attention RMS output (post-norm), consumed by QKV GEMM: [batch_size, HIDDEN_DIM]
        __device__ __forceinline__ auto get_attn_rms_out() const {
            return make_tensor(
                make_gmem_ptr(attn_rms_out_ptr),
                make_shape(batch_size, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // MLP RMS output (post-norm), consumed by UpGate GEMM: [batch_size, HIDDEN_DIM]
        __device__ __forceinline__ auto get_mlp_rms_out() const {
            return make_tensor(
                make_gmem_ptr(mlp_rms_out_ptr),
                make_shape(batch_size, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Final RMS output (post-norm), consumed by LM-head GEMM: [batch_size, HIDDEN_DIM]
        __device__ __forceinline__ auto get_lm_head_rms_out() const {
            return make_tensor(
                make_gmem_ptr(lm_head_rms_out_ptr),
                make_shape(batch_size, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Q after RoPE: [batch_size, HIDDEN_DIM]
        __device__ __forceinline__ auto get_q_post_rope() const {
            return make_tensor(
                make_gmem_ptr(q_post_rope_ptr),
                make_shape(batch_size, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Attn norm weights: [NUM_LAYERS, HIDDEN_DIM]
        __device__ __forceinline__ auto get_attn_norm() const {
            return make_tensor(
                make_gmem_ptr(attn_norm_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // QKV weights: [NUM_LAYERS, QKV_OUT_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_qkv_weight() const {
            return make_tensor(
                make_gmem_ptr(qkv_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           Int<LlamaConfig::QKV_OUT_DIM>{},
                           Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // O projection weights: [NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_o_weight() const {
            return make_tensor(
                make_gmem_ptr(o_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           Int<LlamaConfig::HIDDEN_DIM>{},
                           Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // MLP norm weights: [NUM_LAYERS, HIDDEN_DIM]
        __device__ __forceinline__ auto get_mlp_norm() const {
            return make_tensor(
                make_gmem_ptr(mlp_norm_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{}, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Up projection weights: [NUM_LAYERS, INTERM_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_up_weight() const {
            return make_tensor(
                make_gmem_ptr(up_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           Int<LlamaConfig::INTERM_DIM>{},
                           Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Gate projection weights: [NUM_LAYERS, INTERM_DIM, HIDDEN_DIM]
        __device__ __forceinline__ auto get_gate_weight() const {
            return make_tensor(
                make_gmem_ptr(gate_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           Int<LlamaConfig::INTERM_DIM>{},
                           Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Down projection weights: [NUM_LAYERS, HIDDEN_DIM, INTERM_DIM]
        __device__ __forceinline__ auto get_down_weight() const {
            return make_tensor(
                make_gmem_ptr(down_weight_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           Int<LlamaConfig::HIDDEN_DIM>{},
                           Int<LlamaConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        // Final RMS norm weights: [HIDDEN_DIM]
        __device__ __forceinline__ auto get_lm_head_norm() const {
            return make_tensor(
                make_gmem_ptr(lm_head_norm_ptr),
                make_shape(Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // LM head weights: [VOCAB_SIZE, HIDDEN_DIM]
        __device__ __forceinline__ auto get_lm_head_weight() const {
            return make_tensor(
                make_gmem_ptr(lm_head_weight_ptr),
                make_shape(Int<LlamaConfig::VOCAB_SIZE>{},
                           Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // RoPE cos/sin: [max_seq_len, HEAD_DIM]
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

        // KV cache: [NUM_LAYERS, batch_size, max_seq_len, KV_HEADS, HEAD_DIM]
        __device__ __forceinline__ auto get_k_cache() const {
            return make_tensor(
                make_gmem_ptr(k_cache_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           batch_size,
                           max_seq_len,
                           Int<LlamaConfig::NUM_KV_HEADS>{},
                           Int<LlamaConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }
        __device__ __forceinline__ auto get_v_cache() const {
            return make_tensor(
                make_gmem_ptr(v_cache_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           batch_size,
                           max_seq_len,
                           Int<LlamaConfig::NUM_KV_HEADS>{},
                           Int<LlamaConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        // Barriers: [NUM_LAYERS, num_layer_barriers, barrier_stride * batch_size]
        // Uses the actual last-dim size so multi-layer strides are correct.
        __device__ __forceinline__ auto get_barrier() const {
            return make_tensor(
                make_gmem_ptr(barrier_ptr),
                make_shape(Int<LlamaConfig::NUM_LAYERS>{},
                           num_layer_barriers,
                           (int)(barrier_stride * batch_size)),
                LayoutRight{}
            );
        }

        // Instructions: [num_insts, 32]  (per-SM slice already applied before passing)
        __device__ __forceinline__ auto get_instruction() const {
            return make_tensor(
                make_gmem_ptr(instruction_ptr),
                make_shape(num_insts, Int<32>{}),
                LayoutRight{}
            );
        }

        // Attention LSE intermediate view shared with the single-batch layout:
        // [NUM_HEADS, max_splits].
        __device__ __forceinline__ auto get_attn_lse_intermediates() const {
            return make_tensor(
                make_gmem_ptr(attn_lse_ptr),
                make_shape(Int<LlamaConfig::NUM_HEADS>{}, max_splits),
                LayoutRight{}
            );
        }

        // Multi-batch LSE: [batch_size, NUM_Q_HEADS, max_splits]
        __device__ __forceinline__ auto get_attn_lse_inter_mb() const {
            return make_tensor(
                make_gmem_ptr(attn_lse_ptr),
                make_shape(batch_size,
                           Int<LlamaConfig::NUM_Q_HEADS>{},
                           max_splits),
                LayoutRight{}
            );
        }

        // Multi-batch O intermediates: [batch_size, NUM_Q_HEADS, max_splits, HEAD_DIM]
        __device__ __forceinline__ auto get_attn_out_inter_mb() const {
            return make_tensor(
                make_gmem_ptr(attn_out_inter_ptr),
                make_shape(batch_size,
                           Int<LlamaConfig::NUM_Q_HEADS>{},
                           max_splits,
                           Int<LlamaConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        // Multi-batch partial validity: [batch_size, NUM_Q_HEADS, max_splits]
        __device__ __forceinline__ auto get_attn_partial_valid_mb() const {
            return make_tensor(
                make_gmem_ptr(attn_partial_valid_ptr),
                make_shape(batch_size,
                           Int<LlamaConfig::NUM_Q_HEADS>{},
                           max_splits),
                LayoutRight{}
            );
        }

        // Final attention output: [batch_size, HIDDEN_DIM]
        __device__ __forceinline__ auto get_attn_out() const {
            return make_tensor(
                make_gmem_ptr(attn_out_ptr),
                make_shape(batch_size, Int<LlamaConfig::HIDDEN_DIM>{}),
                LayoutRight{}
            );
        }

        // Multi-batch attn output viewed as [batch_size, NUM_Q_HEADS, HEAD_DIM]
        __device__ __forceinline__ auto get_attn_out_mb() const {
            return make_tensor(
                make_gmem_ptr(attn_out_ptr),
                make_shape(batch_size,
                           Int<LlamaConfig::NUM_Q_HEADS>{},
                           Int<LlamaConfig::HEAD_DIM>{}),
                LayoutRight{}
            );
        }

        // SiLU output: [batch_size, INTERM_DIM]
        __device__ __forceinline__ auto get_silu_out() const {
            return make_tensor(
                make_gmem_ptr(silu_out_ptr),
                make_shape(batch_size, Int<LlamaConfig::INTERM_DIM>{}),
                LayoutRight{}
            );
        }

        // Final logits: [batch_size, VOCAB_SIZE]
        __device__ __forceinline__ auto get_logits() const {
            return make_tensor(
                make_gmem_ptr(logits_ptr),
                make_shape(batch_size, Int<LlamaConfig::VOCAB_SIZE>{}),
                LayoutRight{}
            );
        }

        // Volatile pointer to a specific barrier slot for a given batch row.
        // barriers layout: [NUM_LAYERS, num_layer_barriers, barrier_stride * batch_size]
        // barrier_stride = NUM_Q_HEADS + 2*NUM_KV_HEADS = 48.
        __device__ __forceinline__ volatile int32_t* barrier_for_batch(
            int layer, int row, int batch_idx, int head_offset
        ) const {
            long flat_last = (long)barrier_stride * batch_size;
            long idx = (long)layer * num_layer_barriers * flat_last
                     + (long)row   * flat_last
                     + (long)batch_idx * barrier_stride
                     + head_offset;
            return reinterpret_cast<volatile int32_t*>(barrier_ptr + idx);
        }

        // Global (non-per-batch) barrier pointer: barriers[layer, row, head_offset].
        // Used for OPCODE_QKV_GEMM_ROPE barriers which are written once for all
        // batches by the QKV CTA (which processes every batch row in one sweep).
        __device__ __forceinline__ volatile int32_t* barrier_global(
            int layer, int row, int head_offset
        ) const {
            long flat_last = (long)barrier_stride * batch_size;
            long idx = (long)layer * num_layer_barriers * flat_last
                     + (long)row   * flat_last
                     + head_offset;
            return reinterpret_cast<volatile int32_t*>(barrier_ptr + idx);
        }
    };

    /** RoPE / KV slot index for batch row m (PyVM decode_pos_id_for_batch). */
    __device__ __forceinline__ int32_t row_position(LlamaGlobals const& g, int m) {
        if (g.position_ids_ptr != nullptr) {
            return g.position_ids_ptr[m];
        }
        return static_cast<int32_t>(g.pos_id);
    }

    /** If active_mask_ptr set, only nonzero rows participate in epilogue writes. */
    __device__ __forceinline__ bool row_active(LlamaGlobals const& g, int m) {
        if (g.active_mask_ptr == nullptr) {
            return true;
        }
        return g.active_mask_ptr[m] != static_cast<uint8_t>(0);
    }

}  // namespace megakernel
