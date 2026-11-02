#pragma once

#include "kittens.cuh"
#include "kernel_config.cuh"
#include "llama_config.cuh"

namespace llama_mb {

struct globals {
    using bf16 = kittens::bf16;
    using weight_tile = kittens::st_bf<64, K_TILE>;
    using activation_tile = kittens::st_bf<BATCH_CAPACITY, K_TILE>;
    // Hopper's N8 WGMMA consumes only the first eight rows, but TK shared
    // descriptors retain the native 16-row swizzled tile used by the proven
    // static kernel.  The upper eight rows are padding.
    using activation_n8_tile = kittens::st_bf<16, K_TILE>;

    using instruction_t =
        kittens::gl<int, 1, -1, -1, h100_mb16_config::INSTRUCTION_WIDTH>;
    using barrier_t = kittens::gl<uint, 1, NUM_LAYERS, 10, 48>;

    using hidden_t = kittens::gl<bf16, 1, 1, BATCH_CAPACITY, HIDDEN_DIM,
                                 activation_tile>;
    using hidden_half_t =
        kittens::gl<bf16, 1, 1, -1, HIDDEN_DIM, activation_n8_tile>;
    using intermediate_t =
        kittens::gl<bf16, 1, 1, BATCH_CAPACITY, INTERMEDIATE_DIM,
                    activation_tile>;
    using logits_t = kittens::gl<bf16, 1, 1, BATCH_CAPACITY, VOCAB_SIZE>;

    template <int K>
    using layered_weight_t =
        kittens::gl<bf16, 1, NUM_LAYERS, -1, K, weight_tile>;
    using lm_head_weight_t =
        kittens::gl<bf16, 1, 1, VOCAB_SIZE, HIDDEN_DIM, weight_tile>;
    using norm_t = kittens::gl<bf16, 1, 1, NUM_LAYERS, HIDDEN_DIM>;
    using final_norm_t = kittens::gl<bf16, 1, 1, 1, HIDDEN_DIM>;
    using rope_t = kittens::gl<float, 1, 1, -1, HEAD_DIM>;

    // Python presents [L,B,S,KVH,D] as [L,B*S,KVH,D] because TK's GL ABI is
    // four-dimensional.  Indexing below reconstructs the logical B/S axes.
    using kv_t =
        kittens::gl<bf16, NUM_LAYERS, -1, NUM_KV_HEADS, HEAD_DIM>;
    using kv_decode_t =
        kittens::gl<bf16, NUM_LAYERS, -1, -1, HEAD_DIM,
                    kittens::st_bf<64, HEAD_DIM>>;
    using down_partials_t =
        kittens::gl<float, 1, DOWN_SPLIT_K, BATCH_CAPACITY, HIDDEN_DIM>;
    using control_i64_t = kittens::gl<long, 1, 1, 1, -1>;
    using control_u8_t = kittens::gl<uint8_t, 1, 1, 1, -1>;

    instruction_t instructions;
    hidden_t hidden_states;
    hidden_t q_post_rope;
    kv_t k_cache;
    kv_t v_cache;
    kv_decode_t k_cache_decode;
    kv_decode_t v_cache_decode;
    hidden_t attn_out;
    intermediate_t silu_out;
    logits_t logits;
    hidden_t attn_rms_out;
    hidden_t mlp_rms_out;
    hidden_t lm_head_rms_out;
    hidden_half_t attn_rms_out_low;
    hidden_half_t attn_rms_out_high;
    hidden_half_t attn_out_low;
    hidden_half_t attn_out_high;
    down_partials_t down_partials;

    norm_t attn_norm_weights;
    layered_weight_t<HIDDEN_DIM> qkv_weights;
    rope_t rope_cos;
    rope_t rope_sin;
    layered_weight_t<HIDDEN_DIM> o_proj_weights;
    norm_t mlp_norm_weights;
    layered_weight_t<HIDDEN_DIM> up_proj_weights;
    layered_weight_t<HIDDEN_DIM> gate_proj_weights;
    layered_weight_t<INTERMEDIATE_DIM> down_proj_weights;
    final_norm_t lm_head_norm_weights;
    lm_head_weight_t lm_head_weights;
    barrier_t Bar;

    control_i64_t position_ids;
    control_u8_t active_mask;
    int batch_size;
    int max_seq_len;
    float attn_scale;
    float rms_norm_eps;

    dim3 grid() const { return dim3(H100_SM_COUNT); }
    dim3 block() const { return dim3(h100_mb16_config::NUM_THREADS); }
    int dynamic_shared_memory() const {
        return h100_mb16_config::DYNAMIC_SHARED_MEMORY;
    }

    __device__ inline bool row_active(int b) const {
        return b >= 0 && b < batch_size && active_mask.raw_ptr[b] != 0;
    }
    __device__ inline int position(int b) const {
        return static_cast<int>(position_ids.raw_ptr[b]);
    }
    __device__ inline long kv_index(int layer, int b, int pos, int head,
                                    int d) const {
        return (((long)layer * batch_size + b) * max_seq_len + pos) *
                   NUM_KV_HEADS * HEAD_DIM +
               head * HEAD_DIM + d;
    }
};

}  // namespace llama_mb
