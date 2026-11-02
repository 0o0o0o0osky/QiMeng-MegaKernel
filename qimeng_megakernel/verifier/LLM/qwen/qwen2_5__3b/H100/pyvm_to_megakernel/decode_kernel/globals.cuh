#pragma once

#include "kittens.cuh"
#include "megakernel.cuh"
#include "kernel_config.cuh"

// ========== Model / shape constants (Qwen2.5 3B, same layout as LLaMA 3B for TMA) ==========
// hidden_dim (hidden_states size) drives NUM_CONSUMER_WARPS: 2048 -> 16, 3072 -> 24 (all ops unified).
#define QWEN_3B_NUM_LAYERS 36
#define QWEN_3B_HIDDEN_DIM 2048
#define QWEN_3B_INTERMEDIATE_DIM 11008
#define QWEN_3B_HEAD_DIM 128
#define QWEN_3B_NUM_ATTENTION_HEADS 16
#define QWEN_3B_NUM_KV_HEADS 2
#define QWEN_3B_KV_BLOCK_SIZE 16
#define QWEN_3B_MATVEC_BLOCK_SIZE 16
#define QWEN_3B_LM_HEAD_BLOCK_SIZE 32
#define QWEN_3B_VOCAB_SIZE 151936
#define QWEN_3B_DOWNPROJ_SPLIT_K_GRANULARITY 2048
#define QWEN_3B_UPGATE_OUTPUT_BLOCKS \
    (QWEN_3B_INTERMEDIATE_DIM / QWEN_3B_MATVEC_BLOCK_SIZE)
#define QWEN_3B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET \
    (QWEN_3B_DOWNPROJ_SPLIT_K_GRANULARITY / QWEN_3B_MATVEC_BLOCK_SIZE)
#define QWEN_3B_UPGATE_SIGNAL_BUCKETS \
    ((QWEN_3B_UPGATE_OUTPUT_BLOCKS + QWEN_3B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET - 1) / \
     QWEN_3B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET)
#define QWEN_3B_UPGATE_LAST_SIGNAL_BUCKET_BLOCKS \
    (QWEN_3B_UPGATE_OUTPUT_BLOCKS - \
     (QWEN_3B_UPGATE_SIGNAL_BUCKETS - 1) * \
         QWEN_3B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET)
#define QWEN_3B_UPGATE_SIGNAL_BUCKET_EXPECTED(BUCKET_IDX) \
    (((BUCKET_IDX) + 1 < QWEN_3B_UPGATE_SIGNAL_BUCKETS) \
         ? QWEN_3B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET \
         : QWEN_3B_UPGATE_LAST_SIGNAL_BUCKET_BLOCKS)
#define QWEN_3B_DOWNPROJ_BARRIER_COUNT \
    ((QWEN_3B_HIDDEN_DIM / QWEN_3B_MATVEC_BLOCK_SIZE) * \
     QWEN_3B_UPGATE_SIGNAL_BUCKETS)

static_assert(QWEN_3B_INTERMEDIATE_DIM % QWEN_3B_MATVEC_BLOCK_SIZE == 0,
              "upgate output must contain whole signal blocks");
static_assert(QWEN_3B_DOWNPROJ_SPLIT_K_GRANULARITY %
                      QWEN_3B_MATVEC_BLOCK_SIZE == 0,
              "an upgate output block must not straddle signal buckets");
static_assert(QWEN_3B_UPGATE_SIGNAL_BUCKETS == 6,
              "Qwen2.5-3B uses five full split-K buckets plus one tail");
static_assert(QWEN_3B_UPGATE_LAST_SIGNAL_BUCKET_BLOCKS == 48,
              "the 768-element tail must publish 48 blocks");
static_assert((QWEN_3B_UPGATE_SIGNAL_BUCKETS - 1) *
                          QWEN_3B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET +
                      QWEN_3B_UPGATE_LAST_SIGNAL_BUCKET_BLOCKS ==
                  QWEN_3B_UPGATE_OUTPUT_BLOCKS,
              "per-bucket arrivals must conserve every upgate output block");

// ========== Semaphore / slot layout (matches MegaKernel controller, same as LLaMA) ==========
constexpr int ATOMIC_ADD_START = megakernel::FREE_SLOTS_START;
constexpr int ATOMIC_ADD_END = ATOMIC_ADD_START + 1;
constexpr int EPILOGUE_START = ATOMIC_ADD_END + 1;
constexpr int ACT_WAIT_DONE = EPILOGUE_START + 2;
constexpr int WEIGHT_WAIT_START = ACT_WAIT_DONE + 3;
constexpr int WEIGHT_WAIT_DONE = WEIGHT_WAIT_START + 4;
constexpr int RMS_START = WEIGHT_WAIT_DONE + 4;
constexpr int RMS_SCALE_WAIT_START = RMS_START + 1;
constexpr int RMS_SCALE_WAIT_DONE = RMS_SCALE_WAIT_START + 1;
constexpr int RMS_DONE = RMS_SCALE_WAIT_DONE + 1;

constexpr int TEMP1 = RMS_DONE + 1;
constexpr int TEMP2 = TEMP1 + 1;
constexpr int TEMP3 = TEMP2 + 1;
constexpr int TEMP4 = TEMP3 + 1;
constexpr int TEMP5 = TEMP4 + 1;
constexpr int TEMP6 = TEMP5 + 1;

using config = megakernel::default_config;

// ========== Globals: same type definitions as LLaMA 3B (kv_cache_t, attn_*_intermediates_t) for TMA ==========
template <int _num_layers, int _hidden_dim, int _intermediate_dim,
          int _head_dim, int _num_attention_heads, int _num_kv_heads,
          int _kv_block_size, int _matvec_block_size, int _sm_count>
struct globals_t {

    constexpr static int num_layers = _num_layers;
    constexpr static int matvec_block_size = _matvec_block_size;
    constexpr static int kv_block_size = _kv_block_size;
    constexpr static int head_dim = _head_dim;
    constexpr static int hidden_dim = _hidden_dim;
    constexpr static int intermediate_dim = _intermediate_dim;
    constexpr static int num_attention_heads = _num_attention_heads;
    constexpr static int num_kv_heads = _num_kv_heads;
    constexpr static int sm_count = _sm_count;

    using instruction_layout =
        megakernel::instruction_layout<config>;

    using weights_t =
        kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,
           kittens::st_bf<matvec_block_size, 512>>;
    using weights_big_indim_t =
        kittens::gl<kittens::bf16, 1, -1, -1, intermediate_dim,
           kittens::st_bf<matvec_block_size, 512>>;

    using activations_t = kittens::gl<kittens::bf16, 1, 1, 1, hidden_dim, kittens::sv_bf<hidden_dim>,
                             kittens::sv_bf<head_dim>, kittens::sv_bf<matvec_block_size>>;
    using activations_big_indim_t =
        kittens::gl<kittens::bf16, 1, 1, 1, intermediate_dim, kittens::sv_bf<intermediate_dim>,
           kittens::sv_bf<hidden_dim>, kittens::sv_bf<matvec_block_size>>;
    using logits_t = kittens::gl<kittens::bf16, 1, 1, 1, -1, kittens::sv_bf<matvec_block_size>>;

    using norm_weights_t = kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim, kittens::sv_bf<hidden_dim>,
                              kittens::sv_bf<matvec_block_size>>;
    using rope_table_t = kittens::gl<float, 1, 1, -1, head_dim, kittens::sv_fl<head_dim>>;
    // Same as LLaMA 3B: TMA descriptor for 16 x head_dim tile
    using kv_cache_t = kittens::gl<kittens::bf16, -1, -1, -1, head_dim, kittens::sv_bf<matvec_block_size>,
                          kittens::tma::descriptor<kittens::st_bf<kv_block_size, head_dim>, 1>>;

    using attn_out_intermediates_t =
        kittens::gl<float, 1, num_attention_heads, -1, head_dim, kittens::sv_fl<head_dim>>;
    using attn_lse_intermediates_t = kittens::gl<float, 1, 1, num_attention_heads, -1,
                                        kittens::sv_fl<((sm_count + 15) / 16) * 16>>;

    using barriers =
        kittens::gl<uint, 1, -1, -1, num_attention_heads + 2 * num_kv_heads>;

    // Qwen-only: QKV bias (same as models)
    static constexpr int kv_hidden_dim = num_kv_heads * head_dim;
    static constexpr int QKV_DIM = hidden_dim + 2 * kv_hidden_dim;
    static constexpr int QKV_NUM_BLOCKS = QKV_DIM / 16;
    using bias_qkv_concat_t = kittens::gl<kittens::bf16, 1, -1, -1, 16, kittens::sv_bf<16>>;

    // Member order must match mk.cu bind_kernel / Python call order.

    instruction_layout instructions;
    activations_t hidden_states;
    activations_t q_post_rope;
    kv_cache_t k_cache;
    kv_cache_t v_cache;
    activations_t attn_out;
    attn_lse_intermediates_t attn_lse_intermediates;
    attn_out_intermediates_t attn_out_intermediates;
    activations_big_indim_t silu_out;
    logits_t logits;
    norm_weights_t attn_norm_weights;
    weights_t qkv_weights;
    rope_table_t rope_cos;
    rope_table_t rope_sin;
    bias_qkv_concat_t qkv_bias;
    weights_t o_weights;
    norm_weights_t mlp_norm_weights;
    weights_t up_weights;
    weights_t gate_weights;
    weights_big_indim_t down_weights;
    norm_weights_t lm_head_norm_weights;
    weights_t lm_head_weights;
    barriers Bar;
    unsigned int pos_id;
    float attn_scale;
    float rms_norm_eps;

    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

typedef globals_t<QWEN_3B_NUM_LAYERS, QWEN_3B_HIDDEN_DIM,
                  QWEN_3B_INTERMEDIATE_DIM, QWEN_3B_HEAD_DIM,
                  QWEN_3B_NUM_ATTENTION_HEADS, QWEN_3B_NUM_KV_HEADS,
                  QWEN_3B_KV_BLOCK_SIZE, QWEN_3B_MATVEC_BLOCK_SIZE,
#ifndef KITTENS_BLACKWELL
                  H100_SM_COUNT>
#else
                  B200_SM_COUNT>
#endif
    qwen_2_5_3b_globals;
