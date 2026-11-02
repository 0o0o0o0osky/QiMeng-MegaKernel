#pragma once

#include "kittens.cuh"
#include "megakernel.cuh"
#include "kernel_config.cuh"

// ========== Qwen3 0.6B model / shape constants ==========
#define QWEN3_0_6B_NUM_LAYERS 28
#define QWEN3_0_6B_HIDDEN_DIM 1024
#define QWEN3_0_6B_INTERMEDIATE_DIM 3072
#define QWEN3_0_6B_HEAD_DIM 128
#define QWEN3_0_6B_NUM_ATTENTION_HEADS 16
#define QWEN3_0_6B_NUM_KV_HEADS 8
#define QWEN3_0_6B_KV_BLOCK_SIZE 16
#define QWEN3_0_6B_MATVEC_BLOCK_SIZE 16
#define QWEN3_0_6B_LM_HEAD_BLOCK_SIZE 32
#define QWEN3_0_6B_VOCAB_SIZE 151936
// Oproj adds one per 16-element block; upgate waits for this (1024/16 = 64).
#define QWEN3_0_6B_OPROJ_BARRIER_COUNT (QWEN3_0_6B_HIDDEN_DIM / 16)
// UpGate -> DownProj has three independent 1024-element split-K buckets.
#define QWEN3_0_6B_UPGATE_SIGNAL_BUCKETS 3
#define QWEN3_0_6B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET (1024 / 16)
static_assert(QWEN3_0_6B_UPGATE_SIGNAL_BUCKETS *
                      QWEN3_0_6B_UPGATE_BLOCKS_PER_SIGNAL_BUCKET ==
                  QWEN3_0_6B_INTERMEDIATE_DIM / 16,
              "UpGate barrier buckets must cover every output block");
// DownProj split-K=3: each split contributes hidden_dim/16 block arrivals (total 192).
#define QWEN3_0_6B_DOWNPROJ_BARRIER_COUNT ((QWEN3_0_6B_HIDDEN_DIM / 16) * 3)

// ========== Semaphore / slot layout (matches MegaKernel controller) ==========
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

// ========== Globals: Qwen3 0.6B — no qkv_bias; q_norm_weights, k_norm_weights per head ==========
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

    // (1,1,1,1024): tile sv_bf<64> so warp::load(..., {warpid()}) loads 64 elements per warp (16 warps).
    using activations_t = kittens::gl<kittens::bf16, 1, 1, 1, hidden_dim, kittens::sv_bf<hidden_dim>,
                             kittens::sv_bf<head_dim>, kittens::sv_bf<matvec_block_size>, kittens::sv_bf<64>>;
    static constexpr int q_hidden_dim = num_attention_heads * head_dim;
    static constexpr int kv_hidden_dim = num_kv_heads * head_dim;
    static constexpr int QKV_DIM = q_hidden_dim + 2 * kv_hidden_dim;
    static constexpr int QKV_NUM_BLOCKS = QKV_DIM / 16;
    // Q output only: num_attention_heads * head_dim (e.g. 16*128 = 2048)
    static constexpr int q_post_rope_cols = num_attention_heads * head_dim;
    using q_post_rope_t = kittens::gl<kittens::bf16, 1, 1, 1, q_post_rope_cols, kittens::sv_bf<q_post_rope_cols>,
                             kittens::sv_bf<head_dim>, kittens::sv_bf<matvec_block_size>>;
    // o_proj weight: (hidden_dim, q_hidden_dim) = (1024, 2048)
    using weights_oproj_t =
        kittens::gl<kittens::bf16, 1, -1, -1, q_hidden_dim,
           kittens::st_bf<matvec_block_size, 512>>;
    static constexpr int qkv_raw_cols = QKV_DIM;
    using qkv_raw_t = kittens::gl<kittens::bf16, 1, 1, 1, qkv_raw_cols, kittens::sv_bf<matvec_block_size>>;
    using activations_big_indim_t =
        kittens::gl<kittens::bf16, 1, 1, 1, intermediate_dim, kittens::sv_bf<intermediate_dim>,
           kittens::sv_bf<hidden_dim>, kittens::sv_bf<matvec_block_size>>;
    using logits_t = kittens::gl<kittens::bf16, 1, 1, 1, -1, kittens::sv_bf<matvec_block_size>>;

    using norm_weights_t = kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim, kittens::sv_bf<hidden_dim>,
                              kittens::sv_bf<matvec_block_size>>;
    // Qwen3: per-head norm (L, head_dim) for q and k
    using head_norm_weights_t = kittens::gl<kittens::bf16, 1, 1, -1, head_dim, kittens::sv_bf<head_dim>>;
    using rope_table_t = kittens::gl<float, 1, 1, -1, head_dim, kittens::sv_fl<head_dim>>;
    using kv_cache_t = kittens::gl<kittens::bf16, -1, -1, -1, head_dim, kittens::sv_bf<matvec_block_size>,
                          kittens::tma::descriptor<kittens::st_bf<kv_block_size, head_dim>, 1>>;

    using attn_out_intermediates_t =
        kittens::gl<float, 1, num_attention_heads, -1, head_dim, kittens::sv_fl<head_dim>>;
    using attn_lse_intermediates_t = kittens::gl<float, 1, 1, num_attention_heads, -1,
        kittens::sv_fl<(((sm_count / num_kv_heads) + 15) / 16) * 16>>;

    using barriers =
        kittens::gl<uint, 1, -1, -1, num_attention_heads + 2 * num_kv_heads>;

    // Member order must match mk.cu bind_kernel / Python call order.
    // No qkv_bias; q_norm_weights/k_norm_weights used in Op2.
    // attn_out: 16 heads * head_dim = q_hidden_dim (2048), not hidden_dim (1024); reduction writes 128 per head.
    instruction_layout instructions;
    activations_t hidden_states;
    qkv_raw_t qkv_raw;
    q_post_rope_t q_post_rope;
    kv_cache_t k_cache;
    kv_cache_t v_cache;
    q_post_rope_t attn_out;
    attn_lse_intermediates_t attn_lse_intermediates;
    attn_out_intermediates_t attn_out_intermediates;
    activations_big_indim_t silu_out;
    logits_t logits;
    norm_weights_t attn_norm_weights;
    head_norm_weights_t q_norm_weights;
    head_norm_weights_t k_norm_weights;
    weights_t qkv_weights;
    rope_table_t rope_cos;
    rope_table_t rope_sin;
    weights_oproj_t o_weights;
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

typedef globals_t<QWEN3_0_6B_NUM_LAYERS, QWEN3_0_6B_HIDDEN_DIM,
                  QWEN3_0_6B_INTERMEDIATE_DIM, QWEN3_0_6B_HEAD_DIM,
                  QWEN3_0_6B_NUM_ATTENTION_HEADS, QWEN3_0_6B_NUM_KV_HEADS,
                  QWEN3_0_6B_KV_BLOCK_SIZE, QWEN3_0_6B_MATVEC_BLOCK_SIZE,
#ifndef KITTENS_BLACKWELL
                  H100_SM_COUNT>
#else
                  B200_SM_COUNT>
#endif
    qwen3_0_6b_globals;
