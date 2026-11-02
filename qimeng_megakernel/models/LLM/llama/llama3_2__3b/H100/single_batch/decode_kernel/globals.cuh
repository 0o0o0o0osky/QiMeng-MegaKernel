#pragma once

#include "kittens.cuh"
#include "megakernel.cuh"
#include "kernel_config.cuh"

// ========== Model / shape constants (Llama 3.2 3B) ==========
#define LLAMA_3B_NUM_LAYERS 28
#define LLAMA_3B_HIDDEN_DIM 3072
#define LLAMA_3B_INTERMEDIATE_DIM 8192
#define LLAMA_3B_HEAD_DIM 128
#define LLAMA_3B_NUM_ATTENTION_HEADS 24
#define LLAMA_3B_NUM_KV_HEADS 8
#define LLAMA_3B_KV_BLOCK_SIZE 16
#define LLAMA_3B_MATVEC_BLOCK_SIZE 16
#define LLAMA_3B_LM_HEAD_BLOCK_SIZE 32
#define LLAMA_3B_VOCAB_SIZE 128256
#define LLAMA_3B_ATTN_INTERMEDIATE_ALIGNMENT 16
#define LLAMA_3B_ATTN_INTERMEDIATE_COLS_FOR_SM_COUNT(SM_COUNT)             \
    ((((SM_COUNT) + LLAMA_3B_ATTN_INTERMEDIATE_ALIGNMENT - 1) /           \
      LLAMA_3B_ATTN_INTERMEDIATE_ALIGNMENT) *                              \
     LLAMA_3B_ATTN_INTERMEDIATE_ALIGNMENT)
// ========== UPGATE -> DOWNPROJ ragged-bucket dependency ==============
// Three DownProj K windows: 3072 + 3072 + 2048.  UpGate
// publishes every 16-element output block into the slot for the segment that
// consumes it.  The exact accounting is [192, 192, 128], not one device-wide 512
// count and not a uniform 128 launch threshold.  A per-slot count only proves
// that many arbitrary producers completed; waiting for the full bucket is what
// proves every element in the consumed segment is ready.
#define LLAMA_3B_DOWNPROJ_SPLIT_K_GRANULARITY LLAMA_3B_HIDDEN_DIM
#define LLAMA_3B_UPGATE_OUTPUT_BLOCKS                                      \
    (LLAMA_3B_INTERMEDIATE_DIM / LLAMA_3B_MATVEC_BLOCK_SIZE)
#define LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET                             \
    (LLAMA_3B_DOWNPROJ_SPLIT_K_GRANULARITY /                               \
     LLAMA_3B_MATVEC_BLOCK_SIZE)
#define LLAMA_3B_UPGATE_SIGNAL_BUCKETS                                     \
    ((LLAMA_3B_UPGATE_OUTPUT_BLOCKS +                                      \
      LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET - 1) /                        \
     LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET)
#define LLAMA_3B_UPGATE_LAST_BUCKET_BLOCKS                                 \
    (LLAMA_3B_UPGATE_OUTPUT_BLOCKS -                                       \
     (LLAMA_3B_UPGATE_SIGNAL_BUCKETS - 1) *                                \
         LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET)
#define LLAMA_3B_UPGATE_BUCKET_EXPECTED(BUCKET_IDX)                        \
    (((BUCKET_IDX) + 1 < LLAMA_3B_UPGATE_SIGNAL_BUCKETS)                   \
         ? LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET                          \
         : LLAMA_3B_UPGATE_LAST_BUCKET_BLOCKS)

// Each of the three DownProj splits covers every hidden output tile.  QKV and
// LMHead wait for this dense successor accounting before reading hidden_states.
#define LLAMA_3B_DOWNPROJ_BARRIER_COUNT                                    \
    ((LLAMA_3B_HIDDEN_DIM / LLAMA_3B_MATVEC_BLOCK_SIZE) *                  \
     LLAMA_3B_UPGATE_SIGNAL_BUCKETS)

static_assert(LLAMA_3B_INTERMEDIATE_DIM % LLAMA_3B_MATVEC_BLOCK_SIZE == 0,
              "UpGate must publish whole 16-element blocks");
static_assert(LLAMA_3B_DOWNPROJ_SPLIT_K_GRANULARITY %
                      LLAMA_3B_MATVEC_BLOCK_SIZE ==
                  0,
              "a signal block must not straddle DownProj segments");
static_assert(LLAMA_3B_UPGATE_SIGNAL_BUCKETS == 3,
              "Llama3B H100 DownProj must retain three splits");
static_assert(LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET == 192 &&
                      LLAMA_3B_UPGATE_LAST_BUCKET_BLOCKS == 128,
              "Llama3B H100 UpGate accounting must be [192, 192, 128]");
static_assert((LLAMA_3B_UPGATE_SIGNAL_BUCKETS - 1) *
                              LLAMA_3B_UPGATE_BLOCKS_PER_FULL_BUCKET +
                          LLAMA_3B_UPGATE_LAST_BUCKET_BLOCKS ==
                      LLAMA_3B_UPGATE_OUTPUT_BLOCKS,
              "ragged bucket arrivals must conserve all 512 UpGate blocks");
static_assert(LLAMA_3B_DOWNPROJ_BARRIER_COUNT == 576,
              "three DownProj splits must publish 576 output tiles");

static_assert(LLAMA_3B_ATTN_INTERMEDIATE_COLS_FOR_SM_COUNT(H100_SM_COUNT) ==
                  144,
              "H100 attention scratch ABI must use 144 physical columns");

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

// ========== Globals (tensor shapes and buffers) ==========
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
    constexpr static int attn_intermediate_cols =
        LLAMA_3B_ATTN_INTERMEDIATE_COLS_FOR_SM_COUNT(sm_count);

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
    using kv_cache_t = kittens::gl<kittens::bf16, -1, -1, -1, head_dim, kittens::sv_bf<matvec_block_size>,
                          kittens::tma::descriptor<kittens::st_bf<kv_block_size, head_dim>, 1>>;

    using attn_out_intermediates_t =
        kittens::gl<float, 1, num_attention_heads, -1, head_dim, kittens::sv_fl<head_dim>>;
    using attn_lse_intermediates_t = kittens::gl<float, 1, 1, num_attention_heads, -1,
                                        kittens::sv_fl<attn_intermediate_cols>>;

    using barriers =
        kittens::gl<uint, 1, -1, -1, num_attention_heads + 2 * num_kv_heads>;

    // Member order must match A100 mk() / Python call order for bind_kernel.
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

typedef globals_t<LLAMA_3B_NUM_LAYERS, LLAMA_3B_HIDDEN_DIM,
                  LLAMA_3B_INTERMEDIATE_DIM, LLAMA_3B_HEAD_DIM,
                  LLAMA_3B_NUM_ATTENTION_HEADS, LLAMA_3B_NUM_KV_HEADS,
                  LLAMA_3B_KV_BLOCK_SIZE, LLAMA_3B_MATVEC_BLOCK_SIZE,
#ifndef KITTENS_BLACKWELL
                  H100_SM_COUNT>
#else
                  B200_SM_COUNT>
#endif
    llama_3b_globals;
