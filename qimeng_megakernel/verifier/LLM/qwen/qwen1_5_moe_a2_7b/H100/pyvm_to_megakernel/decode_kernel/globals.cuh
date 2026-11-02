#pragma once

#include "kittens.cuh"
#include "megakernel.cuh"
#include "kernel_config.cuh"

#include <cstdint>

// ============================================================================
// Qwen1.5-MoE-A2.7B / H100 MegaKernel ABI:
//     member order of `qwen_moe_globals`
//  == `&qwen_moe_globals::x` order in mk.cu's `kittens::py::bind_kernel<...>`
//  == `py::arg` order
//  == argument order in `decode_schedule/mk_kernel.py::call_mk`.
//
// Shape note: `pyutils.cuh::from_object` left-pads a torch shape with 1s to
// 4 dims, so a 1-D [2048] tensor arrives as {1,1,1,2048} and a 2-D
// [24, 6144] tensor as {1,1,24,6144}.  A pinned (positive) dim in the `gl<>`
// declaration is therefore a runtime shape assertion.  Pinned dimensions
// below are the model constants used by this route.
//
// TMA descriptor note: a `gl<>` only carries descriptors for the shared types
// listed in its template tail.  `kittens::warp::load/store` on a `sv_*` does
// uses plain global access; `kittens::tma::*_async` requires a descriptor.
// The template tails list the shared layouts used by this route's TMA sites.
// ============================================================================

// ---------------------------------------------------------------------------
// Model / shape constants
// ---------------------------------------------------------------------------
#define QWEN_MOE_NUM_LAYERS 24
#define QWEN_MOE_HIDDEN_DIM 2048
#define QWEN_MOE_QKV_DIM 6144 // Q(2048) || K(2048) || V(2048), Q/K interleaved
#define QWEN_MOE_HEAD_DIM 128
#define QWEN_MOE_NUM_ATTENTION_HEADS 16
#define QWEN_MOE_NUM_KV_HEADS 16 // MHA: GQA group == 1
#define QWEN_MOE_NUM_PARTIALS 8  // floor(132 / 16)
#define QWEN_MOE_VOCAB_SIZE 151936
#define QWEN_MOE_NUM_EXPERTS 60
#define QWEN_MOE_NUM_EXPERTS_PER_TOK 4
#define QWEN_MOE_MOE_INTERMEDIATE_DIM 1408
#define QWEN_MOE_SHARED_INTERMEDIATE_DIM 5632
#define QWEN_MOE_KV_BLOCK_SIZE 16
#define QWEN_MOE_MATVEC_BLOCK_SIZE 16
// Shared-expert down-projection uses four 1408-wide split-K segments.
#define QWEN_MOE_SPLIT_K_SHARED 1408
// Flattened routed-expert weight index = layer_idx * 60 + physical_expert_id.
#define QWEN_MOE_EXPERT_FLAT_ROWS (QWEN_MOE_NUM_LAYERS * QWEN_MOE_NUM_EXPERTS)

// Barrier tensor: [24, 15, 128] int32.
// row = opcode - 1 (rows 0..13 used, row 14 reserved); slot = dependency slot.
#define QWEN_MOE_BARRIER_ROWS 15
#define QWEN_MOE_BARRIER_SLOTS 128

// `attn_lse_intermediates` is [16, num_partials] with num_partials == 8.  TK
// shared vectors come in multiples of 16, so a whole LSE row is one
// sv_fl<16>; columns 8..15 are outside the tensor and TMA zero-fills them.
#define QWEN_MOE_ROUNDED_NUM_PARTIALS 16

using config = megakernel::default_config;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
template <int _num_layers, int _hidden_dim, int _head_dim,
          int _num_attention_heads, int _num_kv_heads, int _num_partials,
          int _num_experts, int _num_experts_per_tok, int _moe_intermediate_dim,
          int _shared_intermediate_dim, int _qkv_dim, int _kv_block_size,
          int _matvec_block_size, int _sm_count>
struct globals_t {

    constexpr static int num_layers = _num_layers;
    constexpr static int hidden_dim = _hidden_dim;
    constexpr static int head_dim = _head_dim;
    constexpr static int num_attention_heads = _num_attention_heads;
    constexpr static int num_kv_heads = _num_kv_heads;
    constexpr static int num_partials = _num_partials;
    constexpr static int num_experts = _num_experts;
    constexpr static int num_experts_per_tok = _num_experts_per_tok;
    constexpr static int moe_intermediate_dim = _moe_intermediate_dim;
    constexpr static int shared_intermediate_dim = _shared_intermediate_dim;
    constexpr static int qkv_dim = _qkv_dim;
    constexpr static int kv_block_size = _kv_block_size;
    constexpr static int matvec_block_size = _matvec_block_size;
    constexpr static int sm_count = _sm_count;

    // Derived, used by the ops (not part of the ABI).
    constexpr static int expert_flat_rows = num_layers * num_experts;
    constexpr static int split_k_shared = QWEN_MOE_SPLIT_K_SHARED;
    constexpr static int barrier_rows = QWEN_MOE_BARRIER_ROWS;
    constexpr static int barrier_slots = QWEN_MOE_BARRIER_SLOTS;

    static_assert(num_attention_heads == num_kv_heads,
                  "Qwen1.5-MoE is MHA; a GQA ratio != 1 would change op2/op3");
    static_assert(shared_intermediate_dim == 4 * split_k_shared,
                  "op11 is scheduled as exactly 4 split-K segments of 1408");

    using instruction_layout = megakernel::instruction_layout<config>;

    // ---- weight tensor families ------------------------------------------
    // K == hidden_dim (2048): matvec_pipeline lays it out as 4 x st_bf<16,512>.
    using weights_hidden_k_t =
        kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,
                    kittens::st_bf<matvec_block_size, 512>>;
    // K == 1408: 512 + 512 + 384, so BOTH page tile shapes need a descriptor.
    using weights_moe_k_t =
        kittens::gl<kittens::bf16, 1, -1, -1, moe_intermediate_dim,
                    kittens::st_bf<matvec_block_size, 512>,
                    kittens::st_bf<matvec_block_size, 384>>;
    // K == 5632, consumed as 4 split-K segments of 1408 (512 + 512 + 384).
    using weights_shared_k_t =
        kittens::gl<kittens::bf16, 1, -1, -1, shared_intermediate_dim,
                    kittens::st_bf<matvec_block_size, 512>,
                    kittens::st_bf<matvec_block_size, 384>>;

    // RMS scales and other [layers, hidden_dim] row-vector weights.  The whole
    // row is TMA-loaded as one sv_bf<hidden_dim> by rms_matvec_pipeline.
    using norm_weights_t =
        kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim,
                    kittens::sv_bf<hidden_dim>,
                    kittens::sv_bf<matvec_block_size>>;
    // [24, 6144] 2-D (see the qkv_bias note at the bottom of this file).
    using qkv_bias_t = kittens::gl<kittens::bf16, 1, 1, -1, qkv_dim,
                                   kittens::sv_bf<matvec_block_size>>;
    using rope_table_t =
        kittens::gl<float, 1, 1, -1, head_dim, kittens::sv_fl<head_dim>>;

    // ---- activation buffers ----------------------------------------------
    // Every [hidden_dim] bf16 buffer.  sv_bf<hidden_dim> for whole-vector TMA,
    // sv_bf<head_dim> for per-head slices, sv_bf<16> for matvec output tiles.
    using activations_t =
        kittens::gl<kittens::bf16, 1, 1, 1, hidden_dim,
                    kittens::sv_bf<hidden_dim>, kittens::sv_bf<head_dim>,
                    kittens::sv_bf<matvec_block_size>>;
    using shared_silu_out_t =
        kittens::gl<kittens::bf16, 1, 1, 1, shared_intermediate_dim,
                    kittens::sv_bf<matvec_block_size>>;
    // [num_experts_per_tok, moe_intermediate_dim]; row index is the TOP-K SLOT.
    using routed_silu_buffer_t =
        kittens::gl<kittens::bf16, 1, 1, -1, moe_intermediate_dim,
                    kittens::sv_bf<matvec_block_size>>;
    using routed_out_f32_t = kittens::gl<float, 1, 1, 1, hidden_dim,
                                         kittens::sv_fl<matvec_block_size>>;
    using logits_t = kittens::gl<kittens::bf16, 1, 1, 1, -1,
                                 kittens::sv_bf<matvec_block_size>>;

    // KV cache: [layers, seq, kv_heads, head_dim] (batch axis dropped by
    // mk_kernel.py).  axis-1 st descriptor = one 16-position block per head.
    using kv_cache_t =
        kittens::gl<kittens::bf16, -1, -1, -1, head_dim,
                    kittens::sv_bf<matvec_block_size>,
                    kittens::tma::descriptor<
                        kittens::st_bf<kv_block_size, head_dim>, 1>>;

    using attn_out_intermediates_t =
        kittens::gl<float, 1, num_attention_heads, -1, head_dim,
                    kittens::sv_fl<head_dim>>;
    using attn_lse_intermediates_t =
        kittens::gl<float, 1, 1, num_attention_heads, -1,
                    kittens::sv_fl<QWEN_MOE_ROUNDED_NUM_PARTIALS>>;

    // ---- MoE routing scratch (small; plain global loads, no descriptors) ---
    using router_logits_t = kittens::gl<float, 1, 1, 1, num_experts>;
    using expert_indices_t =
        kittens::gl<int64_t, 1, 1, 1, num_experts_per_tok>;
    using expert_weights_t = kittens::gl<float, 1, 1, 1, num_experts_per_tok>;

    // ---- cross-instruction GMEM barrier counters --------------------------
    using barriers_t = kittens::gl<uint, 1, -1, -1, barrier_slots>;

    // =======================================================================
    // MEMBERS — declaration order is the native-call ABI.
    // =======================================================================
    instruction_layout instructions;          // #1  [132, 205, 32]  int32
    // --- activation buffers ---
    activations_t hidden_states;              // #2  [2048]          bf16
    activations_t normed_hidden_states;       // #3  [2048]          bf16
    activations_t q_post_rope;                // #4  [2048]          bf16
    kv_cache_t k_cache;                       // #5  [24, S, 16, 128] bf16
    kv_cache_t v_cache;                       // #6  [24, S, 16, 128] bf16
    activations_t attn_out;                   // #7  [2048]          bf16
    attn_lse_intermediates_t
        attn_lse_intermediates;               // #8  [16, 8]         fp32
    attn_out_intermediates_t
        attn_out_intermediates;               // #9  [16, 8, 128]    fp32
    shared_silu_out_t shared_expert_silu_out; // #10 [5632]          bf16
    activations_t shared_out;                 // #11 [2048]          bf16
    routed_silu_buffer_t routed_silu_buffer;  // #12 [4, 1408]       bf16
    routed_out_f32_t routed_out_f32;          // #13 [2048]          fp32
    router_logits_t router_logits_buf;        // #14 [60]            fp32
    expert_indices_t expert_indices;          // #15 [4]             int64
    expert_weights_t expert_weights;          // #16 [4]             fp32
    logits_t logits;                          // #17 [151936]        bf16
    // --- weights ---
    norm_weights_t attn_norm_weights;         // #18 [24, 2048]      bf16
    weights_hidden_k_t qkv_weights;           // #19 [24, 6144, 2048] bf16
    qkv_bias_t qkv_bias;                      // #20 [24, 6144]      bf16
    rope_table_t rope_cos;                    // #21 [8192, 128]     fp32
    rope_table_t rope_sin;                    // #22 [8192, 128]     fp32
    weights_hidden_k_t o_proj_weights;        // #23 [24, 2048, 2048] bf16
    norm_weights_t mlp_norm_weights;          // #24 [24, 2048]      bf16
    weights_hidden_k_t
        shared_expert_gate_proj_weights;      // #25 [24, 5632, 2048] bf16
    weights_hidden_k_t
        shared_expert_up_proj_weights;        // #26 [24, 5632, 2048] bf16
    weights_shared_k_t
        shared_expert_down_proj_weights;      // #27 [24, 2048, 5632] bf16
    norm_weights_t shared_expert_gate_weights;// #28 [24, 2048]      bf16
    weights_hidden_k_t router_weights;        // #29 [24, 60, 2048]  bf16
    weights_hidden_k_t
        expert_gate_up_proj_weights_flat;     // #30 [1440, 2816, 2048] bf16
    weights_moe_k_t
        expert_down_proj_weights_flat;        // #31 [1440, 2048, 1408] bf16
    norm_weights_t lm_head_norm_weights;      // #32 [2048]          bf16
    weights_hidden_k_t lm_head_weights;       // #33 [151936, 2048]  bf16
    // --- sync + scalars ---
    barriers_t Bar;                           // #34 [24, 15, 128]   int32
    unsigned int pos_id;                      // #35 scalar
    float attn_scale;                         // #36 scalar (1/sqrt(128))
    float rms_norm_eps;                       // #37 scalar (1e-6)

    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

typedef globals_t<QWEN_MOE_NUM_LAYERS, QWEN_MOE_HIDDEN_DIM, QWEN_MOE_HEAD_DIM,
                  QWEN_MOE_NUM_ATTENTION_HEADS, QWEN_MOE_NUM_KV_HEADS,
                  QWEN_MOE_NUM_PARTIALS, QWEN_MOE_NUM_EXPERTS,
                  QWEN_MOE_NUM_EXPERTS_PER_TOK, QWEN_MOE_MOE_INTERMEDIATE_DIM,
                  QWEN_MOE_SHARED_INTERMEDIATE_DIM, QWEN_MOE_QKV_DIM,
                  QWEN_MOE_KV_BLOCK_SIZE, QWEN_MOE_MATVEC_BLOCK_SIZE,
#ifndef KITTENS_BLACKWELL
                  H100_SM_COUNT>
#else
                  B200_SM_COUNT>
#endif
    qwen_moe_globals;

// ============================================================================
// `qkv_bias` uses the 2-D ABI shape [24, 6144].  A `coord<sv_bf<16>>`
// multiplies its final component by the vector length, so
//
//     kittens::tma::load_async(bias_smem, g.qkv_bias, {layer_idx, block_idx}, sem)
//
// resolves to element offset `block_idx * 16` in row `layer_idx`.
// The generated descriptor is legal (gmem_stride[0] = 6144 * 2 = 12288 B,
// a multiple of 16; each 16-element box is 32 B and 16 B-aligned).
// ============================================================================
