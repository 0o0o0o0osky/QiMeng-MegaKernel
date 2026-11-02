#pragma once
#include <cstdint>

namespace megakernel {
    struct QwenMoeConfig {
        // Qwen1.5-MoE-A2.7B: hidden=2048, layers=24, heads=16Q+16KV, head_dim=128
        static constexpr int32_t NUM_LAYERS    = 24;
        static constexpr int32_t HIDDEN_DIM    = 2048;
        static constexpr int32_t INTERM_DIM    = 5632;  // shared expert intermediate size
        static constexpr int32_t HEAD_DIM      = 128;
        static constexpr int32_t NUM_Q_HEADS   = 16;
        static constexpr int32_t NUM_KV_HEADS  = 16;
        static constexpr int32_t GQA_GROUP     = NUM_Q_HEADS / NUM_KV_HEADS;      // 1
        static constexpr int32_t ATTN_REDUCTION_SIZE = 4;
        // Actual head-count derived dims (QKV projection, etc.)
        static constexpr int32_t NUM_HEADS     = NUM_Q_HEADS + NUM_KV_HEADS * 2;  // 48
        static constexpr int32_t QKV_OUT_DIM   = NUM_HEADS * HEAD_DIM;            // 6144
        static constexpr int32_t QKV_NUM_BLOCKS = QKV_OUT_DIM / 16;               // 384
        static constexpr int32_t VOCAB_SIZE    = 151936;
        // Router (opcode 8): 60 experts, top-4 per token
        static constexpr int32_t NUM_EXPERTS        = 60;
        static constexpr int32_t NUM_EXPERTS_PER_TOK = 4;
        // For layer_idx > 0: wait for prev layer MoE combine (opcode 14), barrier row 13.
        // Static tile-wise combine commits one count per hidden tile (128 total).
        static constexpr int32_t PREV_LAYER_BARRIER_ROW = 13;
        static constexpr int32_t PREV_LAYER_EXPECTED   = 128;
        // Tiling
        static constexpr int32_t TILE_N_GRANULARITY = 16;
        static constexpr int32_t NUM_HIDDEN_BLOCKS  = HIDDEN_DIM / TILE_N_GRANULARITY;  // 128
        // Barrier layout: barriers[L, 15, BARRIER_SLOTS]
        //   Attention uses slots [0, NUM_ATTN_BARRIER_SLOTS) = [0, 48)
        //   Downproj completion (op11/op13) uses slots [0, NUM_HIDDEN_BLOCKS) = [0, 128)
        //   BARRIER_SLOTS = max(NUM_ATTN_BARRIER_SLOTS, NUM_HIDDEN_BLOCKS) = 128
        static constexpr int32_t NUM_ATTN_BARRIER_SLOTS = NUM_HEADS;               // 48
        static constexpr int32_t BARRIER_SLOTS          = NUM_HIDDEN_BLOCKS;       // 128
        static constexpr int32_t NUM_OPCODE_ROWS        = 15;
        // Shared expert (op10, op11): split-K 4
        static constexpr int32_t NUM_DOWNPROJ_SPLIT_K_SHARED      = 4;
        static constexpr int32_t SPLIT_K_GRANULARITY_SHARED       = INTERM_DIM / NUM_DOWNPROJ_SPLIT_K_SHARED;  // 1408
        static constexpr int32_t NUM_SHARED_UPGATE_TILES          = INTERM_DIM / TILE_N_GRANULARITY;           // 352
        static constexpr int32_t NUM_SHARED_DOWNPROJ_ACCUM_BLOCKS = NUM_DOWNPROJ_SPLIT_K_SHARED * NUM_HIDDEN_BLOCKS;  // 512
        // Routed experts (op12, op13): moe_intermediate=1408, 4 active experts per token
        static constexpr int32_t MOE_INTERM_DIM              = 1408;
        static constexpr int32_t NUM_MOE_INTERMEDIATE_BLOCKS = MOE_INTERM_DIM / TILE_N_GRANULARITY;        // 88
        static constexpr int32_t NUM_ROUTED_UPGATE_BLOCKS    = NUM_EXPERTS_PER_TOK * NUM_MOE_INTERMEDIATE_BLOCKS; // 352
        static constexpr int32_t NUM_ROUTED_DOWNPROJ_BLOCKS  = NUM_EXPERTS_PER_TOK * NUM_HIDDEN_BLOCKS;    // 512
    };
}
