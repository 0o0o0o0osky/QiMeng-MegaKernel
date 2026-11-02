#pragma once
#include <cstdint>

namespace megakernel {
    struct QwenConfig {
        // Qwen3-0.6B: hidden=1024, layers=28, heads=16Q+8KV, inter=3072, head_dim=128
        static constexpr int32_t NUM_LAYERS    = 28;
        static constexpr int32_t HIDDEN_DIM    = 1024;
        static constexpr int32_t INTERM_DIM    = 3072;
        static constexpr int32_t HEAD_DIM      = 128;
        static constexpr int32_t NUM_Q_HEADS   = 16;
        static constexpr int32_t NUM_KV_HEADS  = 8;
        static constexpr int32_t NUM_HEADS     = NUM_Q_HEADS + NUM_KV_HEADS * 2;  // 32
        static constexpr int32_t ATTN_OUT_DIM  = NUM_Q_HEADS * HEAD_DIM;          // 2048
        static constexpr int32_t QKV_OUT_DIM   = NUM_HEADS * HEAD_DIM;            // 4096
        static constexpr int32_t VOCAB_SIZE    = 151936;
        static constexpr int32_t GQA_GROUP     = NUM_Q_HEADS / NUM_KV_HEADS;      // 2
        static constexpr int32_t ATTN_REDUCTION_SIZE = 4;
    };
}
