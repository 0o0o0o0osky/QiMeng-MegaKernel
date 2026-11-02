#pragma once

#include <cstdint>

namespace megakernel {
    struct LlamaConfig {
        static constexpr int32_t NUM_LAYERS = 28;
        static constexpr int32_t HIDDEN_DIM = 3072;
        static constexpr int32_t INTERM_DIM = 8192;
        static constexpr int32_t HEAD_DIM = 128;
        static constexpr int32_t NUM_Q_HEADS = 24;
        static constexpr int32_t NUM_KV_HEADS = 8;
        static constexpr int32_t NUM_HEADS = NUM_Q_HEADS + NUM_KV_HEADS * 2;  // 40
        static constexpr int32_t QKV_OUT_DIM = NUM_HEADS * HEAD_DIM;          // 5120
        static constexpr int32_t ATTN_REDUCTION_SIZE = 3;
        static constexpr int32_t VOCAB_SIZE = 128256;
    };
}
