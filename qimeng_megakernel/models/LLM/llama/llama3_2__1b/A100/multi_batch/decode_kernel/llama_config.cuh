#pragma once

#include <cstdint>

namespace megakernel {
    struct LlamaConfig {
        static constexpr int32_t NUM_LAYERS = 16;
        static constexpr int32_t HIDDEN_DIM = 2048;
        static constexpr int32_t INTERM_DIM = 8192;
        static constexpr int32_t HEAD_DIM = 64;
        static constexpr int32_t NUM_Q_HEADS = 32;
        static constexpr int32_t NUM_KV_HEADS = 8;
        static constexpr int32_t NUM_HEADS = NUM_Q_HEADS + NUM_KV_HEADS * 2;
        static constexpr int32_t QKV_OUT_DIM = NUM_HEADS * HEAD_DIM;
        static constexpr int32_t VOCAB_SIZE = 128256;
    };
}
