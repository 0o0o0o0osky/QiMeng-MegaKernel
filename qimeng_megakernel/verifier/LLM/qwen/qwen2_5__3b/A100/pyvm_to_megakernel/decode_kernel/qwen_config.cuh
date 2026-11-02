#pragma once
#include <cstdint>

namespace megakernel {
    struct QwenConfig {
        static constexpr int32_t NUM_LAYERS    = 36;
        static constexpr int32_t HIDDEN_DIM    = 2048;
        static constexpr int32_t INTERM_DIM    = 11008;
        static constexpr int32_t HEAD_DIM      = 128;
        static constexpr int32_t NUM_Q_HEADS   = 16;
        static constexpr int32_t NUM_KV_HEADS  = 2;
        static constexpr int32_t NUM_HEADS     = NUM_Q_HEADS + NUM_KV_HEADS * 2;
        static constexpr int32_t GQA_GROUP     = NUM_Q_HEADS / NUM_KV_HEADS;
        static constexpr int32_t ATTN_REDUCTION_SIZE = 4;
        static constexpr int32_t QKV_OUT_DIM   = NUM_HEADS * HEAD_DIM;
        static constexpr int32_t QKV_NUM_BLOCKS = QKV_OUT_DIM / 16;
        static constexpr int32_t VOCAB_SIZE    = 151936;
    };
}
