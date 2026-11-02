#pragma once

#include <cstdint>

namespace pi0_a100 {

enum SiglipOpcode : int32_t {
    SIGLIP_LN_ATTN = 1,
    SIGLIP_LN_FFN = 2,
    SIGLIP_QKV = 3,
    SIGLIP_ATTN = 4,
    SIGLIP_OPROJ = 5,
    SIGLIP_FC1 = 6,
    SIGLIP_FC2 = 7,
    SIGLIP_PATCH = 8,
    SIGLIP_FINAL_NORM = 9,
    SIGLIP_PROJECTOR = 10,
};

enum GemmaOpcode : int32_t {
    GEMMA_RMSN_ATTN = 1,
    GEMMA_RMSN_FFN = 2,
    GEMMA_QKV_ROPE = 3,
    GEMMA_ATTN_QK = 4,
    GEMMA_ATTN_SOFTMAX = 5,
    GEMMA_OPROJ = 6,
    GEMMA_GATE = 7,
    GEMMA_UP_MUL = 8,
    GEMMA_DOWN = 9,
    GEMMA_DOWN_SPLITK = 10,
    // Materialized attention's third phase.  Appended rather than slotted
    // between 5 and 6 so the six opcodes after it keep their numbers, which
    // keeps `gemma::kBarrierRows` at 11: rows 0..10 for opcodes 1..11.
    GEMMA_ATTN_PV = 11,
};

enum AeOpcode : int32_t {
    AE_RMSN_ATTN = 1,
    AE_RMSN_FFN = 2,
    AE_QKV_ROPE = 3,
    AE_ATTN_PARTIAL = 4,
    AE_ATTN_REDUCE = 5,
    AE_OPROJ = 6,
    AE_GATE = 7,
    AE_UP_MUL = 8,
    AE_DOWN = 9,
    AE_FFN_DUAL = 10,
    AE_ACT_IN = 11,
    AE_ACT_MLP = 12,
    AE_RMSN_OUT = 13,
    AE_ACT_OUT = 14,
    AE_RESIDUAL_REDUCE = 15,
};

__host__ __device__ constexpr int barrier_row(int opcode) { return opcode - 1; }

}  // namespace pi0_a100
