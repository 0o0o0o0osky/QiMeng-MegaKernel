#pragma once

namespace pi0 {

#define OPCODE_GEMMA_RMSN_ATTN     1
#define OPCODE_GEMMA_RMSN_FFN      2
#define OPCODE_GEMMA_QKV_ROPE      3
#define OPCODE_GEMMA_ATTN_PARTIAL  4
#define OPCODE_GEMMA_ATTN_REDUCE   5
#define OPCODE_GEMMA_OPROJ         6
#define OPCODE_GEMMA_GATE          7   // GEMM(row_scale) -> gate_buf
#define OPCODE_GEMMA_UP_MUL        8   // GEMM(row_scale) x gelu(gate) -> up_buf
#define OPCODE_GEMMA_DOWN          9

#define OPCODE_GEMMA_DOWN_SPLITK  10

#define BARRIER_ROW(opcode) ((opcode) - 1)

static constexpr int GEMMA_NUM_BARRIER_ROWS = 11;
static constexpr int GEMMA_BARRIER_SLOTS = 64;

}  // namespace pi0
