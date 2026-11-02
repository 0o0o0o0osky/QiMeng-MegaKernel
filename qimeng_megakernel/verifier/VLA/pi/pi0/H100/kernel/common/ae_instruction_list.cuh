#pragma once

namespace pi0 {

#define OPCODE_AE_RMSN_ATTN     1
#define OPCODE_AE_RMSN_FFN      2
#define OPCODE_AE_QKV_ROPE      3
#define OPCODE_AE_ATTN_PARTIAL  4
#define OPCODE_AE_ATTN_REDUCE   5
#define OPCODE_AE_OPROJ         6
#define OPCODE_AE_GATE          7
#define OPCODE_AE_UP_MUL        8   // split-K GEMM(row_scale) * gelu(gate), store_add -> ffn_buf
#define OPCODE_AE_DOWN          9
#define OPCODE_AE_FFN_DUAL     10

#define OPCODE_AE_ACT_IN       11   // in_proj + b_time[step] + silu   -> act_h
#define OPCODE_AE_ACT_MLP      12
#define OPCODE_AE_RMSN_OUT     13
#define OPCODE_AE_ACT_OUT      14

#define BARRIER_ROW(opcode) ((opcode) - 1)

static constexpr int AE_NUM_BARRIER_ROWS = 14;
static constexpr int AE_BARRIER_SLOTS    = 64;

}  // namespace pi0
