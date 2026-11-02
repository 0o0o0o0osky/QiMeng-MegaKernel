#pragma once

namespace llama_mb {

constexpr int OPCODE_QKV = 1;
constexpr int OPCODE_ATTENTION = 2;
constexpr int OPCODE_ATTENTION_REDUCTION = 3; // ABI-reserved; direct attention does not emit it.
constexpr int OPCODE_OPROJ = 4;
constexpr int OPCODE_UPGATE = 5;
constexpr int OPCODE_DOWN = 6;
constexpr int OPCODE_LMHEAD = 7;
constexpr int OPCODE_RMS_ATTN = 8;
constexpr int OPCODE_RMS_MLP = 9;
constexpr int OPCODE_RMS_LMHEAD = 10;
constexpr int OPCODE_DOWN_REDUCE = 11;

constexpr int BAR_QKV = 0;
constexpr int BAR_ATTENTION = 1;
constexpr int BAR_OPROJ = 2;
constexpr int BAR_UPGATE = 3;
constexpr int BAR_DOWN = 4;
constexpr int BAR_LAYER_READY = 5;
constexpr int BAR_RMS_ATTN = 7;
constexpr int BAR_RMS_MLP = 8;
constexpr int BAR_RMS_LMHEAD = 9;

template <typename Config> struct range_instruction {
    int layer;
    int begin;
    int end;
    int stride;
    int dep_layer;
    int dep_row;
    int dep_expected;
    int dep_slot;
    int publish_row;
    int publish_slot;
    int batch_size;

    __device__ explicit range_instruction(
        typename Config::instruction_t &instruction)
        : layer(instruction[1]), begin(instruction[2]),
          end(instruction[3]), stride(instruction[4]),
          dep_layer(instruction[5]), dep_row(instruction[6]),
          dep_expected(instruction[7]), dep_slot(instruction[8]),
          publish_row(instruction[9]), publish_slot(instruction[10]),
          batch_size(instruction[11]) {}

    template <typename State>
    __device__ explicit range_instruction(State &s)
        : range_instruction(s.instruction()) {}
};

} // namespace llama_mb
