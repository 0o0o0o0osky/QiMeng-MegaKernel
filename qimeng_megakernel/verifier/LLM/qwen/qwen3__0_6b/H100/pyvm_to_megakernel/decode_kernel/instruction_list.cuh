#pragma once

#include "globals.cuh"
#include "barrier.cuh"

// Instruction opcodes (aligned with decode_schedule)
#define OPCODE_RMS_QKV_GEMV        1
#define OPCODE_ATTENTION_PARTIAL   2
#define OPCODE_DECODE_ATTENTION_SPLIT_KV 2
#define OPCODE_ATTENTION_REDUCTION 3
#define OPCODE_O_PROJ_RESIDUAL     4
#define OPCODE_RMS_UPGATE_SILU     5
#define OPCODE_DOWN_PROJ_RESIDUAL  6
#define OPCODE_RMS_LM_HEAD         7

// Bar[layer, row, slot] uses row == opcode - 1. Centralise this opcode mapping
// so CUDA call sites cannot silently drift from the Python instruction accounting.
#define BARRIER_ROW(opcode) ((opcode) - 1)

template <typename config = config, typename globals = qwen3_0_6b_globals>
struct attention_partial;

template <typename config = config, typename globals = qwen3_0_6b_globals>
struct attention_reduction;

template <typename config = config, typename globals = qwen3_0_6b_globals>
struct rms_qkv_raw;

template <typename config = config, typename globals = qwen3_0_6b_globals>
struct downproj;

template <typename config = config, typename globals = qwen3_0_6b_globals>
struct o_proj;

template <typename config = config, typename globals = qwen3_0_6b_globals>
struct rms_upgate_silu;

template <typename config = config, typename globals = qwen3_0_6b_globals>
struct rms_lm_head;
